# /// script
# dependencies = [
#   "gymnasium[box2d]>=1.0.0",
#   "stable-baselines3>=2.3.0",
#   "torch>=2.2.0",
#   "numpy>=1.24.0",
#   "pygame>=2.5.0",
#   "matplotlib>=3.8.0",
# ]
# [tool.uv]
# exclude-newer = "30 days"
# ///

import argparse
import concurrent.futures
import os
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from utils import (
    TrainingStatsCallback,
    run_grid_test,
    safe_name,
    save_training_graph,
)


SLEEP_VARIANT = "sleep"
NOSLEEP_VARIANT = "nosleep"
VARIANTS = (SLEEP_VARIANT, NOSLEEP_VARIANT)
TRAINING_SEED = 2028
MODEL_DIR = Path("models_multi")
GRAPH_DIR = Path("training_graphs_multi")


@dataclass(frozen=True)
class EnvSpec:
    key: str
    env_id: str
    title: str
    max_sleep: int
    time_limit: int
    timesteps: int
    action_labels: tuple[str, ...]
    decision_cost: float = 0.0
    oversleep_penalty: float = 0.0


ENV_SPECS = (
    EnvSpec(
        key="lunar",
        env_id="LunarLander-v3",
        title="LunarLander",
        max_sleep=4,
        time_limit=1000,
        timesteps=300_000,
        action_labels=("noop", "left", "main", "right"),
    ),
    EnvSpec(
        key="mountaincar",
        env_id="MountainCar-v0",
        title="MountainCar",
        max_sleep=2,
        time_limit=200,
        timesteps=300_000,
        action_labels=("left", "idle", "right"),
    ),
)


class SleepPlanningWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_sleep: int, decision_cost: float, oversleep_penalty: float):
        super().__init__(env)
        assert isinstance(self.env.action_space, gym.spaces.Discrete), "Base action space must be Discrete."
        self.max_sleep = max_sleep
        self.decision_cost = decision_cost
        self.oversleep_penalty = oversleep_penalty
        self.action_space = gym.spaces.MultiDiscrete([self.env.action_space.n, self.max_sleep])

    def step(self, action):
        core_action = int(action[0])
        sleep_steps = int(action[1]) + 1
        total_reward = -self.decision_cost
        terminated, truncated = False, False
        info = {}
        actual_steps = 0

        for _ in range(sleep_steps):
            obs, reward, terminated, truncated, info = self.env.step(core_action)
            total_reward += reward
            actual_steps += 1
            if terminated or truncated:
                break

        if self.oversleep_penalty and (terminated or truncated) and actual_steps < sleep_steps:
            info["overslept_by"] = sleep_steps - actual_steps
            total_reward -= info["overslept_by"] * self.oversleep_penalty
        info["core_action"] = core_action
        info["intended_sleep"] = sleep_steps
        info["steps_slept"] = actual_steps
        return obs, total_reward, terminated, truncated, info


class DecisionCostWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, decision_cost: float):
        super().__init__(env)
        self.decision_cost = decision_cost

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.decision_cost
        info["core_action"] = int(action)
        info["intended_sleep"] = 1
        info["steps_slept"] = 1
        return obs, reward, terminated, truncated, info


class FlatVectorWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(self.env.observation_space, gym.spaces.Box), "Observation space must be Box."
        low = np.asarray(self.env.observation_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.env.observation_space.high, dtype=np.float32).reshape(-1)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten(obs), reward, terminated, truncated, info

    def _flatten(self, obs):
        return np.asarray(obs, dtype=np.float32).reshape(-1)


class EpisodeInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, time_limit: int):
        super().__init__(env)
        self.time_limit = time_limit
        self.elapsed_steps = 0

    def reset(self, **kwargs):
        self.elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += int(info.get("steps_slept", 1))
        info["elapsed_steps"] = self.elapsed_steps
        info["time_limit"] = self.time_limit
        info["time_remaining"] = max(0, self.time_limit - self.elapsed_steps)
        return obs, reward, terminated, truncated, info


class LunarRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, safe_speed: float = 0.35, speed_penalty: float = 10.0):
        super().__init__(env)
        self.safe_speed = safe_speed
        self.speed_penalty = speed_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base_reward = reward
        if terminated:
            landing_speed = float(np.linalg.norm(obs[2:4]))
            speed_excess = max(0.0, landing_speed - self.safe_speed)
            safe_landing = bool(base_reward > 0.0 and obs[6] > 0.5 and obs[7] > 0.5)
            if safe_landing:
                reward -= self.speed_penalty * speed_excess
            info["landing_speed"] = landing_speed
            info["safe_landing"] = safe_landing
        return obs, reward, terminated, truncated, info


class MountainCarRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        position = float(obs[0])
        velocity = float(obs[1])
        force = int(action) - 1

        reward += 100.0 * force * velocity
        reward += 0.25 * (position + 1.2)
        if terminated:
            reward += 100.0

        info["position"] = position
        info["velocity"] = velocity
        info["force_alignment"] = force * velocity
        info["success"] = bool(terminated)
        return obs, reward, terminated, truncated, info


def spec_by_key(key: str) -> EnvSpec:
    return next(spec for spec in ENV_SPECS if spec.key == key)


def model_path(spec: EnvSpec, variant: str, run_name: str) -> Path:
    return MODEL_DIR / f"{spec.key}_{safe_name(run_name)}_{variant}.zip"


def graph_path(spec: EnvSpec, variant: str, run_name: str) -> Path:
    return GRAPH_DIR / f"{spec.key}_{safe_name(run_name)}_{variant}_training.png"


def make_base_env(spec: EnvSpec, render_mode: str | None):
    env = gym.make(spec.env_id, render_mode=render_mode)
    if spec.key == "lunar":
        env = LunarRewardWrapper(env)
    elif spec.key == "mountaincar":
        env = MountainCarRewardWrapper(env)
    env = FlatVectorWrapper(env)
    assert isinstance(env.action_space, gym.spaces.Discrete), "Only discrete-action envs are supported."
    return env


def make_environment(spec: EnvSpec, use_sleep: bool, render_mode: str | None = None):
    env = make_base_env(spec, render_mode)
    if use_sleep:
        env = SleepPlanningWrapper(env, spec.max_sleep, spec.decision_cost, spec.oversleep_penalty)
    else:
        env = DecisionCostWrapper(env, spec.decision_cost)
    return EpisodeInfoWrapper(env, spec.time_limit)


def train_one(spec_key: str, variant: str, run_name: str, timesteps_override: int | None, n_envs: int):
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    spec = spec_by_key(spec_key)
    use_sleep = variant == SLEEP_VARIANT
    timesteps = timesteps_override or spec.timesteps
    path = model_path(spec, variant, run_name)
    path.parent.mkdir(exist_ok=True)

    print(
        f"[TRAIN START] env={spec.title} variant={variant} timesteps={timesteps} "
        f"n_envs={n_envs} path={path}",
        flush=True,
    )
    vec_env = make_vec_env(
        lambda: make_environment(spec, use_sleep),
        n_envs=n_envs,
        seed=TRAINING_SEED,
    )
    try:
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=512, seed=TRAINING_SEED, verbose=0)
        callback = TrainingStatsCallback()
        model.learn(total_timesteps=timesteps, callback=callback)
        model.save(path.with_suffix(""))
        plot_path = save_training_graph(
            graph_path(spec, variant, run_name),
            f"{spec.title} {variant}",
            timesteps,
            callback.episode_returns,
            callback.episode_lengths,
            callback.episode_env_steps,
            callback.episode_timesteps,
        )
    finally:
        vec_env.close()
    return f"[TRAIN DONE] env={spec.title} variant={variant} path={path} graph={plot_path}"


def train_all(run_name: str, timesteps_override: int | None):
    jobs = [(spec.key, variant) for spec in ENV_SPECS for variant in VARIANTS]
    cpu_count = os.cpu_count() or 4
    workers = min(len(jobs), max(1, cpu_count // 2))
    n_envs = max(1, min(4, cpu_count // max(1, workers)))
    print(f"[TRAIN] jobs={len(jobs)} workers={workers} vec_envs_per_job={n_envs}", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(train_one, spec_key, variant, run_name, timesteps_override, n_envs)
            for spec_key, variant in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


@dataclass
class CellState:
    spec: EnvSpec
    variant: str
    env: gym.Env
    model: PPO
    obs: np.ndarray
    episode: int = 1
    score: float = 0.0
    decisions: int = 0
    env_steps: int = 0
    last_reward: float = 0.0
    last_action: str = "-"
    last_sleep: str = "-"
    wins: int = 0
    total_score: float = 0.0
    last_outcome: str = "running"
    done: bool = False

    @property
    def completed_runs(self) -> int:
        return self.episode if self.done else self.episode - 1

    @property
    def avg_return(self) -> float | None:
        completed = self.completed_runs
        return None if completed == 0 else self.total_score / completed

    @property
    def action_summary(self) -> str:
        if self.last_action == "-":
            return "-"
        label = self.last_action.split(":", 1)[1]
        if self.variant != SLEEP_VARIANT:
            return label
        actual, planned = self.last_sleep.split("/", 1)
        return f"{label} x{planned}" if actual == planned else f"{label} x{actual}/{planned}"

    def start_next_episode(self, seed: int):
        self.episode += 1
        self.score = 0.0
        self.decisions = 0
        self.env_steps = 0
        self.last_reward = 0.0
        self.last_action = "-"
        self.last_sleep = "-"
        self.last_outcome = "running"
        self.done = False
        self.obs, _ = self.env.reset(seed=seed)

    def step(self):
        if self.done:
            return

        action, _ = self.model.predict(self.obs, deterministic=True)
        env_action = np.asarray(action, dtype=np.int64) if self.variant == SLEEP_VARIANT else int(np.asarray(action).item())
        self.obs, reward, terminated, truncated, info = self.env.step(env_action)
        core_action = int(info.get("core_action", env_action[0] if self.variant == SLEEP_VARIANT else env_action))
        action_label = self.spec.action_labels[core_action] if core_action < len(self.spec.action_labels) else str(core_action)

        self.last_reward = float(reward)
        self.score += self.last_reward
        self.decisions += 1
        self.env_steps = int(info.get("elapsed_steps", self.env_steps + 1))
        self.last_action = f"{core_action}:{action_label}"
        self.last_sleep = f"{info.get('steps_slept', 1)}/{info.get('intended_sleep', 1)}"
        self.done = bool(terminated or truncated)
        if self.done:
            won = episode_won(info)
            self.wins += int(won)
            self.total_score += self.score
            self.last_outcome = "win" if won else ("timeout" if truncated else "loss")
            print(
                f"[EPISODE] env={self.spec.title} variant={self.variant} run={self.episode} "
                f"outcome={self.last_outcome} wins={self.wins}/{self.episode} "
                f"return={self.score:.2f} avg_return={self.avg_return:.2f} decisions={self.decisions} env_steps={self.env_steps}",
                flush=True,
            )

    def render_frame(self):
        frame = self.env.render()
        if isinstance(frame, list):
            frame = frame[-1]
        return np.asarray(frame, dtype=np.uint8)

    def close(self):
        self.env.close()


def load_test_cells(run_name: str, episodes: int) -> list[CellState]:
    missing = [str(model_path(spec, variant, run_name)) for spec in ENV_SPECS for variant in VARIANTS if not model_path(spec, variant, run_name).exists()]
    if missing:
        raise FileNotFoundError("Missing trained models:\n" + "\n".join(missing))

    cells = []
    for row, spec in enumerate(ENV_SPECS):
        for variant in VARIANTS:
            env = make_environment(spec, variant == SLEEP_VARIANT, render_mode="rgb_array")
            obs, _ = env.reset(seed=1000 + row)
            cells.append(CellState(spec=spec, variant=variant, env=env, model=PPO.load(model_path(spec, variant, run_name)), obs=obs))
    print(f"[TEST] visual grid episodes={episodes} run={run_name}", flush=True)
    return cells


def episode_won(info: dict) -> bool:
    if "success" in info:
        return bool(info["success"])
    if "safe_landing" in info:
        return bool(info["safe_landing"])
    if "crashed" in info:
        return not bool(info["crashed"])
    return False


def catch_up_to_timestep(cells: list[CellState], target_step: int):
    for cell in cells:
        while not cell.done and cell.env_steps < target_step:
            cell.step()


def step_synced_pair(cells: list[CellState], episodes: int, seed_base: int):
    if all(cell.done for cell in cells):
        if all(cell.episode >= episodes for cell in cells):
            return
        next_episode = max(cell.episode for cell in cells) + 1
        for cell in cells:
            cell.start_next_episode(seed_base + next_episode)
        return

    if any(cell.done for cell in cells):
        for cell in cells:
            if not cell.done:
                cell.step()
        return

    for cell in cells:
        cell.step()
    catch_up_to_timestep(cells, max(cell.env_steps for cell in cells))


def main():
    parser = argparse.ArgumentParser(description="Train/test sleep vs nosleep PPO across flat discrete envs")
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--run-name", default="multi")
    parser.add_argument("--timesteps", type=int, default=None, help="Override per-env default training timesteps")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "train":
        train_all(args.run_name, args.timesteps)
    else:
        cells = load_test_cells(args.run_name, args.episodes)
        run_grid_test(cells, ENV_SPECS, VARIANTS, args.episodes, step_synced_pair, "Macro-action RL: sleep vs nosleep")


if __name__ == "__main__":
    main()
