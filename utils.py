import re
import time

import numpy as np
import pygame
from stable_baselines3.common.callbacks import BaseCallback


BG_COLOR = (12, 12, 13)
TEXT_COLOR = (232, 233, 237)
MUTED_COLOR = (151, 154, 162)
FRAME_BORDER = (68, 70, 78)
WIN_COLOR = (108, 190, 134)
LOSS_COLOR = (210, 95, 88)
TIMEOUT_COLOR = (214, 166, 82)
RUN_COLOR = (164, 170, 184)

LEFT_MARGIN = 16
TOP_GUTTER = 34
COL_W = 430
CELL_H = 314
GAP = 14
IMAGE_H = 230


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "run"


def moving_average(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=np.float32)
    data = np.asarray(values, dtype=np.float32)
    if len(data) < window:
        return data
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(data, kernel, mode="valid")


class TrainingStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_env_steps: list[int] = []
        self.episode_timesteps: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue
            self.episode_returns.append(float(episode["r"]))
            self.episode_lengths.append(int(episode["l"]))
            self.episode_env_steps.append(int(info.get("elapsed_steps", episode["l"])))
            self.episode_timesteps.append(self.num_timesteps)
        return True


def save_training_graph(path, title: str, timesteps: int, episode_returns, episode_lengths, episode_env_steps, episode_timesteps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(exist_ok=True)
    episodes = np.arange(1, len(episode_returns) + 1)
    window = min(20, max(1, len(episode_returns)))
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    fig.suptitle(f"{title} training ({timesteps:,} timesteps)")

    if len(episodes) == 0:
        for axis in axes:
            axis.axis("off")
        axes[1].text(0.5, 0.5, "No completed episodes recorded during training", ha="center", va="center")
    else:
        return_ma = moving_average(episode_returns, window)
        length_ma = moving_average(episode_lengths, window)
        env_steps_ma = moving_average(episode_env_steps, window)
        ma_episodes = np.arange(window, len(episode_returns) + 1) if len(episode_returns) >= window else episodes

        axes[0].plot(episodes, episode_returns, color="#8fb3ff", alpha=0.35, linewidth=1, label="episode return")
        axes[0].plot(ma_episodes, return_ma, color="#2457c5", linewidth=2, label=f"{window}-episode average")
        axes[0].set_ylabel("Return")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(episodes, episode_env_steps, color="#f0aa5b", alpha=0.4, linewidth=1, label="env steps")
        axes[1].plot(ma_episodes, env_steps_ma, color="#c66a00", linewidth=2, label=f"env steps {window}-episode average")
        axes[1].plot(episodes, episode_lengths, color="#9f8bd3", alpha=0.45, linewidth=1, label="decisions")
        axes[1].plot(ma_episodes, length_ma, color="#5f3aa0", linewidth=2, label=f"decisions {window}-episode average")
        axes[1].set_ylabel("Episode steps")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(episode_timesteps, episode_returns, color="#65a879", linewidth=1.5)
        axes[2].set_xlabel("Training timesteps")
        axes[2].set_ylabel("Return")
        axes[2].grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def grid_screen_size(rows: int) -> tuple[int, int]:
    return LEFT_MARGIN * 2 + COL_W * 2 + GAP, TOP_GUTTER + CELL_H * rows + 10


def make_grid_fonts():
    return {
        "column": pygame.font.SysFont("Menlo", 12) or pygame.font.SysFont(None, 12),
        "label": pygame.font.SysFont("Menlo", 9, bold=True) or pygame.font.SysFont(None, 9, bold=True),
        "value": pygame.font.SysFont("Menlo", 13) or pygame.font.SysFont(None, 13),
    }


def render_fit(surface, font, text: str, color, rect: pygame.Rect, align: str = "left"):
    fitted = text
    text_surf = font.render(fitted, True, color)
    while text_surf.get_width() > rect.width and len(fitted) > 3:
        fitted = fitted[:-4].rstrip() + "..."
        text_surf = font.render(fitted, True, color)
    if align == "center":
        x = rect.x + (rect.width - text_surf.get_width()) // 2
    elif align == "right":
        x = rect.right - text_surf.get_width()
    else:
        x = rect.x
    surface.blit(text_surf, (x, rect.y))


def draw_caption_item(surface, label_font, value_font, x: int, y: int, width: int, label: str, value: str, value_color=TEXT_COLOR):
    render_fit(surface, label_font, label.upper(), MUTED_COLOR, pygame.Rect(x, y, width - 8, 12))
    render_fit(surface, value_font, value, value_color, pygame.Rect(x, y + 13, width - 8, 18))


def outcome_color(outcome: str):
    if outcome == "win":
        return WIN_COLOR
    if outcome == "loss":
        return LOSS_COLOR
    if outcome == "timeout":
        return TIMEOUT_COLOR
    return RUN_COLOR


def draw_grid(screen, cells, episodes: int, fonts):
    screen.fill(BG_COLOR)

    for col, title in enumerate(("sleep", "no sleep")):
        x = LEFT_MARGIN + col * (COL_W + GAP)
        render_fit(screen, fonts["column"], title, MUTED_COLOR, pygame.Rect(x, 12, COL_W, 18), align="center")

    for idx, cell in enumerate(cells):
        row = idx // 2
        col = idx % 2
        x = LEFT_MARGIN + col * (COL_W + GAP)
        y = TOP_GUTTER + row * CELL_H

        frame = cell.render_frame()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        image_rect = pygame.Rect(x, y, COL_W, IMAGE_H)
        surf = pygame.transform.smoothscale(surf, image_rect.size)
        pygame.draw.rect(screen, (3, 3, 4), image_rect)
        screen.blit(surf, image_rect.topleft)
        pygame.draw.rect(screen, FRAME_BORDER, image_rect, width=1)

        caption_y = image_rect.bottom + 12
        completed = cell.completed_runs
        wins = f"{cell.wins}/{completed}" if completed else "0/0"
        avg_return = cell.avg_return
        avg_text = "-" if avg_return is None else f"{avg_return:.1f}"
        avg_color = RUN_COLOR if avg_return is None else (WIN_COLOR if avg_return >= 0 else LOSS_COLOR)
        value_color = WIN_COLOR if cell.done and cell.last_outcome == "win" else TEXT_COLOR
        item_w = COL_W // 4

        draw_caption_item(screen, fonts["label"], fonts["value"], x, caption_y, item_w, "run", f"{cell.episode}/{episodes}", value_color)
        draw_caption_item(screen, fonts["label"], fonts["value"], x + item_w, caption_y, item_w, "won", wins, WIN_COLOR)
        draw_caption_item(screen, fonts["label"], fonts["value"], x + item_w * 2, caption_y, item_w, "return", f"{cell.score:.1f}", value_color)
        draw_caption_item(screen, fonts["label"], fonts["value"], x + item_w * 3, caption_y, item_w, "status", cell.last_outcome, outcome_color(cell.last_outcome))

        caption_y += 36
        draw_caption_item(screen, fonts["label"], fonts["value"], x, caption_y, item_w, "step", f"{cell.env_steps}/{cell.spec.time_limit}", value_color)
        draw_caption_item(screen, fonts["label"], fonts["value"], x + item_w, caption_y, item_w, "action", cell.action_summary, value_color)
        draw_caption_item(screen, fonts["label"], fonts["value"], x + item_w * 2, caption_y, item_w, "decisions", str(cell.decisions), value_color)
        draw_caption_item(screen, fonts["label"], fonts["value"], x + item_w * 3, caption_y, item_w, "avg return", avg_text, avg_color)


def print_live_status(cells, specs, variants, episodes: int, previous: dict[str, str]):
    by_env = {spec.key: [] for spec in specs}
    for cell in cells:
        by_env[cell.spec.key].append(cell)

    current = {}
    for spec in specs:
        parts = []
        ordered = sorted(by_env[spec.key], key=lambda state: variants.index(state.variant))
        for cell in ordered:
            avg_text = "-" if cell.avg_return is None else f"{cell.avg_return:.2f}"
            parts.append(
                f"{cell.variant}: run={cell.episode}/{episodes} wins={cell.wins}/{episodes} "
                f"outcome={cell.last_outcome} action={cell.action_summary} "
                f"return={cell.score:.2f} avg_return={avg_text} t={cell.env_steps}/{cell.spec.time_limit}"
            )
        status = f"[LIVE] {spec.title} | " + " | ".join(parts)
        current[spec.key] = status
        if previous.get(spec.key) != status:
            print(status, flush=True)
    return current


def run_grid_test(cells, specs, variants, episodes: int, step_synced_pair, caption: str):
    pygame.init()
    screen = pygame.display.set_mode(grid_screen_size(len(specs)))
    pygame.display.set_caption(caption)
    fonts = make_grid_fonts()
    clock = pygame.time.Clock()
    last_print = 0.0
    last_status = {}

    try:
        running = True
        while running and any(not cell.done or cell.episode < episodes for cell in cells):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for row in range(len(specs)):
                step_synced_pair(cells[row * 2 : row * 2 + 2], episodes, seed_base=2000 + row * 100)

            draw_grid(screen, cells, episodes, fonts)
            pygame.display.flip()
            now = time.perf_counter()
            if now - last_print >= 1.0:
                last_status = print_live_status(cells, specs, variants, episodes, last_status)
                last_print = now
            clock.tick(20)
    finally:
        for cell in cells:
            cell.close()
        pygame.quit()
