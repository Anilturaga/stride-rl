# Stride RL
Training agents to take fewer, longer-lasting decisions in Gymnasium environments. This is a minimal demonstration of adaptive action repetition and its effect on long-horizon control.

> We refer to models with adaptive action repetition enabled as sleep models and the vanilla/baseline models as no-sleep. In this repository, sleep is just skipping inference for some time steps but still accumulating rewards.

![Sleep policy beats no-sleep baseline](assets/trained-model-test-runs.gif)

## Adaptive action repetition
At each decision point, the agent jointly picks an action and a commitment duration (1–N steps). 
The same action is applied for that many environment steps before the agent re-evaluates. This reduces the total number of decisions per episode and extends the temporal scope of each one.

## Environments
We have the lunar landing and mountain car environments although the script can be extended to other environments with similar observation and action spaces. In lunar landing, the sleep model almost always wins out over the non sleep model but in mountain car, they tend to reach the same outcome often with the sleep models being ahead sometimes. 

I added both to show that sleep as a label is only useful in some scenarios and learning to use it in highly precise ways is hard.

## Reward changes
Because models with sleep(AAR) enabled gets clearer signal from the rewards, they tend to perform reward hacking much earlier than vanilla models. One persistant case I have seen is the model trying to align the lander to the middle and letting it free fall to maximize reward.

To mitigate these, LunarLander gets a small terminal penalty for landing too fast, so it cannot exploit rough landings. MountainCar gets light shaping for pushing with its velocity, moving right, and reaching the goal, so learning is less sparse without directly helping sleep.

## Do it yourself
For training both models on both envs:
`uv run main.py train`

Models will be stored in `models_multi/` and the training graphs will be stored in `training_graphs_multi`

For running interactive inference:
`uv run main.py test`

Optional configs to tune:
uv run main.py train --run-name exp1 --timesteps 100000
uv run main.py test --run-name exp1 --episodes 5


## Reference research
- Sharma, S., Lakshminarayanan, A. S., & Ravindran, B. (2017). Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning. arXiv:1702.06054.
- Srinivas, A., Sharma, S., & Ravindran, B. (2017). Dynamic Action Repetition for Deep Reinforcement Learning. AAAI 2017.
- Biedenkapp, A. et al. (2021). TempoRL: Learning When to Act. ICML 2021.
- Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning. Artificial Intelligence, 112(1–2), 181–211.
- Vezhnevets, A. et al. (2016). Strategic Attentive Writer for Learning Macro-Actions. NeurIPS 2016.