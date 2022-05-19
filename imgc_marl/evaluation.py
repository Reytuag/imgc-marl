from copy import deepcopy
from typing import Dict

import ray
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.tune.logger import pretty_print


def custom_eval_function(trainer, eval_workers: WorkerSet) -> Dict:
    """Evaluates current policy under `evaluation_config` settings.

    Merge evaluation_config with the normal trainer config.
    Run n_episodes of each of the goals provided in the evaluation_config

    """
    eval_cfg = trainer.config["evaluation_config"]

    worker_1, worker_2 = eval_workers.remote_workers()

    # For each goal we will run 10 episodes (using 2 parallel workers)
    for goal in eval_cfg["eval_goals"]:
        worker_1.foreach_env.remote(lambda env: env.set_external_goal(goal, True))
        worker_2.foreach_env.remote(lambda env: env.set_external_goal(goal, True))
        print("Custom evaluation in goals", goal)
        for i in range(5):
            ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    # Collect the accumulated episodes on the workers and summarize results
    episodes, _ = collect_episodes(remote_workers=eval_workers.remote_workers())
    metrics = summarize_episodes(episodes)
    metrics_to_print = deepcopy(metrics)
    keep_keys = [
        "episode_reward_max",
        "episode_reward_min",
        "episode_reward_mean",
        "episode_len_mean",
        "episodes_this_iter",
        "policy_reward_min",
        "policy_reward_max",
        "policy_reward_mean",
        "custom_metrics",
    ]

    keep_custom_keys = [
        "reward for collective goal_mean",
        "reward for collective goal_min",
        "reward for collective goal_max",
        "reward for individual goal_mean",
        "reward for individual goal_min",
        "reward for individual goal_max",
    ]
    metrics_to_print["custom_metrics"] = {
        key: value
        for key, value in metrics_to_print["custom_metrics"].items()
        if key in keep_custom_keys
    }
    metrics_to_print = {
        key: value for key, value in metrics_to_print.items() if key in keep_keys
    }
    print(pretty_print(metrics_to_print))
    return metrics
