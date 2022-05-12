import argparse
import os
from typing import Dict, Tuple

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

from imgc_marl.envs.multiagent import POSSIBLE_GOAL_LINES


class MultiGoalMultiAgentCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        for goal in POSSIBLE_GOAL_LINES:
            goal_name = "".join(str(t) for t in goal)
            episode.hist_data["agent 0 position for " + goal_name] = []
            episode.hist_data["agent 1 position for " + goal_name] = []
        episode.hist_data["agent 0 goal"] = []
        episode.hist_data["agent 1 goal"] = []

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # log reward per goal (super hacky way to encode them but it works)
        # goal_000, goal_010, etc
        agent_0_goal = episode.last_observation_for("agent_0")[-3:].astype(int)
        agent_0_goal_name = "".join(str(t) for t in agent_0_goal)
        episode.custom_metrics[
            "reward for goal " + agent_0_goal_name
        ] = episode.last_reward_for("agent_0")

        agent_1_goal = episode.last_observation_for("agent_1")[-3:].astype(int)
        agent_1_goal_name = "".join(str(t) for t in agent_1_goal)
        episode.custom_metrics[
            "reward for goal " + agent_1_goal_name
        ] = episode.last_reward_for("agent_1")

        # log which goal was sampled
        episode.hist_data["agent 0 goal"].append(
            [i for i, g in enumerate(POSSIBLE_GOAL_LINES) if all(agent_0_goal == g)][0]
        )
        episode.hist_data["agent 1 goal"].append(
            [i for i, g in enumerate(POSSIBLE_GOAL_LINES) if all(agent_1_goal == g)][0]
        )

        # log for each agent and collaborative goal, which position the agent reached when solving it
        # log reward for collective and individual goals separatelty
        if sum(agent_0_goal) > 1:
            episode.custom_metrics[
                "reward for collective goal"
            ] = episode.last_reward_for("agent_0")
            if episode.last_reward_for("agent_0"):
                episode.hist_data["agent 0 position for " + agent_0_goal_name].append(
                    episode.last_info_for("agent_0")["goal_line"]
                )
        else:
            episode.custom_metrics[
                "reward for individual goal"
            ] = episode.last_reward_for("agent_0")

        if sum(agent_1_goal) > 1:
            episode.custom_metrics[
                "reward for collective goal"
            ] = episode.last_reward_for("agent_1")
            if episode.last_reward_for("agent_1"):
                episode.hist_data["agent 1 position for " + agent_1_goal_name].append(
                    episode.last_info_for("agent_1")["goal_line"]
                )
        else:
            episode.custom_metrics[
                "reward for individual goal"
            ] = episode.last_reward_for("agent_1")
