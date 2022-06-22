import json
import os
from typing import Dict, List

import gym
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip
import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy


class GoalLinesCallback(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        for goal in base_env.envs[0].goal_space:
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
        **kwargs,
    ):
        goal_space = base_env.envs[0].goal_space
        goal_repr_dim = base_env.envs[0].goal_repr_dim
        agent_0_info = episode.last_info_for("agent_0")
        agent_1_info = episode.last_info_for("agent_1")
        agent_0_goal = episode.last_observation_for("agent_0")[-goal_repr_dim:].astype(
            int
        )
        agent_0_goal_name = "".join(str(t) for t in agent_0_goal)
        agent_0_reward = episode.last_reward_for("agent_0")
        agent_1_goal = episode.last_observation_for("agent_1")[-goal_repr_dim:].astype(
            int
        )
        agent_1_goal_name = "".join(str(t) for t in agent_1_goal)
        agent_1_reward = episode.last_reward_for("agent_1")

        # log raw rewards and count of goals that been sampled
        episode.media["reward_agent_0_" + agent_0_goal_name] = agent_0_reward
        episode.media["reward_agent_1_" + agent_1_goal_name] = agent_1_reward

        # log learning progress
        # individual LP
        for goal, lp in agent_0_info.get("learning_progress", {}).items():
            episode.custom_metrics["lp agent_0 " + goal] = lp
            episode.media["lp_agent_0_" + goal] = lp
        for goal, lp in agent_1_info.get("learning_progress", {}).items():
            episode.custom_metrics["lp agent_1 " + goal] = lp
            episode.media["lp_agent_1_" + goal] = lp
        for goal, competence in agent_0_info.get("competence", {}).items():
            episode.custom_metrics["competence agent_0 " + goal] = competence
            episode.media["competence_agent_0_" + goal] = competence
        for goal, competence in agent_1_info.get("competence", {}).items():
            episode.custom_metrics["competence agent_1 " + goal] = competence
            episode.media["competence_agent_1_" + goal] = competence
        # joint LP
        episode.media["lp_matrix_agent_0"] = agent_0_info.get("joint_learning_progress")
        episode.media["lp_matrix_agent_1"] = agent_1_info.get("joint_learning_progress")
        episode.media["competence_matrix_agent_0"] = agent_0_info.get(
            "joint_competence"
        )
        episode.media["competence_matrix_agent_1"] = agent_1_info.get(
            "joint_competence"
        )

        # log reward per goal (super hacky way to encode them but it works)
        # goal_000, goal_010, etc
        # log for each agent and collaborative goal, which position the agent reached when solving it
        # log reward for collective and individual goals separatelty
        if agent_0_goal_name == agent_1_goal_name:
            # If both agents had the same goal, log the mean of the rewards
            episode.custom_metrics["reward for goal " + agent_0_goal_name] = (
                agent_0_reward + agent_1_reward
            ) / 2
            if sum(agent_0_goal) > 1:
                # If goal is collective, log collective goal reward + last position
                episode.custom_metrics["reward for collective goal"] = (
                    agent_0_reward + agent_1_reward
                ) / 2
                # logging position of the agent when solving the goal
                if agent_0_reward:
                    episode.hist_data[
                        "agent 0 position for " + agent_0_goal_name
                    ].append(agent_0_info["goal_line"])
                if agent_1_reward:
                    episode.hist_data[
                        "agent 1 position for " + agent_1_goal_name
                    ].append(agent_1_info["goal_line"])
            else:
                episode.custom_metrics["reward for individual goal"] = (
                    agent_0_reward + agent_1_reward
                ) / 2
        else:
            # If agents had different goals, log each of them separately
            episode.custom_metrics[
                "reward for goal " + agent_0_goal_name
            ] = agent_0_reward
            episode.custom_metrics[
                "reward for goal " + agent_1_goal_name
            ] = agent_1_reward
            if sum(agent_0_goal) > 1:
                if agent_0_reward:
                    # logging position of the agent when solving the goal
                    episode.hist_data[
                        "agent 0 position for " + agent_0_goal_name
                    ].append(agent_0_info["goal_line"])
                if sum(agent_1_goal) > 1:
                    if agent_1_reward:
                        episode.hist_data[
                            "agent 1 position for " + agent_1_goal_name
                        ].append(agent_1_info["goal_line"])
                    episode.custom_metrics["reward for collective goal"] = (
                        agent_0_reward + agent_1_reward
                    ) / 2
                else:
                    episode.custom_metrics[
                        "reward for collective goal"
                    ] = agent_0_reward
                    episode.custom_metrics[
                        "reward for individual goal"
                    ] = agent_1_reward
            else:
                if sum(agent_1_goal) > 1:
                    if agent_1_reward:
                        episode.hist_data[
                            "agent 1 position for " + agent_1_goal_name
                        ].append(agent_1_info["goal_line"])
                    episode.custom_metrics[
                        "reward for collective goal"
                    ] = agent_1_reward
                    episode.custom_metrics[
                        "reward for individual goal"
                    ] = agent_0_reward
                else:
                    episode.custom_metrics["reward for individual goal"] = (
                        agent_0_reward + agent_1_reward
                    ) / 2

        # log which goal was sampled
        agent_0_goal_index = [
            i for i, g in enumerate(goal_space) if all(agent_0_goal == g)
        ][0]
        episode.hist_data["agent 0 goal"].append(agent_0_goal_index)
        agent_1_goal_index = [
            i for i, g in enumerate(goal_space) if all(agent_1_goal == g)
        ][0]
        episode.hist_data["agent 1 goal"].append(agent_1_goal_index)

        # log reward matrix to each agent [own_goal, other_goal] = reward obtained
        episode.custom_metrics[
            "matrix0" + str(agent_0_goal_index) + str(agent_1_goal_index)
        ] = agent_0_reward
        episode.custom_metrics[
            "matrix1" + str(agent_1_goal_index) + str(agent_0_goal_index)
        ] = agent_1_reward


def goal_lines_last_callback(trainer, n_goals):
    """
    Builds a video of the reward matrices for GoalLinesEnv
    """
    print("Building reward matrices videos ...")
    agent_imgs = [[], []]
    # Open result file and reformat the matrix metrics in to matrix
    result_dump = open(os.path.join(trainer.logdir, "result.json"), "r")
    idx = 0
    for result in result_dump:
        metrics = json.loads(result).get("evaluation")
        if metrics is not None:
            metrics = metrics.get("custom_metrics")
            matrix_0 = np.zeros([n_goals, n_goals])
            matrix_1 = np.zeros([n_goals, n_goals])
            for i in range(n_goals):
                for j in range(n_goals):
                    matrix_0[i, j] = metrics.get(
                        "matrix0" + str(i) + str(j) + "_mean", 0
                    )
                    matrix_1[i, j] = metrics.get(
                        "matrix1" + str(i) + str(j) + "_mean", 0
                    )
            # Generate matrix images
            for k, matrix in enumerate([matrix_0, matrix_1]):
                fig = plt.figure(num=1, clear=True)
                ax = fig.add_subplot()
                ax.imshow(matrix)
                ax.set_title(f"Agent {k}: evaluation iteration {idx}")
                if matrix.shape[0] < 10:
                    for (i, j), z in np.ndenumerate(matrix):
                        plt.text(j, i, "{:0.1f}".format(z), ha="center", va="center")
                canvas = fig.figure.canvas
                canvas.draw()
                data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                img = data.reshape(canvas.get_width_height()[::-1] + (3,))
                agent_imgs[k].append(img)
            idx += 1

    agent_0_matrix = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
        agent_imgs[0], fps=2
    )
    agent_0_matrix.write_videofile(os.path.join(trainer.logdir, "agent_0_rewards.mp4"))
    agent_1_matrix = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
        agent_imgs[1], fps=2
    )
    agent_1_matrix.write_videofile(os.path.join(trainer.logdir, "agent_1_rewards.mp4"))


def legacy_after_training_eval_rllib(
    trainer,
    eval_env: gym.Env,
    eval_episodes: int = 5,
    multiagent: bool = True,
) -> None:
    """Final evaluation function called after rllib training loop"""
    frames = []
    for n in range(eval_episodes):
        done = False
        obs = eval_env.reset()
        episode_reward = 0.0
        while not done:
            frame = eval_env.render()
            frames.append(frame)
            # Compute an action
            if not multiagent:
                a = trainer.compute_single_action(
                    observation=obs,
                    explore=False,
                )
                # Send the computed action `a` to the env.
                obs, reward, done, _ = eval_env.step(a)
                episode_reward += reward
            else:
                action = {}
                for agent_id, agent_obs in obs.items():
                    policy_id = trainer.config["multiagent"]["policy_mapping_fn"](
                        agent_id
                    )
                    action[agent_id] = trainer.compute_single_action(
                        agent_obs, policy_id=policy_id, explore=False
                    )
                obs, reward, dones, info = eval_env.step(action)
                done = dones["__all__"]
                # sum up reward for all agents
                episode_reward += sum(reward.values())
            # Is the episode `done`? -> Reset.
            if done:
                print(f"Episode done: Total reward = {episode_reward}")
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=30)
    clip.write_videofile(os.path.join(trainer.logdir, "trained_agent.mp4"))


def after_training_eval_rllib(
    trainer,
    eval_env: gym.Env,
    goal_list: List[Dict[str, int]],
) -> None:
    """Final evaluation function called after rllib training loop"""
    print("Starting final evaluation!")
    frames = []
    results = {}
    for goal in goal_list:
        eval_env.set_external_goal(goal, True)
        goal_name = str(eval_env.goal_space[list(goal.values())[0]])
        reward_for_this_goal = []
        for _ in range(10):
            done = False
            obs = eval_env.reset()
            episode_reward = 0.0
            while not done:
                frame = eval_env.render()
                frames.append(frame)
                # Compute an action
                action = {}
                for agent_id, agent_obs in obs.items():
                    policy_id = trainer.config["multiagent"]["policy_mapping_fn"](
                        agent_id
                    )
                    action[agent_id] = trainer.compute_single_action(
                        agent_obs, policy_id=policy_id, explore=True
                    )
                obs, reward, dones, info = eval_env.step(action)
                done = dones["__all__"]
                # sum up reward for all agents
                episode_reward += sum(reward.values())
            # Is the episode `done`? -> Reset.
            reward_for_this_goal.append(episode_reward)
        results[goal_name] = np.mean(reward_for_this_goal)
    print("Evaluation results over 10 episodes for each goal")
    print(results)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=30)
    clip.write_videofile(os.path.join(trainer.logdir, "trained_agent.mp4"))


class NewEnvCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        goal_space = base_env.envs[0].goal_space
        goal_repr_dim = base_env.envs[0].goal_repr_dim
        agent_0_goal = episode.last_observation_for("agent_0")[-goal_repr_dim:].astype(
            int
        )
        agent_0_goal_name = "".join(str(t) for t in agent_0_goal)
        agent_0_reward = episode.last_reward_for("agent_0")
        agent_1_goal = episode.last_observation_for("agent_1")[-goal_repr_dim:].astype(
            int
        )
        agent_1_goal_name = "".join(str(t) for t in agent_1_goal)
        agent_1_reward = episode.last_reward_for("agent_1")

        # log reward per goal and per agent-goal
        episode.custom_metrics[
            "reward for agent 0 " + agent_0_goal_name
        ] = agent_0_reward
        episode.custom_metrics[
            "reward for agent 1 " + agent_1_goal_name
        ] = agent_1_reward

        if np.all(agent_0_goal == agent_1_goal):
            # If both agents had the same goal, log the mean of the rewards
            episode.custom_metrics["reward for goal " + agent_0_goal_name] = (
                agent_0_reward + agent_1_reward
            ) / 2
        else:
            # If agents had different goals, log each of them separately
            episode.custom_metrics[
                "reward for goal " + agent_0_goal_name
            ] = agent_0_reward
            episode.custom_metrics[
                "reward for goal " + agent_1_goal_name
            ] = agent_1_reward

        # log which goal was sampled
        agent_0_goal_index = [
            i for i, g in enumerate(goal_space) if all(agent_0_goal == g)
        ][0]
        episode.hist_data["agent 0 goal"] = [agent_0_goal_index]
        agent_1_goal_index = [
            i for i, g in enumerate(goal_space) if all(agent_1_goal == g)
        ][0]
        episode.hist_data["agent 1 goal"] = [agent_1_goal_index]


class LargeGoalLinesCallback(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        for goal in base_env.envs[0].goal_space:
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
        **kwargs,
    ):
        goal_space = base_env.envs[0].goal_space
        goal_repr_dim = base_env.envs[0].goal_repr_dim
        agent_0_info = episode.last_info_for("agent_0")
        agent_1_info = episode.last_info_for("agent_1")
        agent_0_goal = episode.last_observation_for("agent_0")[-goal_repr_dim:].astype(
            int
        )
        agent_0_goal_name = "".join(str(t) for t in agent_0_goal)
        agent_0_reward = episode.last_reward_for("agent_0")
        agent_1_goal = episode.last_observation_for("agent_1")[-goal_repr_dim:].astype(
            int
        )
        agent_1_goal_name = "".join(str(t) for t in agent_1_goal)
        agent_1_reward = episode.last_reward_for("agent_1")

        episode.custom_metrics["reward for goal " + agent_0_goal_name] = agent_0_reward
        episode.custom_metrics["reward for goal " + agent_1_goal_name] = agent_1_reward

        if agent_0_goal_name == agent_1_goal_name:
            episode.custom_metrics["reward for same goal"] = agent_0_reward + agent_1_reward
        elif np.bitwise_or.reduce(np.vstack([agent_0_goal, agent_1_goal])).sum() == 3:
            episode.custom_metrics["reward for compatible goal"] = agent_0_reward + agent_1_reward
        if agent_0_reward:
            # logging position of the agent when solving the goal
            episode.hist_data["agent 0 position for " + agent_0_goal_name].append(
                agent_0_info["goal_line"]
            )
        if agent_1_reward:
            episode.hist_data["agent 1 position for " + agent_1_goal_name].append(
                agent_1_info["goal_line"]
            )

        # log which goal was sampled
        agent_0_goal_index = [
            i for i, g in enumerate(goal_space) if all(agent_0_goal == g)
        ][0]
        episode.hist_data["agent 0 goal"].append(agent_0_goal_index)
        agent_1_goal_index = [
            i for i, g in enumerate(goal_space) if all(agent_1_goal == g)
        ][0]
        episode.hist_data["agent 1 goal"].append(agent_1_goal_index)
