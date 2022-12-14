import json
import os
from typing import Dict, List, Optional

import gym
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip
import numpy as np
import torch
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
        allowed_goals = base_env.envs[0].allowed_training_goals
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

        # log 0 shot generalization
        if (
            agent_0_goal.tolist() not in allowed_goals
            and agent_1_goal.tolist() not in allowed_goals
        ):
            episode.custom_metrics["reward_zero_shot"] = (
                agent_0_reward + agent_1_reward
            ) / 2
        elif agent_0_goal.tolist() not in allowed_goals:
            episode.custom_metrics["reward_zero_shot"] = agent_0_reward
        elif agent_1_goal.tolist() not in allowed_goals:
            episode.custom_metrics["reward_zero_shot"] = agent_1_reward


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
    video_path: Optional[str] = None,
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
    if video_path is not None:
        clip.write_videofile(
            os.path.join(video_path, f"checkpoint_{str(trainer.iteration)}.mp4")
        )
    else:
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
        allowed_goals = base_env.envs[0].allowed_training_goals
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
            episode.custom_metrics["reward for same goal"] = (
                agent_0_reward + agent_1_reward
            )

            if sum(agent_0_goal) > 1:
                episode.custom_metrics["goal_alignment"] = 1
                episode.custom_metrics["reward for collective goal"] = (
                    agent_0_reward + agent_1_reward
                )
            else:
                episode.custom_metrics["reward for individual goal"] = (
                    agent_0_reward + agent_1_reward
                )

        else:

            if np.bitwise_or.reduce(np.vstack([agent_0_goal, agent_1_goal])).sum() == 3:
                episode.custom_metrics["reward for partially compatible goal"] = (
                    agent_0_reward + agent_1_reward
                )
            if np.bitwise_or.reduce(np.vstack([agent_0_goal, agent_1_goal])).sum() == 2:
                episode.custom_metrics["reward for compatible goal"] = (
                    agent_0_reward + agent_1_reward
                )
            if sum(agent_0_goal) > 1 and sum(agent_1_goal) > 1:
                episode.custom_metrics["goal_alignment"] = 0
                episode.custom_metrics["reward for collective goal"] = (
                    agent_0_reward + agent_1_reward
                )
            elif sum(agent_0_goal) <= 1 and sum(agent_1_goal) <= 1:
                episode.custom_metrics["reward for individual goal"] = (
                    agent_0_reward + agent_1_reward
                )
            elif sum(agent_0_goal) > 1:
                episode.custom_metrics["goal_alignment"] = 0
                episode.custom_metrics["reward for collective goal"] = agent_0_reward
                episode.custom_metrics["reward for individual goal"] = agent_1_reward
            else:
                episode.custom_metrics["goal_alignment"] = 0
                episode.custom_metrics["reward for collective goal"] = agent_1_reward
                episode.custom_metrics["reward for individual goal"] = agent_0_reward

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

        # log 0 shot generalization
        if (
            agent_0_goal.tolist() not in allowed_goals
            and agent_1_goal.tolist() not in allowed_goals
        ):
            episode.custom_metrics["reward_zero_shot"] = (
                agent_0_reward + agent_1_reward
            ) / 2
        elif agent_0_goal.tolist() not in allowed_goals:
            episode.custom_metrics["reward_zero_shot"] = agent_0_reward
        elif agent_1_goal.tolist() not in allowed_goals:
            episode.custom_metrics["reward_zero_shot"] = agent_1_reward


class LargeGoalLinesBasicCommunicationCallback(LargeGoalLinesCallback):
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
        # e-greedy threshold
        e_greedy = base_env.envs[0].eps_communication
        # decide which agent will take the lead
        sampled_goal = base_env.envs[0].goal_space[
            np.random.randint(0, base_env.envs[0].goal_space_dim)
        ]
        if np.random.random() > 0.5:
            # agent 0 lead
            with torch.no_grad():
                predicted_values = policies["agent_0"].model._communication_branch(
                    torch.tensor(sampled_goal, dtype=torch.float)
                )

            # e-greedy
            if np.random.random() < e_greedy:
                selected_goal_index = np.random.randint(
                    0, base_env.envs[0].goal_space_dim
                )
            else:
                selected_goal_index = predicted_values.argmax().item()

            # boltzman sampling
            # with torch.no_grad():
            #     p = torch.nn.functional.softmax(predicted_values).numpy().astype('float64')
            #     p = p / sum(p)
            #     selected_goal_index = np.random.choice(
            #         range(len(base_env.envs[0].goal_space)),
            #         p=p,
            #     )

            agent_1_goal = base_env.envs[0].goal_space[selected_goal_index]
            agent_0_goal = sampled_goal
            message = {
                "agent_0": {
                    "input_goal": sampled_goal,
                    "output_goal_index": selected_goal_index,
                }
            }
        else:
            # agent 1 lead
            with torch.no_grad():
                predicted_values = policies["agent_1"].model._communication_branch(
                    torch.tensor(sampled_goal, dtype=torch.float)
                )

            # e-greedy
            if np.random.random() < e_greedy:
                selected_goal_index = np.random.randint(
                    0, base_env.envs[0].goal_space_dim
                )
            else:
                selected_goal_index = predicted_values.argmax().item()

            # boltzman sampling
            # with torch.no_grad():
            #     p = torch.nn.functional.softmax(predicted_values).numpy().astype('float64')
            #     p = p / sum(p)
            #     selected_goal_index = np.random.choice(
            #         range(len(base_env.envs[0].goal_space)),
            #         p=p,
            #     )

            agent_1_goal = sampled_goal
            agent_0_goal = base_env.envs[0].goal_space[selected_goal_index]
            message = {
                "agent_1": {
                    "input_goal": sampled_goal,
                    "output_goal_index": selected_goal_index,
                }
            }

        goals = {"agent_0": agent_0_goal, "agent_1": agent_1_goal}

        worker.foreach_env(lambda env: env.set_goal_and_message(goals, message))

        for goal in base_env.envs[0].goal_space:
            goal_name = "".join(str(t) for t in goal)
            episode.hist_data["agent 0 position for " + goal_name] = []
            episode.hist_data["agent 1 position for " + goal_name] = []
        episode.hist_data["agent 0 goal"] = []
        episode.hist_data["agent 1 goal"] = []


# Add this for backward compatibility
class LargeGoalLinesCommunicationCallback(LargeGoalLinesCallback):
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
        # e-greedy threshold
        e_greedy = base_env.envs[0].eps_communication
        # decide which agent will take the lead
        sampled_goal = base_env.envs[0].goal_space[
            np.random.randint(0, base_env.envs[0].goal_space_dim)
        ]
        if np.random.random() > 0.5:
            # agent 0 lead
            with torch.no_grad():
                predicted_values = policies["agent_0"].model._communication_branch(
                    torch.tensor(sampled_goal, dtype=torch.float)
                )
            # e-greedy
            if np.random.random() < e_greedy:
                selected_goal_index = np.random.randint(
                    0, base_env.envs[0].goal_space_dim
                )
            else:
                selected_goal_index = predicted_values.argmax().item()

            agent_1_goal = base_env.envs[0].goal_space[selected_goal_index]
            agent_0_goal = sampled_goal
            message = {
                "agent_0": {
                    "input_goal": sampled_goal,
                    "output_goal_index": selected_goal_index,
                }
            }
        else:
            # agent 1 lead
            with torch.no_grad():
                predicted_values = policies["agent_1"].model._communication_branch(
                    torch.tensor(sampled_goal, dtype=torch.float)
                )
            # e-greedy
            if np.random.random() < e_greedy:
                selected_goal_index = np.random.randint(
                    0, base_env.envs[0].goal_space_dim
                )
            else:
                selected_goal_index = predicted_values.argmax().item()

            agent_1_goal = sampled_goal
            agent_0_goal = base_env.envs[0].goal_space[selected_goal_index]
            message = {
                "agent_1": {
                    "input_goal": sampled_goal,
                    "output_goal_index": selected_goal_index,
                }
            }

        goals = {"agent_0": agent_0_goal, "agent_1": agent_1_goal}

        worker.foreach_env(lambda env: env.set_goal_and_message(goals, message))

        for goal in base_env.envs[0].goal_space:
            goal_name = "".join(str(t) for t in goal)
            episode.hist_data["agent 0 position for " + goal_name] = []
            episode.hist_data["agent 1 position for " + goal_name] = []
        episode.hist_data["agent 0 goal"] = []
        episode.hist_data["agent 1 goal"] = []


class LargeGoalLinesFullCommunicationCallback(LargeGoalLinesCallback):
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
        # e-greedy threshold
        e_greedy = base_env.envs[0].eps_communication
        # n-goals
        n_goals = base_env.envs[0].goal_space_dim
        # decide which agent will take the lead
        sampled_goal = base_env.envs[0].goal_space[np.random.randint(0, n_goals)]
        sampled_goal_tensor = torch.tensor(sampled_goal, dtype=torch.float)
        possible_messages = torch.nn.functional.one_hot(
            torch.arange(0, n_goals),
        ).float()
        possible_goals = torch.tensor(base_env.envs[0].goal_space, dtype=torch.float)
        if np.random.random() > 0.5:
            # agent 0 is the leader
            agent_0_goal = sampled_goal
            # e-greedy
            if np.random.random() < e_greedy:
                message_index = np.random.randint(0, n_goals)
            else:
                leader_input = torch.cat(
                    [sampled_goal_tensor.repeat(n_goals, 1), possible_messages], 1
                )
                with torch.no_grad():
                    leader_prediction = policies["agent_0"].model._communication_branch(
                        leader_input
                    )
                    message_index = leader_prediction.argmax().item()

            # agent 1 is the follower
            message = possible_messages[message_index]
            # e-greedy
            if np.random.random() < e_greedy:
                goal_index = np.random.randint(0, n_goals)
            else:
                follower_input = torch.cat(
                    [possible_goals, message.repeat(n_goals, 1)], 1
                )
                with torch.no_grad():
                    follower_prediction = policies[
                        "agent_1"
                    ].model._communication_branch(follower_input)
                    goal_index = follower_prediction.argmax().item()
            agent_1_goal = base_env.envs[0].goal_space[goal_index]
            info = {
                "agent_0": {
                    "leader": True,
                    "follower": False,
                    "input_goal": sampled_goal_tensor,
                    "output_message": message,
                },
                "agent_1": {
                    "leader": False,
                    "follower": True,
                    "input_message": message,
                    "output_goal": torch.tensor(agent_1_goal, dtype=torch.float),
                },
            }
        else:
            # agent 1 is the leader
            agent_1_goal = sampled_goal
            # e-greedy
            if np.random.random() < e_greedy:
                message_index = np.random.randint(0, n_goals)
            else:
                leader_input = torch.cat(
                    [sampled_goal_tensor.repeat(n_goals, 1), possible_messages], 1
                )
                with torch.no_grad():
                    leader_prediction = policies["agent_1"].model._communication_branch(
                        leader_input
                    )
                    message_index = leader_prediction.argmax().item()

            # agent 0 is the follower
            message = possible_messages[message_index]
            # e-greedy
            if np.random.random() < e_greedy:
                goal_index = np.random.randint(0, n_goals)
            else:
                follower_input = torch.cat(
                    [possible_goals, message.repeat(n_goals, 1)], 1
                )
                with torch.no_grad():
                    follower_prediction = policies[
                        "agent_0"
                    ].model._communication_branch(follower_input)
                    goal_index = follower_prediction.argmax().item()
            agent_0_goal = base_env.envs[0].goal_space[goal_index]
            info = {
                "agent_0": {
                    "leader": False,
                    "follower": True,
                    "input_message": message,
                    "output_goal": torch.tensor(agent_0_goal, dtype=torch.float),
                },
                "agent_1": {
                    "leader": True,
                    "follower": False,
                    "input_goal": sampled_goal_tensor,
                    "output_message": message,
                },
            }

        goals = {"agent_0": agent_0_goal, "agent_1": agent_1_goal}

        worker.foreach_env(lambda env: env.set_goal_and_message(goals, info))

        for goal in base_env.envs[0].goal_space:
            goal_name = "".join(str(t) for t in goal)
            episode.hist_data["agent 0 position for " + goal_name] = []
            episode.hist_data["agent 1 position for " + goal_name] = []
        episode.hist_data["agent 0 goal"] = []
        episode.hist_data["agent 1 goal"] = []


class LargeGoalLinesBasicNamingGame(LargeGoalLinesCallback):
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
        # decide which agent will take the lead
        leader_goal_index = np.random.randint(0, base_env.envs[0].goal_space_dim)
        leader_goal = base_env.envs[0].goal_space[leader_goal_index]
        if np.random.random() > 0.5:
            # agent 0 lead
            with torch.no_grad():
                # SOFTMAX
                #     scores = (
                #         torch.nn.functional.softmax(
                #             policies["agent_0"].model._communication_matrix[
                #                 leader_goal_index
                #             ]
                #         )
                #         .detach()
                #         .numpy()
                #     )
                # follower_goal_index = np.random.choice(range(len(scores)), 1, p=scores)[0]

                # E-GREEDY
                if np.random.random() < 0.2:
                    follower_goal_index = np.random.choice(
                        range(policies["agent_0"].model._communication_matrix.shape[0])
                    )
                else:
                    follower_goal_index = (
                        policies["agent_0"]
                        .model._communication_matrix[leader_goal_index]
                        .argmax()
                        .item()
                    )

            follower_goal = base_env.envs[0].goal_space[follower_goal_index]
            agent_0_goal = leader_goal
            agent_1_goal = follower_goal
            message = {
                "agent_0": {
                    "leader_goal": leader_goal_index,
                    "follower_goal": follower_goal_index,
                }
            }
        else:
            # agent 1 lead
            with torch.no_grad():
                # SOFTMAX
                #     scores = (
                #         torch.nn.functional.softmax(
                #             policies["agent_1"].model._communication_matrix[
                #                 leader_goal_index
                #             ]
                #         )
                #         .detach()
                #         .numpy()
                #     )
                # follower_goal_index = np.random.choice(range(len(scores)), 1, p=scores)[0]

                # E-GREEDY
                if np.random.random() < 0.2:
                    follower_goal_index = np.random.choice(
                        range(policies["agent_1"].model._communication_matrix.shape[0])
                    )
                else:
                    follower_goal_index = (
                        policies["agent_1"]
                        .model._communication_matrix[leader_goal_index]
                        .argmax()
                        .item()
                    )

            follower_goal = base_env.envs[0].goal_space[follower_goal_index]
            agent_1_goal = leader_goal
            agent_0_goal = follower_goal
            message = {
                "agent_1": {
                    "leader_goal": leader_goal_index,
                    "follower_goal": follower_goal_index,
                }
            }

        goals = {"agent_0": agent_0_goal, "agent_1": agent_1_goal}

        worker.foreach_env(lambda env: env.set_goal_and_message(goals, message))

        for goal in base_env.envs[0].goal_space:
            goal_name = "".join(str(t) for t in goal)
            episode.hist_data["agent 0 position for " + goal_name] = []
            episode.hist_data["agent 1 position for " + goal_name] = []
        episode.hist_data["agent 0 goal"] = []
        episode.hist_data["agent 1 goal"] = []

    def on_train_result(self, *, trainer: "Trainer", result: dict, **kwargs) -> None:
        """Called at the end of Trainable.train().

        Args:
            trainer: Current trainer instance.
            result: Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        result["naming_matrix_0"] = (
            trainer.get_policy("agent_0")
            .model._communication_matrix.detach()
            .cpu()
            .numpy()
        )
        result["naming_matrix_1"] = (
            trainer.get_policy("agent_1")
            .model._communication_matrix.detach()
            .cpu()
            .numpy()
        )


class PopGoalLinesCallback(DefaultCallbacks):
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
        agent_a = base_env.envs[0].playground.agents[0]
        agent_b = base_env.envs[0].playground.agents[1]
        agent_a_name = agent_a.name
        agent_b_name = agent_b.name
        goal_repr_dim = base_env.envs[0].goal_repr_dim
        agent_a_info = episode.last_info_for(agent_a_name)
        agent_b_info = episode.last_info_for(agent_b_name)
        agent_a_goal = episode.last_observation_for(agent_a_name)[
            -goal_repr_dim:
        ].astype(int)
        agent_a_goal_name = "".join(str(t) for t in agent_a_goal)
        agent_a_reward = episode.last_reward_for(agent_a_name)
        agent_b_goal = episode.last_observation_for(agent_b_name)[
            -goal_repr_dim:
        ].astype(int)
        agent_b_goal_name = "".join(str(t) for t in agent_b_goal)
        agent_b_reward = episode.last_reward_for(agent_b_name)
        # log reward per goal (super hacky way to encode them but it works)
        # goal_000, goal_010, etc
        # log for each agent and collaborative goal, which position the agent reached when solving it
        # log reward for collective and individual goals separatelty
        if agent_a_goal_name == agent_b_goal_name:
            # If both agents had the same goal, log the mean of the rewards
            episode.custom_metrics["reward for goal " + agent_a_goal_name] = (
                agent_a_reward + agent_b_reward
            ) / 2
            if sum(agent_a_goal) > 1:
                episode.custom_metrics["goal_alignment"] = 1
                # If goal is collective, log collective goal reward + last position
                episode.custom_metrics["reward for collective goal"] = (
                    agent_a_reward + agent_b_reward
                ) / 2
                # logging position of the agent when solving the goal
                if agent_a_reward:
                    episode.hist_data[
                        f"{agent_a_name} position for " + agent_a_goal_name
                    ] = [agent_a_info["goal_line"]]
                if agent_b_reward:
                    episode.hist_data[
                        f"{agent_b_name} position for " + agent_b_goal_name
                    ] = [agent_b_info["goal_line"]]
            else:
                episode.custom_metrics["reward for individual goal"] = (
                    agent_a_reward + agent_b_reward
                ) / 2
        else:
            # If agents had different goals, log each of them separately
            episode.custom_metrics[
                "reward for goal " + agent_a_goal_name
            ] = agent_a_reward
            episode.custom_metrics[
                "reward for goal " + agent_b_goal_name
            ] = agent_b_reward
            if sum(agent_a_goal) > 1:
                episode.custom_metrics["goal_alignment"] = 0
                if agent_a_reward:
                    # logging position of the agent when solving the goal
                    episode.hist_data[
                        f"{agent_a_name} position for " + agent_a_goal_name
                    ] = [agent_a_info["goal_line"]]
                if sum(agent_b_goal) > 1:
                    if agent_b_reward:
                        episode.hist_data[
                            f"{agent_b_name} position for " + agent_b_goal_name
                        ] = [agent_b_info["goal_line"]]
                    episode.custom_metrics["reward for collective goal"] = (
                        agent_a_reward + agent_b_reward
                    ) / 2
                else:
                    episode.custom_metrics[
                        "reward for collective goal"
                    ] = agent_a_reward
                    episode.custom_metrics[
                        "reward for individual goal"
                    ] = agent_b_reward
            else:
                if sum(agent_b_goal) > 1:
                    episode.custom_metrics["goal_alignment"] = 0
                    if agent_b_reward:
                        episode.hist_data[
                            f"{agent_b_name} position for " + agent_b_goal_name
                        ] = [agent_b_info["goal_line"]]
                    episode.custom_metrics[
                        "reward for collective goal"
                    ] = agent_b_reward
                    episode.custom_metrics[
                        "reward for individual goal"
                    ] = agent_a_reward
                else:
                    episode.custom_metrics["reward for individual goal"] = (
                        agent_a_reward + agent_b_reward
                    ) / 2


class PopGoalLinesCommunicationCallback(PopGoalLinesCallback):
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
        # Greedy communication-leader strategy over fixed leader's goal (only during evaluation)
        if base_env.envs[0].fixed_leader_goal is not None:
            agent_a = base_env.envs[0].playground.agents[0]
            agent_b = base_env.envs[0].playground.agents[1]
            leader_goal = base_env.envs[0].goal_space[
                base_env.envs[0].fixed_leader_goal
            ]
            if np.random.random() > 0.5:
                # agent a lead
                with torch.no_grad():
                    predicted_values = policies[
                        agent_a.name
                    ].model._communication_branch(
                        torch.tensor(leader_goal, dtype=torch.float)
                    )
                    selected_goal_index = predicted_values.argmax().item()
                agent_a_goal_index = base_env.envs[0].fixed_leader_goal
                agent_b_goal_index = selected_goal_index
            else:
                # agent b lead
                with torch.no_grad():
                    predicted_values = policies[
                        agent_b.name
                    ].model._communication_branch(
                        torch.tensor(leader_goal, dtype=torch.float)
                    )
                    selected_goal_index = predicted_values.argmax().item()
                agent_a_goal_index = selected_goal_index
                agent_b_goal_index = base_env.envs[0].fixed_leader_goal
            goals = {agent_a.name: agent_a_goal_index, agent_b.name: agent_b_goal_index}
            worker.foreach_env(lambda env: env.set_external_goal(goals))

        # e-greedy (during training)
        else:
            # e-greedy threshold
            e_greedy = base_env.envs[0].eps_communication
            # decide which agent will take the lead
            sampled_goal = base_env.envs[0].goal_space[
                np.random.randint(0, base_env.envs[0].goal_space_dim)
            ]
            agent_a = base_env.envs[0].playground.agents[0]
            agent_b = base_env.envs[0].playground.agents[1]
            if np.random.random() > 0.5:
                # agent a lead
                with torch.no_grad():
                    predicted_values = policies[
                        agent_a.name
                    ].model._communication_branch(
                        torch.tensor(sampled_goal, dtype=torch.float)
                    )
                # e-greedy
                if np.random.random() < e_greedy:
                    selected_goal_index = np.random.randint(
                        0, base_env.envs[0].goal_space_dim
                    )
                else:
                    selected_goal_index = predicted_values.argmax().item()

                agent_b_goal = base_env.envs[0].goal_space[selected_goal_index]
                agent_a_goal = sampled_goal
                message = {
                    agent_a.name: {
                        "input_goal": sampled_goal,
                        "output_goal_index": selected_goal_index,
                    }
                }
            else:
                # agent b lead
                with torch.no_grad():
                    predicted_values = policies[
                        agent_b.name
                    ].model._communication_branch(
                        torch.tensor(sampled_goal, dtype=torch.float)
                    )
                # e-greedy
                if np.random.random() < e_greedy:
                    selected_goal_index = np.random.randint(
                        0, base_env.envs[0].goal_space_dim
                    )
                else:
                    selected_goal_index = predicted_values.argmax().item()

                agent_b_goal = sampled_goal
                agent_a_goal = base_env.envs[0].goal_space[selected_goal_index]
                message = {
                    agent_b.name: {
                        "input_goal": sampled_goal,
                        "output_goal_index": selected_goal_index,
                    }
                }

            goals = {agent_a.name: agent_a_goal, agent_b.name: agent_b_goal}

            worker.foreach_env(lambda env: env.set_goal_and_message(goals, message))

temper=30
class PopGoalLinesNamingCallback(PopGoalLinesCallback):
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
        # Greedy communication-leader strategy over fixed leader's goal (only during evaluation)
        if base_env.envs[0].fixed_leader_goal is not None:
            agent_a = base_env.envs[0].playground.agents[0]
            agent_b = base_env.envs[0].playground.agents[1]
            leader_goal = base_env.envs[0].goal_space[
                base_env.envs[0].fixed_leader_goal
            ]
            leader_goal_index = base_env.envs[0].fixed_leader_goal
            if np.random.random() > 0.5:
                # agent a lead
                with torch.no_grad():
                    leader_msg_index = (
                        policies[agent_a.name]
                        .model._leader_matrix[leader_goal_index]
                        .argmax()
                        .item()
                    )
                    follower_goal_index = (
                        policies[agent_b.name]
                        .model._follower_matrix[leader_msg_index]
                        .argmax()
                        .item()
                    )
                    agent_a_goal_index = leader_goal_index
                    agent_b_goal_index = follower_goal_index
            else:
                # agent b lead
                with torch.no_grad():
                    leader_msg_index = np.argmax(
                        policies[agent_b.name].model._leader_matrix[leader_goal_index]
                    )
                    follower_goal_index = np.argmax(
                        policies[agent_a.name].model._follower_matrix[leader_msg_index]
                    )
                    agent_b_goal_index = leader_goal_index
                    agent_a_goal_index = follower_goal_index

            goals = {agent_a.name: agent_a_goal_index, agent_b.name: agent_b_goal_index}
            worker.foreach_env(lambda env: env.set_external_goal(goals))

        # e-greedy (during training)
        else:
            # e-greedy threshold
            e_greedy = base_env.envs[0].eps_communication
            # decide which agent will take the lead
            leader_goal_index = np.random.randint(0, base_env.envs[0].goal_space_dim)
            agent_a = base_env.envs[0].playground.agents[0]
            agent_b = base_env.envs[0].playground.agents[1]
            if np.random.random() > 0.5:
                # agent a lead
                # e-greedy
                #if np.random.random() < e_greedy:
                #    leader_msg_index = np.random.randint(
                #        0, base_env.envs[0].goal_space_dim
                #    )
                #else:
                #    with torch.no_grad():
                #        leader_msgs = (
                #            policies[agent_a.name]
                #            .model._leader_matrix[leader_goal_index]
                #            .numpy()
                #        )
                #    leader_msg_index = np.random.choice(
                #        np.flatnonzero(leader_msgs == leader_msgs.max())
                #    )
                #if np.random.random() < e_greedy:
                #    follower_goal_index = np.random.randint(
                #        0, base_env.envs[0].goal_space_dim
                #    )
                #else:
                #    with torch.no_grad():
                #        follower_goals = (
                #            policies[agent_b.name]
                #            .model._follower_matrix[leader_msg_index]
                #            .numpy()
                #        )
                #    follower_goal_index = np.random.choice(
                #        np.flatnonzero(follower_goals == follower_goals.max())
                #    )

                # SOFTMAX
                with torch.no_grad():
                    scores = (
                         torch.nn.functional.softmax(
                            temper*policies[agent_a.name].model._leader_matrix[leader_goal_index]
                         )
                         .detach()
                         .numpy()
                     )
                    
                
                leader_msg_index = np.random.choice(range(len(scores)), 1, p=scores)[0]

                with torch.no_grad():
                    scores = (
                         torch.nn.functional.softmax(
                            temper*policies[agent_b.name].model._follower_matrix[:,leader_msg_index]
                         )
                         .detach()
                         .numpy()
                     )
                follower_goal_index = np.random.choice(range(len(scores)), 1, p=scores)[0]

                    

                agent_a_goal = base_env.envs[0].goal_space[leader_goal_index]
                agent_b_goal = base_env.envs[0].goal_space[follower_goal_index]
                message = {
                    agent_a.name: {
                        "leader_goal_index": leader_goal_index,
                        "leader_msg_index": leader_msg_index,
                    },
                    agent_b.name: {
                        "follower_goal_index": follower_goal_index,
                        "leader_msg_index": leader_msg_index,
                    },
                }
            else:
                # agent b lead
                # e-greedy
                #if np.random.random() < e_greedy:
                #    leader_msg_index = np.random.randint(
                #        0, base_env.envs[0].goal_space_dim
                #    )
                #else:
                #    with torch.no_grad():
                #        leader_msgs = (
                #            policies[agent_b.name]
                #            .model._leader_matrix[leader_goal_index]
                #            .numpy()
                #        )
                #    leader_msg_index = np.random.choice(
                #        np.flatnonzero(leader_msgs == leader_msgs.max())
                #    )

                #if np.random.random() < e_greedy:
                #    follower_goal_index = np.random.randint(
                #        0, base_env.envs[0].goal_space_dim
                #    )
                #else:
                #    with torch.no_grad():
                #        follower_goals = (
                #            policies[agent_a.name]
                #            .model._follower_matrix[leader_msg_index]
                #            .numpy()
                #        )
                #    follower_goal_index = np.random.choice(
                #        np.flatnonzero(follower_goals == follower_goals.max())
                #    )
                
                
                with torch.no_grad():
                    scores = (
                         torch.nn.functional.softmax(
                            temper*policies[agent_b.name].model._leader_matrix[leader_goal_index]
                         )
                         .detach()
                         .numpy()
                     )
                    
                
                leader_msg_index = np.random.choice(range(len(scores)), 1, p=scores)[0]

                with torch.no_grad():
                    scores = (
                         torch.nn.functional.softmax(
                            temper*policies[agent_a.name].model._follower_matrix[:,leader_msg_index]
                         )
                         .detach()
                         .numpy()
                     )
                follower_goal_index = np.random.choice(range(len(scores)), 1, p=scores)[0]

                agent_b_goal = base_env.envs[0].goal_space[leader_goal_index]
                agent_a_goal = base_env.envs[0].goal_space[follower_goal_index]
                message = {
                    agent_b.name: {
                        "leader_goal_index": leader_goal_index,
                        "leader_msg_index": leader_msg_index,
                    },
                    agent_a.name: {
                        "follower_goal_index": follower_goal_index,
                        "leader_msg_index": leader_msg_index,
                    },
                }

            goals = {agent_a.name: agent_a_goal, agent_b.name: agent_b_goal}

            worker.foreach_env(lambda env: env.set_goal_and_message(goals, message))



def entropy_softmax(x,temperature1=15,temperature2=1):
    with torch.no_grad():
        p=torch.nn.functional.softmax(temperature1*x,dim=1)
        ent=-(p*torch.log(p+1e-10)).sum(axis=1)
        p_g=torch.nn.functional.softmax(temperature2*ent)
    return p_g
    

class PopGoalLinesNamingCallback1Matrix(PopGoalLinesCallback):
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
        # Greedy communication-leader strategy over fixed leader's goal (only during evaluation)
        if base_env.envs[0].fixed_leader_goal is not None:
            agent_a = base_env.envs[0].playground.agents[0]
            agent_b = base_env.envs[0].playground.agents[1]
            leader_goal = base_env.envs[0].goal_space[
                base_env.envs[0].fixed_leader_goal
            ]
            leader_goal_index = base_env.envs[0].fixed_leader_goal
            if np.random.random() > 0.5:
                # agent a lead
                with torch.no_grad():
                    leader_msg_index = (
                        policies[agent_a.name]
                        .model._matrix[leader_goal_index]
                        .argmax()
                        .item()
                    )
                    follower_goal_index = (
                        policies[agent_b.name]
                        .model._matrix[:, leader_msg_index]
                        .argmax()
                        .item()
                    )
                    agent_a_goal_index = leader_goal_index
                    agent_b_goal_index = follower_goal_index
            else:
                # agent b lead
                with torch.no_grad():
                    leader_msg_index = np.argmax(
                        policies[agent_b.name].model._matrix[leader_goal_index]
                    )
                    follower_goal_index = np.argmax(
                        policies[agent_a.name].model._matrix[:, leader_msg_index]
                    )
                    agent_b_goal_index = leader_goal_index
                    agent_a_goal_index = follower_goal_index

            goals = {agent_a.name: agent_a_goal_index, agent_b.name: agent_b_goal_index}
            worker.foreach_env(lambda env: env.set_external_goal(goals))

        # e-greedy (during training)
        else:
            # e-greedy threshold
            e_greedy = base_env.envs[0].eps_communication
            # decide which agent will take the lead
            leader_goal_index = np.random.randint(0, base_env.envs[0].goal_space_dim)
            agent_a = base_env.envs[0].playground.agents[0]
            agent_b = base_env.envs[0].playground.agents[1]
            if np.random.random() > 0.5:
                
                # agent a lead
                
                #if selection based on entropy of the line in matrix 
                #p_g=entropy_softmax(policies[agent_a.name].model._matrix).detach().numpy()
                #leader_goal_index=np.random.choice(range(base_env.envs[0].goal_space_dim), 1, p=p_g)[0]
                
                
                # e-greedy
                #if np.random.random() < e_greedy:
                #    leader_msg_index = np.random.randint(
                #        0, base_env.envs[0].goal_space_dim
                #    )
                #else:
                #    with torch.no_grad():
                #        leader_msgs = (
                #            policies[agent_a.name]
                #            .model._matrix[leader_goal_index]
                #            .numpy()
                #        )
                #    leader_msg_index = np.random.choice(
                #        np.flatnonzero(leader_msgs == leader_msgs.max())
                #    )

                #if np.random.random() < e_greedy:
                #    follower_goal_index = np.random.randint(
                #        0, base_env.envs[0].goal_space_dim
                #    )
                #else:
                #    with torch.no_grad():
                #        follower_goals = (
                #            policies[agent_b.name]
                #            .model._matrix[:, leader_msg_index]
                #            .numpy()
                #        )
                #    follower_goal_index = np.random.choice(
                #        np.flatnonzero(follower_goals == follower_goals.max())
                #   )
                #softmax
                with torch.no_grad():
                    scores = (
                         torch.nn.functional.softmax(
                            temper*policies[agent_a.name].model._matrix[leader_goal_index]
                         )
                         .detach()
                         .numpy()
                     )
                leader_msg_index = np.random.choice(range(len(scores)), 1, p=scores)[0]


                with torch.no_grad():
                    scores = (
                         torch.nn.functional.softmax(
                             temper*policies[agent_b.name].model._matrix[:,leader_msg_index]
                         )
                         .detach()
                         .numpy()
                     )
                follower_goal_index = np.random.choice(range(len(scores)), 1, p=scores)[0]

                agent_a_goal = base_env.envs[0].goal_space[leader_goal_index]
                agent_b_goal = base_env.envs[0].goal_space[follower_goal_index]
                message = {
                    agent_a.name: {
                        "leader_goal_index": leader_goal_index,
                        "leader_msg_index": leader_msg_index,
                    },
                    agent_b.name: {
                        "follower_goal_index": follower_goal_index,
                        "leader_msg_index": leader_msg_index,
                    },
                }
            else:
                # agent b lead
                
                #p_g=entropy_softmax(policies[agent_b.name].model._matrix).detach().numpy()
                #leader_goal_index=np.random.choice(range(base_env.envs[0].goal_space_dim), 1, p=p_g)[0]
                
                # e-greedy
                #if np.random.random() < e_greedy:
                #    leader_msg_index = np.random.randint(
                #        0, base_env.envs[0].goal_space_dim
                #    )
                #else:
                #    with torch.no_grad():
                #        leader_msgs = (
                #            policies[agent_b.name]
                #            .model._matrix[leader_goal_index]
                #            .numpy()
                #        )
                #    leader_msg_index = np.random.choice(
                #        np.flatnonzero(leader_msgs == leader_msgs.max())
                #    )

                #if np.random.random() < e_greedy:
                #    follower_goal_index = np.random.randint(
                #        0, base_env.envs[0].goal_space_dim
                #    )
                #else:
                #    with torch.no_grad():
                #        follower_goals = (
                #            policies[agent_a.name]
                #            .model._matrix[:, leader_msg_index]
                #            .numpy()
                #        )
                #    follower_goal_index = np.random.choice(
                #        np.flatnonzero(follower_goals == follower_goals.max())
                #    )

                #softmax
                with torch.no_grad():
                    scores = (
                         torch.nn.functional.softmax(
                            temper*policies[agent_b.name].model._matrix[leader_goal_index]
                         )
                         .detach()
                         .numpy()
                     )
                leader_msg_index = np.random.choice(range(len(scores)), 1, p=scores)[0]


                with torch.no_grad():
                    scores = (
                         torch.nn.functional.softmax(
                             temper*policies[agent_a.name].model._matrix[:,leader_msg_index]
                         )
                         .detach()
                         .numpy()
                     )
                follower_goal_index = np.random.choice(range(len(scores)), 1, p=scores)[0]


                agent_b_goal = base_env.envs[0].goal_space[leader_goal_index]
                agent_a_goal = base_env.envs[0].goal_space[follower_goal_index]
                message = {
                    agent_b.name: {
                        "leader_goal_index": leader_goal_index,
                        "leader_msg_index": leader_msg_index,
                    },
                    agent_a.name: {
                        "follower_goal_index": follower_goal_index,
                        "leader_msg_index": leader_msg_index,
                    },
                }

            goals = {agent_a.name: agent_a_goal, agent_b.name: agent_b_goal}

            worker.foreach_env(lambda env: env.set_goal_and_message(goals, message))
