import os
from typing import Any, Dict, List

import cv2
import gym
import moviepy.video.io.ImageSequenceClip
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from imgc_marl.envs.multiagent import POSSIBLE_GOAL_LINES

# from stable_baselines3.common.logger import Video

font = cv2.FONT_HERSHEY_SIMPLEX


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        logdir: str,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env``
        """
        super().__init__()
        self._eval_env = eval_env
        self.logdir = logdir
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> None:
        return True

    def _on_training_end(self) -> bool:
        print("----Final Evaluation----")
        screens = []

        def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
            """
            Renders the environment in its current state, recording the screen in the captured `screens` list

            :param _locals: A dictionary containing all local variables of the callback's scope
            :param _globals: A dictionary containing all global variables of the callback's scope
            """
            screen = self._eval_env.render()
            # # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
            # screens.append(screen.transpose(2, 0, 1))
            screens.append(screen)

        mean_reward, std_reward = evaluate_policy(
            self.model,
            self._eval_env,
            callback=grab_screens,
            n_eval_episodes=self._n_eval_episodes,
            deterministic=self._deterministic,
        )

        # TODO: resolve issue when logging images to tensorboard
        # video_array = th.ByteTensor(np.array(screens))
        # self.logger.record(
        #     "trajectory/video",
        #     Video(video_array, fps=40),
        #     exclude=("stdout", "log", "json", "csv"),
        # )
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(screens, fps=30)
        clip.write_videofile(os.path.join(self.logdir, "trained_agent.mp4"))

        print(f"Reward: {mean_reward}+-{std_reward}")
        return True


def after_training_eval_rllib(
    trainer,
    eval_env: gym.Env,
    eval_episodes: int = 5,
    multiagent: bool = True,
    goal_dict: Dict[str, List] = None,
) -> None:
    """Final evaluation function called after rllib training loop"""
    frames = []
    if goal_dict is not None:
        eval_episodes = len(list(goal_dict.values())[0])
    for n in range(eval_episodes):
        done = False
        if goal_dict is not None:
            obs = eval_env.reset({agent: goal[n] for agent, goal in goal_dict.items()})
            # hack to print the evaluation goal. this is super tied to the goal lines env
            # TODO: refactor this in the future in a more general and flexible way not tied
            # to the environment!
            goals = [POSSIBLE_GOAL_LINES[n] for n in list(goal_dict.values())[0]]
        else:
            obs = eval_env.reset()
        episode_reward = 0.0
        while not done:
            frame = eval_env.render()
            if goal_dict is not None:
                cv2.putText(
                    frame,
                    str(goals[n]),
                    (10, 35),
                    font,
                    1,
                    (255, 255, 255),
                    1,
                )
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
