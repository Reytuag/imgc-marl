from copy import deepcopy
import os
from typing import Any, Dict

import gym
import moviepy.video.io.ImageSequenceClip
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# from stable_baselines3.common.logger import Video


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


def keep_relevant_results(results):
    results_to_print = deepcopy(results)
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
        "timesteps_total",
        "timesteps_this_iter",
        "episodes_total",
        "training_iteration",
        "time_this_iter_s",
        "time_total_s",
    ]

    keep_custom_keys = [
        "reward for collective goal_mean",
        "reward for collective goal_min",
        "reward for collective goal_max",
        "reward for individual goal_mean",
        "reward for individual goal_min",
        "reward for individual goal_max",
    ]
    results_to_print["custom_metrics"] = {
        key: value
        for key, value in results_to_print["custom_metrics"].items()
        if key in keep_custom_keys
    }
    return {key: value for key, value in results_to_print.items() if key in keep_keys}
