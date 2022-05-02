import os
import random

import click
import imgc_marl.envs.single_agent as single_agent
import numpy as np
from imgc_marl.utils import VideoRecorderCallback
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

SEED = 42
TIMESTEPS = 7.5e4
EVAL_FREQ = 7.5e3

ENVIRONMENTS = {
    "single_agent": {
        "env": single_agent.SimpleEnv,
        "timelimit": single_agent.SIMPLE_TIMELIMIT,
    },
    "goal_conditioned": {
        "env": single_agent.MultiGoalEnv,
        "timelimit": single_agent.GOALCONDITIONED_TIMELIMIT,
    },
    "goal_conditioned_HER": {
        "env": single_agent.MultiGoalEnv,
        "timelimit": single_agent.GOALCONDITIONED_TIMELIMIT,
    },
}


@click.command()
@click.option("--logdir", help="Directory to store training logs.")
@click.option(
    "--environment",
    type=click.Choice(
        ["single_agent", "goal_conditioned", "goal_conditioned_HER"],
        case_sensitive=True,
    ),
)
def train(logdir, environment):
    """Training loop"""

    random.seed(SEED)
    np.random.seed(SEED)
    # Create our env
    env = Monitor(ENVIRONMENTS[environment]["env"](), info_keywords=("is_success",))
    timelimit = ENVIRONMENTS[environment]["timelimit"]
    # Sanity check from stablebaselines3
    check_env(env)

    # Create log directory
    os.makedirs(logdir)

    # Train
    if environment == "single_agent":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, seed=SEED)
    elif environment == "goal_conditioned":
        model = PPO(
            "MultiInputPolicy", env, verbose=1, tensorboard_log=logdir, seed=SEED
        )
    elif environment == "goal_conditioned_HER":
        model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
                online_sampling=True,
                max_episode_length=timelimit,
            ),
            verbose=1,
            tensorboard_log=logdir,
            seed=SEED,
        )

    eval_callback = EvalCallback(
        env, log_path=logdir, eval_freq=EVAL_FREQ, deterministic=True, render=False
    )

    video_recorder = VideoRecorderCallback(env, logdir=logdir)
    callback = CallbackList([eval_callback, video_recorder])
    model.learn(total_timesteps=int(TIMESTEPS), callback=callback)


if __name__ == "__main__":
    train()
