import os
import random

import click
import imgc_marl.envs.single_agent as single_agent
import numpy as np
from imgc_marl.utils import VideoRecorderCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

SEED = 42

ENVIRONMENTS = {
    "single_agent": {
        "env": single_agent.SimpleEnv,
        "timelimit": single_agent.TIMELIMIT,
    },
    "goal_conditioned": None,
}

ALGORITHMS = {
    "PPO": PPO,
}


@click.command()
@click.option("--logdir", help="Directory to store training logs.")
@click.option(
    "--environment",
    type=click.Choice(["single_agent", "goal_conditioned"], case_sensitive=True),
)
@click.option("--algorithm", type=click.Choice(["PPO"]))
def train(logdir, environment, algorithm):
    """Training loop"""

    random.seed(SEED)
    np.random.seed(SEED)
    # Create our env
    env = Monitor(ENVIRONMENTS[environment]["env"]())
    timelimit = ENVIRONMENTS[environment]["timelimit"]
    # Sanity check from stablebaselines3
    check_env(env)

    # Create log directory
    os.makedirs(logdir)

    # Train
    model = ALGORITHMS.get(algorithm)(
        "MlpPolicy", env, verbose=1, tensorboard_log=logdir, seed=SEED
    )
    video_recorder = VideoRecorderCallback(env, logdir=logdir)
    model.learn(total_timesteps=int(150_000), callback=video_recorder)


if __name__ == "__main__":
    train()
