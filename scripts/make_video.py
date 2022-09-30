import os
import random
import tempfile
from copy import deepcopy
from datetime import datetime

import click
import imgc_marl.envs.population as population
import numpy as np
import yaml
from imgc_marl.callbacks import (
    PopGoalLinesCallback,
    PopGoalLinesCommunicationCallback,
    PopGoalLinesNamingCallback,
    PopGoalLinesNamingCallback1Matrix,
)
from imgc_marl.evaluation import (
    communication_custom_eval_function,
    custom_eval_function,
)
from imgc_marl.models.basic_communication import BasicCommunicationNetwork
from imgc_marl.models.full_naming_game import FullNamingNetwork
from imgc_marl.models.full_naming_game_single_matrix import FullNamingNetwork1Matrix
from imgc_marl.policies.basic_communication import BasicCommunicationTrainer
from imgc_marl.policies.full_naming_game import FullNamingTrainer
from imgc_marl.policies.full_naming_game_single_matrix import FullNamingTrainer1Matrix
from imgc_marl.utils import keep_relevant_results
from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import UnifiedLogger, pretty_print


@click.command()
@click.option("--environment", type=click.Choice(["goal_lines", "large_goal_lines"]))
@click.argument("config")
@click.argument("checkpoint")
@click.argument("video_dir", required=True, type=str)
@click.option("--evaluate-communication/--no-evaluate-communication", default=False)
def make_video(environment, config, checkpoint, video_dir, evaluate_communication):
    """Evaluate trained agents and make a video of their performance"""

    def custom_logger_creator(config):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        logdir = tempfile.mkdtemp(prefix=timestr, dir=video_dir)
        return UnifiedLogger(config, logdir, loggers=None)

    # Loading user config
    with open(config, "r") as f:
        user_config = yaml.safe_load(f)

    # General settings of the algorithm
    # General settings of the algorithm
    config = deepcopy(DEFAULT_CONFIG)
    config["num_workers"] = user_config["training"].get("num_workers", 0)
    config["framework"] = "torch"
    config["evaluation_interval"] = 10
    config["evaluation_num_workers"] = 2
    use_communication = user_config.get("communication", False)

    config["horizon"] = population.LARGE_GOAL_LINES_TIMELIMIT
    config["rollout_fragment_length"] = config["horizon"]
    config["env_config"] = user_config["env_config"]
    config["train_batch_size"] = 60_000
    config["sgd_minibatch_size"] = 10_000
    config["lambda"] = 0.9
    config["lr"] = 0.0003

    def policy_mapping_fn(agent_id):
        return agent_id

    config["multiagent"] = {
        "policies": {
            f"agent_{i}": PolicySpec(
                policy_class=None, observation_space=None, action_space=None
            )
            for i in range(config["env_config"]["population_size"])
        },
        "policy_mapping_fn": policy_mapping_fn,
    }
    if environment == "goal_lines":
        eval_env = population.PopGoalLinesEnv(config["env_config"])
        train_env = population.PopGoalLinesEnv
    elif environment == "large_goal_lines":
        eval_env = population.PopLargeGoalLinesEnv(config["env_config"])
        train_env = population.PopLargeGoalLinesEnv
    goal_space = eval_env.goal_space
    goal_space_dim = eval_env.goal_space_dim
    goal_repr_dim = eval_env.goal_repr_dim
    config["evaluation_config"] = {
        "eval_goals": [
            {f"agent_{n}": i for n in range(config["env_config"]["population_size"])}
            for i in range(goal_space_dim)
        ],
        "record_env": "videos",
    }
    if use_communication == "basic":
        # If we want to evaluate without centralization:
        if evaluate_communication:
            config["custom_eval_function"] = communication_custom_eval_function
        # If we want to evaluate centralized
        else:
            config["custom_eval_function"] = custom_eval_function
        config["callbacks"] = PopGoalLinesCommunicationCallback
        ModelCatalog.register_custom_model(
            "BasicCommunicationNetwork", BasicCommunicationNetwork
        )
        config["model"] = {
            "custom_model": "BasicCommunicationNetwork",
            "custom_model_config": {
                "number_of_messages": goal_space_dim,
                "input_dim": goal_repr_dim,
            },
        }
        trainer = BasicCommunicationTrainer(
            config=config,
            env=train_env,
            logger_creator=custom_logger_creator,
        )
    elif use_communication == "naming":
        # If we want to evaluate without centralization:
        if evaluate_communication:
            config["custom_eval_function"] = communication_custom_eval_function
        # If we want to evaluate centralized
        else:
            config["custom_eval_function"] = custom_eval_function
        config["callbacks"] = PopGoalLinesNamingCallback1Matrix
        ModelCatalog.register_custom_model(
            "FullNamingNetwork1Matrix", FullNamingNetwork1Matrix
        )
        config["model"] = {
            "custom_model": "FullNamingNetwork1Matrix",
            "custom_model_config": {
                "number_of_goals": goal_space_dim,
                "train_matrix": user_config.get("train_matrix", False),
            },
        }
        trainer = FullNamingTrainer1Matrix(
            config=config,
            env=train_env,
            logger_creator=custom_logger_creator,
        )

    else:
        config["custom_eval_function"] = custom_eval_function
        config["callbacks"] = PopGoalLinesCallback
        trainer = PPOTrainer(
            config=config,
            env=train_env,
            logger_creator=custom_logger_creator,
        )
    trainer.restore(checkpoint)
    trainer.evaluate()


if __name__ == "__main__":
    make_video()
