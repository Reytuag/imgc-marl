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
@click.argument(
    "custom_logdir", required=False, default="/home/elias/ray_results", type=str
)
@click.argument("seed", required=False, default=None, type=int)
def train(environment, config, custom_logdir, seed):
    """Training loop using RLlib for a population of agents"""

    def custom_logger_creator(config):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

        if not os.path.exists(custom_logdir):
            os.makedirs(custom_logdir)
        logdir = tempfile.mkdtemp(prefix=timestr, dir=custom_logdir)
        return UnifiedLogger(config, logdir, loggers=None)

    # Loading user config
    with open(config, "r") as f:
        user_config = yaml.safe_load(f)
    # Seeding everything
    if seed is None:
        seed = random.randint(0, 1e6)
    random.seed(seed)
    np.random.seed(seed)

    # General settings of the algorithm
    config = deepcopy(DEFAULT_CONFIG)
    config["num_workers"] = user_config["training"].get("num_workers", 0)
    config["framework"] = "torch"
    config["seed"] = seed
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
        # "record_env": "videos",
    }
    if use_communication == "basic":
        # If we want to evaluate without centralization:
        # config["custom_eval_function"] = communication_custom_eval_function
        # If we want to evaluate centralized
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
        # config["custom_eval_function"] = communication_custom_eval_function
        # If we want to evaluate centralized
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
                "nb_msg":user_config.get("nb_msg",30)
            },
        }
        trainer = FullNamingTrainer1Matrix(
            config=config,
            env=train_env,
            logger_creator=custom_logger_creator,
        )
        
    elif use_communication == "naming_2":
        # If we want to evaluate without centralization:
        # config["custom_eval_function"] = communication_custom_eval_function
        # If we want to evaluate centralized
        config["custom_eval_function"] = custom_eval_function
        config["callbacks"] = PopGoalLinesNamingCallback
        ModelCatalog.register_custom_model(
            "FullNamingNetwork", FullNamingNetwork
        )
        config["model"] = {
            "custom_model": "FullNamingNetwork",
            "custom_model_config": {
                "number_of_goals": goal_space_dim,
                "train_matrix": user_config.get("train_matrix", False),
            },
        }
        trainer = FullNamingTrainer(
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
    # load previously trained checkpoint
    if user_config.get("checkpoint") is not None:
        trainer.restore(user_config.get("checkpoint"))
    # Train for training_steps iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    for _ in range(user_config["training"]["training_steps"]):
        result = trainer.train()
        print(pretty_print(keep_relevant_results(result)))
        eval_results = result.get("evaluation")
        # Saving a checkpoint each evaluation round
        if eval_results is not None:
            trainer.save()

    # Saving most recent model as well
    trainer.save()


if __name__ == "__main__":
    train()
