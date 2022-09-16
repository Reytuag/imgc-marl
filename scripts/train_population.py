import os
import random
import tempfile
from copy import deepcopy
from datetime import datetime

import click
import imgc_marl.envs.population as population
import numpy as np
import yaml
from imgc_marl.callbacks import PopGoalLinesCallback, PopGoalLinesCommunicationCallback
from imgc_marl.evaluation import (
    custom_eval_function,
    communication_custom_eval_function,
)
from imgc_marl.models.basic_communication import BasicCommunicationNetwork
from imgc_marl.policies.basic_communication import BasicCommunicationTrainer
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
        #"record_env": "videos",
    }
    if use_communication:
        # If we want to evaluate without centralization:
        config["custom_eval_function"] = communication_custom_eval_function
        # If we want to evaluate centralized
        # config["custom_eval_function"] = custom_eval_function
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
    else:
        config["custom_eval_function"] = custom_eval_function
        config["callbacks"] = PopGoalLinesCallback
        trainer = PPOTrainer(
            config=config,
            env=train_env,
            logger_creator=custom_logger_creator,
        )

    # Train for training_steps iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    best_reward = 0.0
    for _ in range(user_config["training"]["training_steps"]):
        result = trainer.train()
        print(pretty_print(keep_relevant_results(result)))
        eval_results = result.get("evaluation")
        if (
            eval_results is not None
            and eval_results["episode_reward_mean"] >= best_reward
        ):
            best_reward = eval_results["episode_reward_mean"]
            save_path = trainer.save()
            print(f"New best model found, saving it in{save_path}")

    # Saving most recent model as well
    trainer.save()


if __name__ == "__main__":
    train()
