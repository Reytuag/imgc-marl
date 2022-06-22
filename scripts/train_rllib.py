import os
import random
import tempfile
from copy import deepcopy
from datetime import datetime

import click
import imgc_marl.envs.multiagent as multiagent
import imgc_marl.envs.single_agent as single_agent
import numpy as np
import yaml
from imgc_marl.callbacks import (GoalLinesCallback, LargeGoalLinesCallback,
                                 NewEnvCallback, after_training_eval_rllib,
                                 goal_lines_last_callback,
                                 legacy_after_training_eval_rllib)
from imgc_marl.evaluation import custom_eval_function
from imgc_marl.utils import keep_relevant_results
from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import UnifiedLogger, pretty_print


@click.command()
@click.option(
    "--environment",
    type=click.Choice(
        ["single_agent", "basic_marl", "goal_lines", "new_env", "large_goal_lines"],
        case_sensitive=True,
    ),
)
@click.argument("config")
@click.argument(
    "custom_logdir", required=False, default="/home/elias/ray_results", type=str
)
@click.argument("seed", required=False, default=None, type=int)
def train(environment, config, custom_logdir, seed):
    """Training loop using RLlib"""

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
    config["evaluation_interval"] = 20
    config["evaluation_num_workers"] = 2

    # Particular settings dependent on the environment
    if environment == "single_agent":
        multiagent_flag = False
        config["horizon"] = single_agent.SIMPLE_TIMELIMIT
        trainer = PPOTrainer(config=config, env=single_agent.SimpleEnv)
        eval_env = single_agent.SimpleEnv()

    elif environment == "basic_marl":
        multiagent_flag = True
        config["horizon"] = multiagent.SIMPLE_TIMELIMIT
        config["env_config"] = user_config["env_config"]
        config["multiagent"] = {
            "policies": {
                "agent_0": PolicySpec(
                    policy_class=None, observation_space=None, action_space=None
                ),
                "agent_1": PolicySpec(
                    policy_class=None, observation_space=None, action_space=None
                ),
            },
            "policy_mapping_fn": lambda agent_id: "agent_0"
            if agent_id.startswith("agent_0")
            else "agent_1",
        }
        trainer = PPOTrainer(config=config, env=multiagent.OneBoxEnv)
        eval_env = multiagent.OneBoxEnv(config["env_config"])

    elif environment == "goal_lines":
        config["horizon"] = multiagent.GOAL_LINES_TIMELIMIT
        config["rollout_fragment_length"] = config["horizon"]
        config["env_config"] = user_config["env_config"]
        config["train_batch_size"] = 16500
        config["callbacks"] = GoalLinesCallback
        config["multiagent"] = {
            "policies": {
                "agent_0": PolicySpec(
                    policy_class=None, observation_space=None, action_space=None
                ),
                "agent_1": PolicySpec(
                    policy_class=None, observation_space=None, action_space=None
                ),
            },
            "policy_mapping_fn": lambda agent_id: "agent_0"
            if agent_id.startswith("agent_0")
            else "agent_1",
        }
        config["custom_eval_function"] = custom_eval_function
        eval_env = multiagent.GoalLinesEnv(config["env_config"])
        goal_space = eval_env.goal_space
        goal_space_dim = eval_env.goal_space_dim
        config["evaluation_config"] = {
            "eval_goals": [{"agent_0": i, "agent_1": i} for i in range(goal_space_dim)],
            "record_env": "videos",
        }
        trainer = PPOTrainer(config=config, env=multiagent.GoalLinesEnv)

    elif environment == "new_env":
        config["horizon"] = multiagent.NEW_ENV_TIMELIMIT
        config["rollout_fragment_length"] = config["horizon"]
        config["env_config"] = user_config["env_config"]
        config["train_batch_size"] = 16_500
        config["callbacks"] = NewEnvCallback
        config["multiagent"] = {
            "policies": {
                "agent_0": PolicySpec(
                    policy_class=None, observation_space=None, action_space=None
                ),
                "agent_1": PolicySpec(
                    policy_class=None, observation_space=None, action_space=None
                ),
            },
            "policy_mapping_fn": lambda agent_id: "agent_0"
            if agent_id.startswith("agent_0")
            else "agent_1",
        }
        config["custom_eval_function"] = custom_eval_function
        eval_env = multiagent.NewEnv(config["env_config"])
        goal_space_dim = eval_env.goal_space_dim
        config["evaluation_config"] = {
            "eval_goals": [{"agent_0": i, "agent_1": i} for i in range(goal_space_dim)],
            "record_env": "videos",
        }
        trainer = PPOTrainer(config=config, env=multiagent.NewEnv)

    elif environment == "large_goal_lines":
        config["evaluation_interval"] = 10
        config["horizon"] = multiagent.LARGE_GOAL_LINES_TIMELIMIT
        config["rollout_fragment_length"] = config["horizon"]
        config["env_config"] = user_config["env_config"]
        config["train_batch_size"] = 60_000
        config["sgd_minibatch_size"] = 10_000
        config["lambda"] = 0.9
        config["lr"] = 0.0003
        config["callbacks"] = LargeGoalLinesCallback
        config["multiagent"] = {
            "policies": {
                "agent_0": PolicySpec(
                    policy_class=None, observation_space=None, action_space=None
                ),
                "agent_1": PolicySpec(
                    policy_class=None, observation_space=None, action_space=None
                ),
            },
            "policy_mapping_fn": lambda agent_id: "agent_0"
            if agent_id.startswith("agent_0")
            else "agent_1",
        }
        config["custom_eval_function"] = custom_eval_function
        eval_env = multiagent.VeryLargeGoalLinesEnv(config["env_config"])
        goal_space = eval_env.goal_space
        goal_space_dim = eval_env.goal_space_dim
        config["evaluation_config"] = {
            "eval_goals": [{"agent_0": i, "agent_1": i} for i in range(goal_space_dim)],
            # "record_env": "videos",
        }
        trainer = PPOTrainer(
            config=config,
            env=multiagent.VeryLargeGoalLinesEnv,
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
    # End of training callbacks + evaluation
    # restoring best model from training
    trainer.restore(save_path)
    if environment == "goal_lines":
        goal_lines_last_callback(trainer, goal_space_dim)
        after_training_eval_rllib(
            trainer,
            eval_env,
            goal_list=config["evaluation_config"]["eval_goals"],
        )
    elif environment == "new_env" or environment == "large_goal_lines":
        after_training_eval_rllib(
            trainer,
            eval_env,
            goal_list=config["evaluation_config"]["eval_goals"],
        )
    else:
        legacy_after_training_eval_rllib(
            trainer,
            eval_env,
            multiagent=multiagent_flag,
        )


if __name__ == "__main__":
    train()
