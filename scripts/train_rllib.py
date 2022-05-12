import random
from copy import deepcopy

import click
import imgc_marl.envs.multiagent as multiagent
import imgc_marl.envs.single_agent as single_agent
import numpy as np
import yaml
from imgc_marl.callbacks import MultiGoalMultiAgentCallbacks
from imgc_marl.utils import after_training_eval_rllib
from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print


@click.command()
@click.option(
    "--environment",
    type=click.Choice(
        ["single_agent", "basic_marl", "goal_lines"],
        case_sensitive=True,
    ),
)
@click.argument("config")
def train(environment, config):
    """Training loop using RLlib"""

    # Loading user config
    with open(config, "r") as f:
        user_config = yaml.safe_load(f)
    # Seeding everything
    seed = user_config.get("seed", random.randint(0, 1e6))
    random.seed(seed)
    np.random.seed(seed)

    # General settings of the algorithm
    config = DEFAULT_CONFIG.copy()
    config["num_workers"] = user_config["training"].get("num_workers", 0)
    config["framework"] = "torch"
    config["seed"] = seed

    # Particular settings dependent on the environment
    if environment == "single_agent":
        multiagent_flag = False
        goal_dict = None
        config["horizon"] = single_agent.SIMPLE_TIMELIMIT
        trainer = PPOTrainer(config=config, env=single_agent.SimpleEnv)
        eval_env = single_agent.SimpleEnv()

    elif environment == "basic_marl":
        multiagent_flag = True
        goal_dict = None
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
        multiagent_flag = True
        goal_dict = {
            "agent_0": list(range(multiagent.N_GOAL_LINES)),
            "agent_1": list(range(multiagent.N_GOAL_LINES)),
        }
        config["horizon"] = multiagent.GOAL_LINES_TIMELIMIT
        config["env_config"] = user_config["env_config"]
        config["callbacks"] = MultiGoalMultiAgentCallbacks
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
        eval_config = deepcopy(config)
        trainer = PPOTrainer(config=config, env=multiagent.GoalLinesEnv)
        eval_env = multiagent.GoalLinesEnv(eval_config["env_config"])

    # Train for training_steps iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    for _ in range(user_config["training"]["training_steps"]):
        result = trainer.train()
        print(pretty_print(result))

    # After training has completed, evaluate the agent
    after_training_eval_rllib(
        trainer, eval_env, multiagent=multiagent_flag, goal_dict=goal_dict
    )


if __name__ == "__main__":
    train()
