import random
from copy import deepcopy

import click
import imgc_marl.envs.multiagent as multiagent
import imgc_marl.envs.single_agent as single_agent
import numpy as np
from imgc_marl.utils import after_training_eval_rllib
from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print

SEED = 42
TRAINING_STEPS = 500
NUM_WORKERS = 8


@click.command()
@click.option(
    "--environment",
    type=click.Choice(
        ["single_agent", "basic_marl", "goal_lines"],
        case_sensitive=True,
    ),
)
def train(environment):
    """Training loop using RLlib"""
    random.seed(SEED)
    np.random.seed(SEED)
    # Configure the algorithm.
    config = DEFAULT_CONFIG.copy()
    config["num_workers"] = NUM_WORKERS
    config["framework"] = "torch"
    config["seed"] = SEED
    multiagent_flag = True
    goal_dict = None

    # Train
    if environment == "single_agent":
        multiagent_flag = False
        config["horizon"] = single_agent.SIMPLE_TIMELIMIT
        # Create our RLlib Trainer.
        trainer = PPOTrainer(config=config, env=single_agent.SimpleEnv)
        eval_env = single_agent.SimpleEnv()

    elif environment == "basic_marl":
        config["horizon"] = multiagent.SIMPLE_TIMELIMIT
        config["env_config"] = {
            "n_agents": 2,
            "continuous": True,
        }
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
        goal_dict = {"agent_0": list(range(6)), "agent_1": list(range(6))}
        config["horizon"] = multiagent.GOAL_LINES_TIMELIMIT
        config["env_config"] = {"continuous": True, "centralized": True}
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

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    for _ in range(TRAINING_STEPS):
        result = trainer.train()
        print(pretty_print(result))

    # After training has completed, evaluate the agent
    after_training_eval_rllib(
        trainer, eval_env, multiagent=multiagent_flag, goal_dict=goal_dict
    )


if __name__ == "__main__":
    train()
