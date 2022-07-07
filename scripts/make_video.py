import os
import shutil
from copy import deepcopy

import click
import yaml
from imgc_marl.callbacks import after_training_eval_rllib
from imgc_marl.envs import multiagent
from imgc_marl.models import CustomNetwork
from imgc_marl.policy import CustomPPOTrainer
from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec


@click.command()
@click.argument("checkpoint", required=True, type=str)
@click.argument("config")
def make_video(checkpoint, config):
    video_path = os.path.dirname(os.path.dirname(checkpoint))

    # Loading user config
    with open(config, "r") as f:
        user_config = yaml.safe_load(f)

    # General settings of the algorithm
    config = deepcopy(DEFAULT_CONFIG)
    config["num_workers"] = 0
    config["framework"] = "torch"

    # Environment settings
    config["env_config"] = user_config["env_config"]
    use_communication = user_config.get("communication", False)
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
    env = multiagent.VeryLargeGoalLinesEnv(config["env_config"])
    goal_space_dim = env.goal_space_dim
    goal_repr_dim = env.goal_repr_dim
    config["evaluation_config"] = {
        "eval_goals": [{"agent_0": i, "agent_1": i} for i in range(goal_space_dim)],
    }
    config["log_level"] = "ERROR"
    if use_communication:
        ModelCatalog.register_custom_model("CustomNetwork", CustomNetwork)
        config["model"] = {
            "custom_model": "CustomNetwork",
            "custom_model_config": {
                "number_of_messages": goal_space_dim,
                "input_dim": goal_repr_dim,
            },
        }
        trainer = CustomPPOTrainer(
            config=config,
            env=multiagent.VeryLargeGoalLinesEnv,
        )
    else:
        trainer = PPOTrainer(
            config=config,
            env=multiagent.VeryLargeGoalLinesEnv,
        )

    # Load trained model
    trainer.restore(checkpoint)

    # Generate video and evaluate
    after_training_eval_rllib(
        trainer, env, config["evaluation_config"]["eval_goals"], video_path
    )

    # Remove logs
    shutil.rmtree(trainer.logdir)


if __name__ == "__main__":
    make_video()
