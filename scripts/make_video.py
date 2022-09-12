import os
import shutil
from copy import deepcopy

import click
import yaml
from imgc_marl.callbacks import after_training_eval_rllib
from imgc_marl.envs import multiagent
from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer
from imgc_marl.policies import (
    BasicCommunicationTrainer,
    FullCommunicationTrainer,
    BasicNamingTrainer,
)
from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from imgc_marl.callbacks import (
    GoalLinesCallback,
    LargeGoalLinesCallback,
    LargeGoalLinesBasicCommunicationCallback,
    LargeGoalLinesFullCommunicationCallback,
    NewEnvCallback,
    LargeGoalLinesBasicNamingGame,
    after_training_eval_rllib,
    goal_lines_last_callback,
    legacy_after_training_eval_rllib,
)
from imgc_marl.evaluation import custom_eval_function
from imgc_marl.models import (
    BasicCommunicationNetwork,
    FullCommunicationNetwork,
    BasicNamingNetwork,
)


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
        if use_communication == "basic":
            config["callbacks"] = LargeGoalLinesBasicCommunicationCallback
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
                env=multiagent.VeryLargeGoalLinesEnv,
            )
        if use_communication == "full":
            config["callbacks"] = LargeGoalLinesFullCommunicationCallback
            ModelCatalog.register_custom_model(
                "FullCommunicationNetwork", FullCommunicationNetwork
            )
            config["model"] = {
                "custom_model": "FullCommunicationNetwork",
                "custom_model_config": {
                    "input_dim": goal_repr_dim + goal_space_dim,
                },
            }
            trainer = FullCommunicationTrainer(
                config=config,
                env=multiagent.VeryLargeGoalLinesEnv,
            )
        if use_communication == "basic_naming":
            config["callbacks"] = LargeGoalLinesBasicNamingGame
            ModelCatalog.register_custom_model("BasicNamingNetwork", BasicNamingNetwork)
            config["model"] = {
                "custom_model": "BasicNamingNetwork",
                "custom_model_config": {
                    "number_of_goals": goal_space_dim,
                    "train_matrix": user_config.get("train_matrix", False),
                },
            }
            trainer = BasicNamingTrainer(
                config=config,
                env=multiagent.VeryLargeGoalLinesEnv,
            )
            if user_config.get("checkpoint") is not None:
                trainer.restore(user_config.get("checkpoint"))
    else:
        config["callbacks"] = LargeGoalLinesCallback
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
