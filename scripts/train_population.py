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
from imgc_marl.models.centralized_critic_models import TorchCentralizedCriticModel
from imgc_marl.policies.basic_communication import BasicCommunicationTrainer
from imgc_marl.policies.full_naming_game import FullNamingTrainer
from imgc_marl.policies.full_naming_game_single_matrix import FullNamingTrainer1Matrix
from imgc_marl.utils import keep_relevant_results
from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer

from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import UnifiedLogger, pretty_print


import argparse
import numpy as np
from gym.spaces import Discrete
import os

import ray
from ray import tune
from ray.rllib.agents.maml.maml_torch_policy import KLCoeffMixin as TorchKLCoeffMixin
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import (
    PPOTFPolicy,
    KLCoeffMixin,
    ppo_surrogate_loss as tf_loss,
)
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.examples.env.two_step_game import TwoStepGame
#from ray.rllib.examples.models.centralized_critic_models import TorchCentralizedCriticModel

from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import (
    LearningRateSchedule as TorchLR,
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_utils import explained_variance, make_tf_callable
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"




class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        if self.config["framework"] != "torch":
            self.compute_central_vf = make_tf_callable(self.get_session())(
                self.model.central_value_function
            )
        else:
            self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(
    policy, sample_batch, other_agent_batches=None, episode=None
):
    pytorch = policy.config["framework"] == "torch"
    if (pytorch and hasattr(policy, "compute_central_vf")) or (
        not pytorch and policy.loss_initialized()
    ):
        assert other_agent_batches is not None
        [(_, opponent_batch)] = list(other_agent_batches.values())

        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

        # overwrite default VF prediction with the central VF
        if policy.config["framework"]  == "torch":
            sample_batch[SampleBatch.VF_PREDS] = (
                policy.compute_central_vf(
                    convert_to_torch_tensor(
                        sample_batch[SampleBatch.CUR_OBS], policy.device
                    ),
                    convert_to_torch_tensor(sample_batch[OPPONENT_OBS], policy.device),
                    convert_to_torch_tensor(
                        sample_batch[OPPONENT_ACTION], policy.device
                    ),
                )
                .cpu()
                .detach()
                .numpy()
            )
        else:
            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                sample_batch[SampleBatch.CUR_OBS],
                sample_batch[OPPONENT_OBS],
                sample_batch[OPPONENT_ACTION],
            )
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32
        )

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = tf_loss if not policy.config["framework"] == "torch" else PPOTorchPolicy.loss

    vf_saved = model.value_function
    #print(train_batch[SampleBatch.CUR_OBS].shape,train_batch[OPPONENT_OBS].shape,train_batch[OPPONENT_ACTION].shape)
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch[OPPONENT_OBS],
        train_batch[OPPONENT_ACTION],
    )

    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss


def setup_tf_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTFPolicy (w/o ValueNetworkMixin).
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(
        policy, config["entropy_coeff"], config["entropy_coeff_schedule"]
    )
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    TorchKLCoeffMixin.__init__(policy, config)
    TorchEntropyCoeffSchedule.__init__(
        policy, config["entropy_coeff"], config["entropy_coeff_schedule"]
    )
    TorchLR.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy._central_value_out
        )
    }


CCPPOTFPolicy = PPOTFPolicy.with_updates(
    name="CCPPOTFPolicy",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_loss_init=setup_tf_mixins,
    grad_stats_fn=central_vf_stats,
    mixins=[
        LearningRateSchedule,
        EntropyCoeffSchedule,
        KLCoeffMixin,
        CentralizedValueMixin,
    ],
)


class CCPPOTorchPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.compute_central_vf = self.model.central_value_function

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return loss_with_central_critic(self, model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return centralized_critic_postprocessing(
            self, sample_batch, other_agent_batches, episode
        )


class CCTrainer(PPOTrainer):
    @override(PPOTrainer)
    def get_default_policy_class(self, config):
        if config["framework"] == "torch":
            return CCPPOTorchPolicy
        else:
            return CCPPOTFPolicy








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
                "nb_msg": user_config.get("nb_msg",30)
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
    elif use_communication == "centralized_critic":
        print("centralized_critic")
        ModelCatalog.register_custom_model(
            "cc_model",
            TorchCentralizedCriticModel
            if config["framework"] == "torch"
            else CentralizedCriticModel,
        )
        config["model"]={
            "custom_model": "cc_model",
        }

        config["custom_eval_function"] = custom_eval_function
        config["callbacks"] = PopGoalLinesCallback
        trainer = CCTrainer(
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
