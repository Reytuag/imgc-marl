from curses.ascii import DEL
import logging
from typing import Dict, List, Type, Union

import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTorchPolicy, PPOTrainer
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
)
from ray.rllib.utils.typing import TensorType

DELTA = 0.1 / (30 * 60)
# scaling factor: 30 iterations x 60 episodes each agent will lead (60 games/training it)
ALPHA= 0.1
torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

communication_criterion = nn.MSELoss()


class FullNamingTrainer1Matrix(PPOTrainer):
    def get_default_policy_class(self, config):
        return FullNamingPolicy1Matrix


class FullNamingPolicy1Matrix(PPOTorchPolicy):
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Constructs the loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = 0
            vf_loss_clipped = mean_vf_loss = 0.0
        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_policy_loss
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        # Add custom naming game weight update
        # only if specified we need to train it
        if model._train_matrix:
            if isinstance(train_batch["infos"][0], Dict):
                update=torch.zeros_like(model._matrix)
                normalization=torch.zeros_like(model._matrix)
                for i, info in enumerate(train_batch["infos"]):
                    m = info.get("message")
                    if m is not None:
                        if m.get("leader_goal_index") is not None:
                            # agent was a leader
                            leader_goal_index = m["leader_goal_index"]
                            leader_msg_index = m["leader_msg_index"]
                            if train_batch["rewards"][i]:
                                with torch.no_grad():
                                    update[
                                        leader_goal_index, leader_msg_index
                                    ] += train_batch["rewards"][i]
                            normalization[leader_goal_index, leader_msg_index] +=1
                        else:
                            # agent was a follower, update follower matrix
                            follower_goal_index = m["follower_goal_index"]
                            leader_msg_index = m["leader_msg_index"]
                            if train_batch["rewards"][i]:
                                with torch.no_grad():
                                    update[
                                        follower_goal_index, leader_msg_index
                                    ] += train_batch["rewards"][i]
                            normalization[follower_goal_index, leader_msg_index]+=1
                with torch.no_grad():
                    model._matrix*=(1-ALPHA)
                    model._matrix+=ALPHA*update/(normalization+1e-10)

        return total_loss
