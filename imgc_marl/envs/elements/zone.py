from abc import ABC
from typing import Optional, Union

from simple_playgrounds.common.definitions import (
    SIMULATION_STEPS,
    CollisionTypes,
    ElementTypes,
)
from simple_playgrounds.configs.parser import parse_configuration
from simple_playgrounds.element.element import InteractiveElement
from simple_playgrounds.element.elements.zone import ZoneElement


class VisibleZoneElement(InteractiveElement, ABC):
    """Base Class for Contact Entities"""

    def __init__(self, **entity_params):

        InteractiveElement.__init__(
            self, visible_shape=True, invisible_shape=True, **entity_params
        )

    def _set_shape_collision(self):
        self.pm_invisible_shape.collision_type = CollisionTypes.CONTACT


class MultiAgentRewardZone(VisibleZoneElement, ABC):
    """
    Reward Zones provide a reward to all the agents in the zone.
    """

    def __init__(
        self,
        reward: float,
        limit: Optional[float] = None,
        config_key: Optional[Union[ElementTypes, str]] = None,
        **entity_params
    ):
        """
        MultiAgentRewardZone entities are invisible zones.
        Provide a reward to the agent which is inside the zone.

        Args:
            **kwargs: other params to configure entity. Refer to Entity class

        Keyword Args:
            reward: Reward provided at each timestep when agent is in the zone
            total_reward: Total reward that the entity can provide during an Episode
        """
        default_config = parse_configuration("element_zone", config_key)
        entity_params = {**default_config, **entity_params}

        super().__init__(reward=reward, **entity_params)

        self._limit = limit
        self._total_reward_provided = 0

    @property
    def reward(self):
        # Provide reward in all steps in all interactions. In RewardZone only provided
        # reward during first step and first interaction, so it doesn't work for more than 1 agent
        rew = self._reward

        if self._limit:
            reward_left = self._limit - self._total_reward_provided

            if abs(rew) > abs(reward_left):
                rew = reward_left

        self._total_reward_provided += rew
        return rew

    @reward.setter
    def reward(self, rew: float):
        self._reward = rew

    def reset(self):
        self._total_reward_provided = 0
        super().reset()

    @property
    def terminate_upon_activation(self):
        return False

    def activate(self, *args):
        return None, None
