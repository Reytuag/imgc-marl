# Supressing pygame greeting msg
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = ""

import random

import gym
import numpy as np
from gym import spaces
from imgc_marl.envs.elements.activable import CustomRewardOnActivation
from simple_playgrounds.agent.actuators import ContinuousActuator
from simple_playgrounds.agent.agents import BaseAgent
from simple_playgrounds.agent.controllers import External
from simple_playgrounds.common.position_utils import CoordinateSampler
from simple_playgrounds.device.sensors.semantic import PerfectSemantic
from simple_playgrounds.element.elements.basic import Wall
from simple_playgrounds.engine import Engine
from simple_playgrounds.playground.layouts import SingleRoom

SIMPLE_PLAYGROUND = GOALCONDITIONED_PLAYGROUND = (200, 200)
SIMPLE_TIMELIMIT = 100
GOALCONDITIONED_TIMELIMIT = 50


class SimpleEnv(gym.Env):
    """Simple Environment that follows gym interface using spg: one agent, single goal.
    Spawns an object that gives a reward to the agent when activated.
    """

    def __init__(self, continuous=True):
        super(SimpleEnv, self).__init__()

        self.continuous = continuous

        self.episodes = 0
        self.time_steps = 0
        # Create playground
        # Minimal environment with 1 room and 1 goal
        self.playground = SingleRoom(
            size=SIMPLE_PLAYGROUND,
        )
        room = self.playground.grid_rooms[0][0]
        center_area, size_area = room.get_partial_area("up-left")
        spawn_area_fountain = CoordinateSampler(
            center_area, area_shape="rectangle", size=size_area
        )
        fountain = CustomRewardOnActivation(reward=1, terminate=True)
        self.playground.add_element(
            fountain, spawn_area_fountain, allow_overlapping=False
        )

        # Init the agent
        self.agent = BaseAgent(controller=External(), interactive=True)
        # Add sensor
        ignore_elements = [
            elem for elem in self.playground.elements if isinstance(elem, Wall)
        ] + [self.agent.parts, self.agent.base_platform]
        self.agent.add_sensor(
            PerfectSemantic(
                self.agent.base_platform,
                invisible_elements=ignore_elements,
                min_range=0,
                max_range=max(SIMPLE_PLAYGROUND),
                name="sensor",
                normalize=True,
            )
        )
        # Add agent to playground
        center_area, size_area = room.get_partial_area("down-right")
        spawn_area_agent = CoordinateSampler(
            center_area, area_shape="rectangle", size=size_area
        )
        self.playground.add_agent(self.agent, spawn_area_agent)

        # Init engine
        self.engine = Engine(playground=self.playground, time_limit=SIMPLE_TIMELIMIT)

        # Define action and observation space
        actuators = self.agent.controller.controlled_actuators
        if not continuous:
            # Discrete action space
            act_spaces = []
            for actuator in actuators:
                if isinstance(actuators, ContinuousActuator):
                    act_spaces.append(3)
                else:
                    act_spaces.append(2)
            self.action_space = spaces.MultiDiscrete(act_spaces)
        else:
            # Continuous action space
            lows = []
            highs = []
            for actuator in actuators:
                lows.append(actuator.min)
                highs.append(actuator.max)

            self.action_space = spaces.Box(
                low=np.array(lows).astype(np.float32),
                high=np.array(highs).astype(np.float32),
                dtype=np.float32,
            )

        # Continuous observation space
        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi]), high=np.array([1, np.pi]), dtype=np.float64
        )

    def process_obs(self):
        """Process observations to match Gym API"""
        try:
            obs = np.array(
                [
                    (v.distance, v.angle)
                    for v in list(self.agent.observations.values())[0]
                ],
                dtype=np.float32,
            )[0, :]
        except:
            # this avoid the error when the agent is over the objects and gets empty observations!
            obs = np.zeros(self.observation_space._shape)
        return obs

    def step(self, action):
        actions_dict = {}
        actuators = self.agent.controller.controlled_actuators

        if not self.continuous:
            for actuator, act in zip(actuators, action):
                if isinstance(actuator, ContinuousActuator):
                    actions_dict[actuator] = [-1, 0, 1][act]
                else:
                    actions_dict[actuator] = [0, 1][act]
        else:
            for actuator, act in zip(actuators, action):
                if isinstance(actuator, ContinuousActuator):
                    actions_dict[actuator] = act
                else:
                    actions_dict[actuator] = round(act)

        self.engine.step({self.agent: actions_dict})
        self.engine.update_observations()
        reward = self.agent.reward
        info = {"is_success": reward > 0}
        done = self.playground.done or not self.engine.game_on
        observations = self.process_obs()
        return observations, reward, done, info

    def reset(self):
        self.engine.reset()
        self.engine.elapsed_time = 0
        self.episodes += 1
        self.engine.update_observations()
        self.time_steps = 0
        observations = self.process_obs()
        return observations

    def render(self, mode=None):
        return (255 * self.engine.generate_agent_image(self.agent)).astype(np.uint8)

    def close(self):
        self.engine.terminate()


class MultiGoalEnv(gym.GoalEnv):
    """Multi-goal Environment that follows gym interface using spg.
    Spawns 3 objects in each episode only one of them gives the agent a reward and ends the episode.
    """

    def __init__(self):
        super(MultiGoalEnv, self).__init__()

        self.episodes = 0
        self.time_steps = 0

        # Create playground
        # Minimal environment with 1 room and 3 goals
        self.playground = SingleRoom(
            size=GOALCONDITIONED_PLAYGROUND,
        )
        room = self.playground.grid_rooms[0][0]

        # Only one object will end the episode
        fountain_with_reward = random.choice([0, 1, 2])
        position = [((50, 50), 0), ((100, 50), 0), ((150, 50), 0)]
        for f in range(3):
            if f == fountain_with_reward:
                fountain = CustomRewardOnActivation(
                    reward=1, terminate=True, name=str(f)
                )
            else:
                fountain = CustomRewardOnActivation(
                    reward=0, terminate=False, name=str(f)
                )
            self.playground.add_element(fountain, position[f])
        self._current_goal = np.zeros(3, dtype=np.uint8)
        self._current_goal[fountain_with_reward] = 1
        self.playground._achieved_goal = np.zeros(3, dtype=np.uint8)

        # Init the agent
        self.agent = BaseAgent(controller=External(), interactive=True)
        # Add sensor
        ignore_elements = [
            elem for elem in self.playground.elements if isinstance(elem, Wall)
        ] + [self.agent.parts, self.agent.base_platform]
        self.agent.add_sensor(
            PerfectSemantic(
                self.agent.base_platform,
                invisible_elements=ignore_elements,
                min_range=0,
                max_range=max(GOALCONDITIONED_PLAYGROUND),
                name="sensor",
                normalize=True,
            )
        )
        # Add agent to playground
        center_area, size_area = room.get_partial_area("down")
        spawn_area_agent = CoordinateSampler(
            center_area, area_shape="rectangle", size=size_area
        )
        self.playground.add_agent(self.agent, spawn_area_agent)

        # Init engine
        self.engine = Engine(
            playground=self.playground, time_limit=GOALCONDITIONED_TIMELIMIT
        )

        # Define action and observation space
        # Continuous action space
        actuators = self.agent.controller.controlled_actuators
        lows = []
        highs = []

        for actuator in actuators:

            lows.append(actuator.min)
            highs.append(actuator.max)

        self.action_space = spaces.Box(
            low=np.array(lows).astype(np.float32),
            high=np.array(highs).astype(np.float32),
            dtype=np.float32,
        )

        # Continuous observation space
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=np.array([0, -np.pi, 0, -np.pi, 0, -np.pi]),
                    high=np.array([1, np.pi, 1, np.pi, 1, np.pi]),
                    dtype=np.float64,
                ),
                "achieved_goal": spaces.Box(
                    low=np.zeros(3), high=np.ones(3), dtype=np.uint8
                ),
                "desired_goal": spaces.Box(
                    low=np.zeros(3), high=np.ones(3), dtype=np.uint8
                ),
            }
        )

    def process_obs(self):
        """Process observations to match Gym API"""
        obs = np.zeros(self.observation_space["observation"].shape, dtype=np.float32)
        try:
            obs = np.array(list(self.agent.observations.values()[0]))
            # raw_obs = list(self.agent.observations.values())[0]
            # for raw_o in raw_obs:
            #     # Sorting observations, each coordinate corresponds to the same object always
            #     # TODO: Discuss if this is needed or not!
            #     start = int(raw_o.entity.name) * 2
            #     end = start + 2
            #     obs[start:end] = np.array((raw_o.distance, raw_o.angle))
        except:
            # this avoid the error when the agent is over the objects and gets empty observations!
            pass
        return obs

    def step(self, action):
        actions_dict = {}
        actuators = self.agent.controller.controlled_actuators
        for actuator, act in zip(actuators, action):
            if isinstance(actuator, ContinuousActuator):
                actions_dict[actuator] = act
            else:
                actions_dict[actuator] = round(act)

        self.engine.step({self.agent: actions_dict})
        self.engine.update_observations()
        done = self.playground.done or not self.engine.game_on
        obs = self.process_obs()

        # Multigoal observations
        achieved_goal = np.zeros(3, dtype=np.uint8)
        # deactivate all objects (this should be done in the engine, but its easier to do it
        # like this for now). When an object is pressed it gets activated, so we need to
        # deactivate it afterewards
        for obj in self.playground.elements:
            if isinstance(obj, CustomRewardOnActivation) and obj.activated:
                achieved_goal[int(obj.name)] = 1
                obj.deactivate()
        observations = {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": self._current_goal,
        }

        reward = self.compute_reward(achieved_goal, self._current_goal, None)
        info = {"is_success": reward > 0}
        return observations, reward, done, info

    def reset(self):
        self.engine.reset()

        # Sample new goal
        fountain_with_reward = random.choice([0, 1, 2])
        for obj in self.playground.elements:
            if isinstance(obj, CustomRewardOnActivation):
                if obj.name == str(fountain_with_reward):
                    obj.change_state(True)
                    self._current_goal[int(obj.name)] = 1
                else:
                    obj.change_state(False)
                    self._current_goal[int(obj.name)] = 0

        self.engine.elapsed_time = 0
        self.episodes += 1
        self.engine.update_observations()
        self.time_steps = 0
        obs = self.process_obs()
        observations = {
            "observation": obs,
            "achieved_goal": np.zeros(3, dtype=np.uint8),
            "desired_goal": self._current_goal,
        }
        return observations

    def compute_reward(self, achieved_goal, desired_goal, info) -> np.float32:
        return np.array_equal(achieved_goal, desired_goal)

    def render(self, mode=None):
        return (255 * self.engine.generate_agent_image(self.agent)).astype(np.uint8)

    def close(self):
        self.engine.terminate()
