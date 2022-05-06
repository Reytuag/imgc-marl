import os
from ast import Mult

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = ""

import random

import gym
import numpy as np
from gym import spaces
from imgc_marl.envs.elements.activable import CustomRewardOnActivation
from ray.rllib.env import MultiAgentEnv
from simple_playgrounds.agent.actuators import ContinuousActuator
from simple_playgrounds.agent.agents import BaseAgent
from simple_playgrounds.agent.controllers import External
from simple_playgrounds.common.position_utils import CoordinateSampler
from simple_playgrounds.device.sensors.semantic import PerfectSemantic
from simple_playgrounds.element.elements.basic import Wall
from simple_playgrounds.engine import Engine
from simple_playgrounds.playground.layouts import SingleRoom

SIMPLE_PLAYGROUND = (250, 250)
SIMPLE_TIMELIMIT = 100


class SimpleEnv(MultiAgentEnv):
    """Simple Environment for multiple agents. Single goal.
    Spawns an object that gives a reward when activated and
    terminates the episode"""

    def __init__(self, config):
        n_agents = config["n_agents"]
        super(SimpleEnv, self).__init__()

        self.continuous = config["continuous"]

        self.episodes = 0
        self.time_steps = 0

        # Create playground
        # Minimal environment with 1 room and 1 goal
        self.playground = SingleRoom(
            size=SIMPLE_PLAYGROUND,
        )
        room = self.playground.grid_rooms[0][0]
        center_area, size_area = room.get_partial_area("center")
        spawn_area_fountain = CoordinateSampler(
            center_area, area_shape="rectangle", size=size_area
        )
        fountain = CustomRewardOnActivation(reward=1, terminate=True)
        self.playground.add_element(
            fountain, spawn_area_fountain, allow_overlapping=False
        )

        # Keep track of the walls bc we are ignoring them in the sensors
        self.playground.walls = [
            elem for elem in self.playground.elements if isinstance(elem, Wall)
        ]

        # Add agents
        x = True
        for n in range(n_agents):
            # Init the agent
            agent_name = f"agent_{str(n)}"
            agent = BaseAgent(controller=External(), interactive=True, name=agent_name)
            # Add sensor
            ignore_elements = [
                elem for elem in self.playground.elements if isinstance(elem, Wall)
            ] + [agent.parts, agent.base_platform]
            agent.add_sensor(
                PerfectSemantic(
                    agent.base_platform,
                    invisible_elements=ignore_elements,
                    min_range=0,
                    max_range=max(SIMPLE_PLAYGROUND),
                    name="sensor",
                    normalize=True,
                )
            )
            # Add agent to playground
            if x:
                center_area, size_area = room.get_partial_area("down")
            else:
                center_area, size_area = room.get_partial_area("up")
            spawn_area_agent = CoordinateSampler(
                center_area, area_shape="rectangle", size=size_area
            )
            self.playground.add_agent(agent, spawn_area_agent)
            self._agent_ids.add(agent_name)
            x = not x

        # Init engine
        self.engine = Engine(playground=self.playground, time_limit=SIMPLE_TIMELIMIT)

        # Define action and observation space
        actuators = agent.controller.controlled_actuators
        if not self.continuous:
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
        number_of_elements = (
            len(self.playground.elements) - len(self.playground.walls) + n_agents - 1
        )
        self.observation_space = spaces.Box(
            low=np.repeat(np.array([0, -np.pi]), number_of_elements, axis=0).flatten(),
            high=np.repeat(np.array([1, np.pi]), number_of_elements, axis=0).flatten(),
            dtype=np.float64,
        )

    def process_obs(self):
        """Process observations to match RLlib API (a dict with obs for each agent)"""
        obs = dict()
        for agent in self.playground.agents:
            agent_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            try:
                agent_obs = np.array(list(agent.observations.values()[0]))
            except:
                pass
            obs[agent.name] = agent_obs
        return obs

    def step(self, action_dict):
        actions = {}
        for agent_name, agent_action in action_dict.items():
            agent = [
                agent for agent in self.playground.agents if agent.name == agent_name
            ][0]
            actions[agent] = {}
            actuators = agent.controller.controlled_actuators

            if not self.continuous:
                for actuator, act in zip(actuators, agent_action):
                    if isinstance(actuator, ContinuousActuator):
                        actions[agent][actuator] = [-1, 0, 1][act]
                    else:
                        actions[agent][actuator] = [0, 1][act]
            else:
                for actuator, act in zip(actuators, agent_action):
                    if isinstance(actuator, ContinuousActuator):
                        actions[agent][actuator] = act
                    else:
                        actions[agent][actuator] = round(act)

        self.engine.step(actions)
        self.engine.update_observations()
        observations = self.process_obs()
        rewards = {}
        dones = {}
        for agent in self.playground.agents:
            rewards[agent.name] = agent.reward
            dones[agent.name] = self.playground.done or not self.engine.game_on
            dones["__all__"] = dones[agent.name]
        return observations, rewards, dones, {}

    def reset(self):
        self.engine.reset()
        self.engine.elapsed_time = 0
        self.episodes += 1
        self.engine.update_observations()
        self.time_steps = 0
        observations = self.process_obs()
        return observations

    def render(self, mode=None):
        return (255 * self.engine.generate_playground_image()).astype(np.uint8)

    def close(self):
        self.engine.terminate()
