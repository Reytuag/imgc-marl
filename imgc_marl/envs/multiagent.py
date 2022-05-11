import os
from typing import Dict

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = ""

import numpy as np
from gym import spaces
from imgc_marl.envs.utils import MetaSampler
from ray.rllib.env import MultiAgentEnv
from simple_playgrounds.agent.actuators import ContinuousActuator
from simple_playgrounds.agent.agents import BaseAgent
from simple_playgrounds.agent.controllers import External
from simple_playgrounds.common.position_utils import CoordinateSampler
from simple_playgrounds.common.texture import UniqueCenteredStripeTexture
from simple_playgrounds.device.sensors.semantic import PerfectSemantic
from simple_playgrounds.element.elements.activable import RewardOnActivation
from simple_playgrounds.element.elements.basic import Wall
from simple_playgrounds.element.elements.zone import RewardZone
from simple_playgrounds.engine import Engine
from simple_playgrounds.playground.layouts import SingleRoom

SIMPLE_PLAYGROUND = (300, 300)
SIMPLE_TIMELIMIT = 100
GOAL_LINES_TIMELIMIT = 150

POSSIBLE_GOAL_LINES = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

N_GOAL_LINES = len(POSSIBLE_GOAL_LINES)


class OneBoxEnv(MultiAgentEnv):
    """Simple Environment for multiple agents. Single goal.
    Spawns an object that gives a reward when activated and
    terminates the episode"""

    def __init__(self, config):
        n_agents = config["n_agents"]
        super(OneBoxEnv, self).__init__()

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
        fountain = RewardOnActivation(reward=1)
        self.playground.add_element(
            fountain, spawn_area_fountain, allow_overlapping=True
        )

        # Keep track of the walls bc we are ignoring them in the sensors
        self.playground.walls = [
            elem for elem in self.playground.elements if isinstance(elem, Wall)
        ]

        # Add agents
        colors = [(51, 204, 51), (37, 150, 190)]
        for n in range(n_agents):
            # Init the agent
            agent_name = f"agent_{str(n)}"
            agent = BaseAgent(
                controller=External(),
                interactive=True,
                name=agent_name,
                texture=UniqueCenteredStripeTexture(
                    color=colors[n],
                    color_stripe=[0, 20, 240],
                    size_stripe=4,
                ),
            )
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
            if n == 0:
                center_area, size_area = room.get_partial_area("down")
            else:
                center_area, size_area = room.get_partial_area("up")
            spawn_area_agent = CoordinateSampler(
                center_area, area_shape="rectangle", size=size_area
            )
            self.playground.add_agent(agent, spawn_area_agent)
            self._agent_ids.add(agent_name)

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
            low=np.array([[0, -np.pi] for i in range(number_of_elements)]).flatten(),
            high=np.array([[1, np.pi] for i in range(number_of_elements)]).flatten(),
            dtype=np.float64,
        )

    def process_obs(self):
        """Process observations to match RLlib API (a dict with obs for each agent)"""
        obs = dict()
        for agent in self.playground.agents:
            agent_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            raw_obs = list(agent.observations.values())[0]
            for i, raw_o in enumerate(raw_obs):
                agent_obs[2 * i : 2 * i + 2] = np.array((raw_o.distance, raw_o.angle))
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
        done = False
        for agent in self.playground.agents:
            rewards[agent.name] = agent.reward
            # The episode is done for all agents either when a reward is obtained or after the time limit
            done = (
                done
                or bool(agent.reward)
                or self.playground.done
                or not self.engine.game_on
            )

        dones = {agent.name: done for agent in self.playground.agents}
        dones["__all__"] = done
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


class GoalLinesEnv(MultiAgentEnv):
    """GoalLinesEnv for multiple agents. Goal conditioned.
    There are different goal areas to be targeted as objectives"""

    def __init__(self, config):
        super(MultiAgentEnv, self).__init__()

        self.continuous = config["continuous"]

        self.episodes = 0
        self.time_steps = 0

        # Create playground
        # One room with 3 goal zones
        self.playground = SingleRoom(size=(400, 400))
        zone_0 = RewardZone(
            reward=1,
            limit=1e6,
            physical_shape="rectangle",
            texture=[255, 0, 0],
            size=(25, 150),
            name="001",
        )
        zone_1 = RewardZone(
            reward=2,
            limit=1e6,
            physical_shape="rectangle",
            texture=[0, 0, 255],
            size=(25, 150),
            name="010",
        )
        zone_2 = RewardZone(
            reward=3,
            limit=1e6,
            physical_shape="rectangle",
            texture=[255, 255, 255],
            size=(25, 150),
            name="100",
        )
        self.playground.add_element(zone_0, ((125, 200), 0))
        self.playground.add_element(zone_1, ((200, 200), 0))
        self.playground.add_element(zone_2, ((275, 200), 0))

        self.playground.walls = [
            elem for elem in self.playground.elements if isinstance(elem, Wall)
        ]

        # Add agents
        self._agent_ids = set()

        sampler1 = CoordinateSampler((200, 50), area_shape="rectangle", size=(400, 100))
        sampler2 = CoordinateSampler(
            (200, 350), area_shape="rectangle", size=(400, 100)
        )
        sampler3 = CoordinateSampler((50, 200), area_shape="rectangle", size=(25, 200))
        sampler4 = CoordinateSampler((350, 200), area_shape="rectangle", size=(25, 200))
        agent_sampler = MetaSampler([sampler1, sampler2, sampler3, sampler4])

        # Agent 0
        agent = BaseAgent(
            controller=External(),
            interactive=False,
            name="agent_0",
            texture=UniqueCenteredStripeTexture(
                color=(0, 200, 0), color_stripe=(0, 0, 200), size_stripe=4
            ),
        )
        ignore_elements = [agent.parts, agent.base_platform]
        agent.add_sensor(
            PerfectSemantic(
                agent.base_platform,
                invisible_elements=ignore_elements,
                min_range=0,
                max_range=400,
                name="sensor",
                normalize=True,
            )
        )
        self.playground.add_agent(agent, agent_sampler)
        self._agent_ids.add("agent_0")
        # Agent 1
        agent = BaseAgent(
            controller=External(),
            interactive=False,
            name="agent_1",
            texture=UniqueCenteredStripeTexture(
                color=(0, 0, 200), color_stripe=(0, 200, 0), size_stripe=4
            ),
        )
        ignore_elements = [agent.parts, agent.base_platform]
        agent.add_sensor(
            PerfectSemantic(
                agent.base_platform,
                invisible_elements=ignore_elements,
                min_range=0,
                max_range=400,
                name="sensor",
                normalize=True,
            )
        )
        self.playground.add_agent(agent, agent_sampler)
        self._agent_ids.add("agent_1")

        # Init engine
        self.engine = Engine(
            playground=self.playground, time_limit=GOAL_LINES_TIMELIMIT
        )

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

        # Continuous observation space + goal representation as ohe
        number_of_elements = len(self.playground.walls) + 1
        self.observation_space = spaces.Box(
            low=np.hstack(
                (
                    np.array(
                        [[0, -2 * np.pi] for i in range(number_of_elements)]
                    ).flatten(),
                    np.zeros(3),
                )
            ),
            high=np.hstack(
                (
                    np.array(
                        [[1, 2 * np.pi] for i in range(number_of_elements)]
                    ).flatten(),
                    np.ones(3),
                )
            ),
            dtype=np.float64,
        )

        # List of active agents, agents can exit early if completed their goal
        self._active_agents = self.playground.agents.copy()

    def process_obs(self):
        """Process observations to match RLlib API (a dict with obs for each agent) and append goal"""
        obs = dict()
        for agent in self._active_agents:
            agent_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            # append agents goal at the end of the obs
            agent_obs[-3:] = agent.goal
            raw_obs = list(agent.observations.values())[0]
            for i, raw_o in enumerate(raw_obs):
                agent_obs[2 * i : 2 * i + 2] = np.array([raw_o.distance, raw_o.angle])
            obs[agent.name] = agent_obs
        return obs

    def compute_rewards(self):
        """
        If goal is individual, the agent must solve it by itself.
        If goal is collective, it doesn't matter which of the two
        goal zones an agent is touching (since both of them are
        required)
        """
        individual_achieved_goals = np.zeros((3, 2), np.uint8)
        rewards = {}
        dones = {}
        # Computing individual achieved goals
        for i, agent in enumerate(self.playground.agents):
            # Hack for identifying which goal is being activated by this agent
            if agent.reward:
                individual_achieved_goals[agent.reward - 1, i] = 1
        # Computing collective goal
        collective_achieved_goal = np.bitwise_or.reduce(
            individual_achieved_goals, axis=1
        )
        # Checking if achieved goal is desired goal (only for active agents)
        for i, agent in enumerate(self._active_agents):
            if (
                np.sum(agent.goal) > 1
                and np.all(agent.goal == collective_achieved_goal)
            ) or (np.all(agent.goal == individual_achieved_goals[:, i])):
                reward = 1
                # If agent achieved its goal, its removed from the active list
                self._active_agents.pop(i)
            else:
                reward = 0
            rewards[agent.name] = reward
            dones[agent.name] = (
                reward or self.playground.done or not self.engine.game_on
            )
        dones["__all__"] = all(dones.values())
        return rewards, dones

    def step(self, action_dict):
        actions = {}
        for agent_name, agent_action in action_dict.items():
            agent = [
                agent for agent in self._active_agents if agent.name == agent_name
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
        rewards, dones = self.compute_rewards()
        observations = self.process_obs()
        return observations, rewards, dones, {}

    def reset(self, external_goals: Dict[str, int] = None):
        self.engine.reset()
        # All agents become active again
        self._active_agents = self.playground.agents.copy()
        # Each agent samples its own goal if not provided externally
        if external_goals is None:
            for agent in self.playground.agents:
                agent.goal = POSSIBLE_GOAL_LINES[np.random.randint(0, N_GOAL_LINES)]
        else:
            # We can externally provide the goals as
            ## {"agent_name": goal_index}
            for agent in self.playground.agents:
                agent.goal = POSSIBLE_GOAL_LINES[external_goals[agent.name]]

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
