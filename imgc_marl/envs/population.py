import os
import random
from typing import Dict

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = ""

from itertools import combinations

import cv2
import numpy as np
import torch
from gym import spaces
from imgc_marl.envs.elements.zone import MultiAgentRewardZone
from ray.rllib.env import MultiAgentEnv
from simple_playgrounds.agent.actuators import ContinuousActuator
from simple_playgrounds.agent.agents import BaseAgent
from simple_playgrounds.agent.controllers import External
from simple_playgrounds.common.position_utils import CoordinateSampler
from simple_playgrounds.common.texture import UniqueCenteredStripeTexture
from simple_playgrounds.device.sensors.semantic import PerfectSemantic
from simple_playgrounds.engine import Engine
from simple_playgrounds.playground.layouts import SingleRoom

font = cv2.FONT_HERSHEY_SIMPLEX

SIMPLE_PLAYGROUND = (300, 300)
SIMPLE_TIMELIMIT = 100
GOAL_LINES_TIMELIMIT = 250
NEW_ENV_TIMELIMIT = 250
LARGE_GOAL_LINES_TIMELIMIT = 500

POPULATION_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 128, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (0, 100, 100),
]
STRIP_COLOR = (255, 255, 255)


class PopGoalLinesEnv(MultiAgentEnv):
    """PopGoalLinesEnv for multiple agents. Goal conditioned for a population of agents.
    There are different goal areas to be targeted as objectives"""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config):
        super().__init__()

        # If agents should sample goals centralized or decentralized
        self.centralized = config.get("centralized", False)
        # If goal is fixed or might be updated
        self.fixed_goal = False
        self.fixed_leader_goal = None
        # If independent agents might get aligned sometimes
        self.alignment_percentage = config.get("alignment_percentage", 0.0)
        # If consider new reward scheme (individual == collective)
        self.new_reward = config.get("new_reward", False)
        # Number of agents in the population
        self.n_agents = config.get("population_size", 4)
        # Epsilon greedy exploration for communication policy
        self.eps_communication = config.get("eps_communication", 0.1)
        # If consider all goals or only collective ones
        self.all_goals = config.get("all_goals", True)
        # Only compatible goals?
        self.only_compatible = config.get("only_compatible", False)
        
        self.reward_multiplier=config.get("reward_multiplier",1.)

        # Goal space
        if self.all_goals:
            self.goal_space = [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ]
        else:
            self.goal_space = [
                [0,1,1],
                [1,0,1],
                [1,1,0],
            ]
        self.goal_space_dim = len(self.goal_space)
        self.goal_repr_dim = 3

        # ZERO-SHOT CONFIG ONLY: Only allowing some goals during training
        # ZERO-SHOT ONLY WORKS WITH CENTRALIZED OR INDEPENDENT
        self.allowed_training_goals = config.get("allowed_goals", self.goal_space)
        self.n_allowed_training_goals = len(self.allowed_training_goals)

        self.episodes = 0
        self.time_steps = 0

        # Create playground
        # One room with 3 goal zones
        self.playground = SingleRoom(size=(400, 400))
        zone_0 = MultiAgentRewardZone(
            reward=1,
            physical_shape="rectangle",
            texture=[255, 0, 0],
            size=(30, 100),
            name="001",
        )
        zone_1 = MultiAgentRewardZone(
            reward=100,
            physical_shape="rectangle",
            texture=[0, 0, 255],
            size=(100, 30),
            name="010",
        )
        zone_2 = MultiAgentRewardZone(
            reward=10_000,
            physical_shape="rectangle",
            texture=[255, 255, 255],
            size=(30, 100),
            name="100",
        )
        self.playground.add_element(zone_0, ((15, 350), 0))
        self.playground.add_element(zone_1, ((200, 15), 0))
        self.playground.add_element(zone_2, ((385, 350), 0))

        # Create agents
        self._agent_ids = set()

        agent_sampler = CoordinateSampler(
            (200, 200), area_shape="rectangle", size=(300, 300)
        )
        self.agent_sampler = agent_sampler

        # Creating each agent
        for i in range(self.n_agents):
            agent = BaseAgent(
                controller=External(),
                interactive=False,
                name=f"agent_{i}",
                texture=UniqueCenteredStripeTexture(
                    color=POPULATION_COLORS[i], color_stripe=STRIP_COLOR, size_stripe=4
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
            self._agent_ids.add(f"agent_{i}")

        # Init engine
        self.engine = Engine(
            playground=self.playground, time_limit=GOAL_LINES_TIMELIMIT
        )

        # Define action and observation space
        actuators = agent.controller.controlled_actuators
        # Discrete action space
        act_spaces = []
        for actuator in actuators:
            if isinstance(actuator, ContinuousActuator):
                act_spaces.append(3)
            else:
                act_spaces.append(2)
        self.action_space = spaces.MultiDiscrete(act_spaces)

        # Continuous observation space + goal representation as ohe
        number_of_elements = len(self.playground.elements) + 1
        self.observation_space = spaces.Box(
            low=np.hstack(
                (
                    np.array(
                        [[0, -2 * np.pi] for i in range(number_of_elements)]
                    ).flatten(),
                    np.zeros(self.goal_repr_dim),
                )
            ),
            high=np.hstack(
                (
                    np.array(
                        [[1, 2 * np.pi] for i in range(number_of_elements)]
                    ).flatten(),
                    np.ones(self.goal_repr_dim),
                )
            ),
            dtype=np.float64,
        )

        # Mapping to keep consistent coordinates of observations for the same objects
        # Elements will have the first coordinates and then the other agent
        for j, agent in enumerate(self.playground.agents):
            agent.COORDINATE_MAP = {
                element: 2 * i for i, element in enumerate(self.playground.elements)
            }
            for n in range(1, self.n_agents):
                agent.COORDINATE_MAP[self.playground.agents[j - n].parts[0]] = (
                    len(self.playground.elements) * 2
                )
        # List of active agents, agents can exit early if completed their goal
        self._active_agents = self.playground.agents.copy()
        # Track the whole population
        self.population = self.playground.agents.copy()

    def process_obs(self):
        """Process observations to match RLlib API (a dict with obs for each agent) and append goal"""
        obs = dict()
        for agent in self._active_agents:
            agent_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            # append own agents goal at the end of the obs
            agent_obs[-self.goal_repr_dim :] = agent.goal
            raw_obs = list(agent.observations.values())[0]
            for raw_o in raw_obs:
                coordinate = agent.COORDINATE_MAP[raw_o.entity]
                agent_obs[coordinate : coordinate + 2] = np.array(
                    [raw_o.distance, raw_o.angle]
                )
            obs[agent.name] = agent_obs
        return obs

    def compute_rewards(self):
        """
        If goal is individual, the agent must solve it by itself.
        If goal is collective, it doesn't matter which of the two
        goal zones an agent is touching (since both of them are
        required)
        """
        individual_achieved_goals = {
            agent.name: np.zeros(self.goal_repr_dim, dtype=int)
            for agent in self.playground.agents
        }
        assert len(individual_achieved_goals) == 2
        agent_a = self.playground.agents[0].name
        agent_b = self.playground.agents[1].name
        rewards = {}
        dones = {}
        info = {}
        # Computing individual achieved goals
        for agent in self.playground.agents:
            # Hack for identifying which goal is being activated by this agent
            if agent.reward:
                if agent.reward < 100:
                    agent.reward = 1
                elif agent.reward < 10_000:
                    agent.reward = 2
                else:
                    agent.reward = 3
                individual_achieved_goals[agent.name][agent.reward - 1] = 1
        # Computing collective goal
        collective_achieved_goal = np.bitwise_or.reduce(
            np.vstack(
                [
                    individual_achieved_goals[agent_a],
                    individual_achieved_goals[agent_b],
                ]
            ),
            axis=0,
        )
        # Checking if achieved goal is desired goal (only for active agents)
        for agent in self._active_agents:
            if not self.new_reward:
                # original reward scheme
                if (
                    np.sum(agent.goal) > 1
                    and np.all(agent.goal == collective_achieved_goal)
                ) :
                    reward = 1
                elif (np.all(agent.goal == individual_achieved_goals[agent.name])):
                    reward = 1/self.reward_multiplier
                else:
                    reward = 0
            else:
                # new reward scheme (individual goals are consistent with collective ones)
                if (
                    np.sum(agent.goal) > 1
                    and np.all(agent.goal == collective_achieved_goal)
                ):
                    reward = 1
                elif ((np.all(agent.goal == individual_achieved_goals[agent_a]))
                    or (np.all(agent.goal == individual_achieved_goals[agent_b]))):
                    reward = 1/self.reward_mutiplier
                else:
                    reward = 0
            rewards[agent.name] = reward
            done = bool(reward) or self.playground.done or not self.engine.game_on
            dones[agent.name] = done
            # logging which goal line the agent achieved (-1 means no goal line)
            info[agent.name] = {"goal_line": agent.reward - 1}
            if done and agent.message is not None:
                info[agent.name]["message"] = agent.message

        # Agents that are done are deleted from the list of active agents
        [
            self._active_agents.remove(agent)
            for agent in self._active_agents
            if dones[agent.name]
        ]
        dones["__all__"] = all(dones.values())
        return rewards, dones, info

    def step(self, action_dict):
        actions = {}
        if action_dict:
            for agent in self._active_agents:
                agent_action = action_dict.get(agent.name)
                actions[agent] = {}
                actuators = agent.controller.controlled_actuators
                for actuator, act in zip(actuators, agent_action):
                    if isinstance(actuator, ContinuousActuator):
                        actions[agent][actuator] = [-1, 0, 1][act]
                    else:
                        actions[agent][actuator] = [0, 1][act]
        self.engine.step(actions)
        self.engine.update_observations()
        observations = self.process_obs()
        rewards, dones, info = self.compute_rewards()
        return observations, rewards, dones, info

    def reset(self):
        self.populate_env_before_episode()
        self.engine.reset()
        # Each agent samples its own goal if not fixed
        if not self.fixed_goal:
            if self.centralized or np.random.random() < self.alignment_percentage:
                # Centralized uniform
                goal = self.allowed_training_goals[np.random.randint(0, self.n_allowed_training_goals)]
                for agent in self.playground.agents:
                    agent.goal = goal
            else:
                # ONLY COMPATIBLE GOALS IS ONLY SUPPORTED WITH 2 AGENTS
                # only safe to use with 2 agents, with more the behavior is not clear
                if self.only_compatible:
                    incompatible_goals = True
                    while incompatible_goals:
                        for agent in self.playground.agents:
                            agent.goal = self.goal_space[
                                np.random.randint(0, self.goal_space_dim)
                            ]
                        if (
                            np.bitwise_or(
                                self.playground.agents[0].goal,
                                self.playground.agents[1].goal,
                            ).sum()
                            <= 2
                        ):
                            incompatible_goals = False
                else:
                    # independent uniform sampling
                    for agent in self.playground.agents:
                        agent.goal = self.allowed_training_goals[
                            np.random.randint(0, self.n_allowed_training_goals)
                        ]
        for agent in self.playground.agents:
            agent.message = None

        self.engine.elapsed_time = 0
        self.episodes += 1
        self.engine.update_observations()
        self.time_steps = 0
        observations = self.process_obs()
        return observations

    def populate_env_before_episode(self):
        """
        Sample two agents from the population and adds them to the env for this episode
        """
        # Restore all agents
        for agent in self.population:
            if agent not in self.playground.agents:
                self.playground.add_agent(agent, self.agent_sampler)
        # Delete all agents that were not selected for playing this round
        for agent in random.sample(self.population, len(self.population) - 2):
            self.playground.remove_agent(agent)
        self._active_agents = self.playground.agents.copy()

    def set_external_goal(
        self, external_goals: Dict[str, int] = None, fix_goal: bool = True
    ):
        # Not sample goal on each reset
        self.fixed_goal = fix_goal
        for agent in self.population:
            if agent.name in external_goals.keys():
                agent.goal = self.goal_space[external_goals[agent.name]]

    def set_external_goal_communication(self, leader_goal_index):
        # Turning on this flag to apply the communication strategy in the callback!
        self.fixed_leader_goal = leader_goal_index

    def render(self, mode=None):
        frame = (255 * self.engine.generate_playground_image()).astype(np.uint8)
        cv2.putText(
            frame,
            str(self.playground.agents[0].goal),
            (10, 35),
            font,
            1,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            str(self.playground.agents[1].goal),
            (250, 35),
            font,
            1,
            (255, 255, 255),
            1,
        )
        return frame

    def close(self):
        self.engine.terminate()

    def set_goal_and_message(
        self,
        external_goals: Dict[str, int],
        message: Dict[str, torch.Tensor],
    ):
        # Not sample goal on each reset
        # self.fixed_goal = True
        if not self.fixed_goal:
            for agent in self.playground.agents:
                agent.goal = external_goals[agent.name]
                for name, msg in message.items():
                    if agent.name == name:
                        agent.message = msg


class PopLargeGoalLinesEnv(PopGoalLinesEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config):
        super(PopGoalLinesEnv, self).__init__()

        # If agents should sample goals centralized or decentralized
        self.centralized = config.get("centralized", False)
        # If goal is fixed or might be updated
        self.fixed_goal = False
        self.fixed_leader_goal = None
        # If independent agents might get aligned sometimes
        self.alignment_percentage = config.get("alignment_percentage", 0.0)
        # If consider new reward scheme (individual == collective)
        self.new_reward = config.get("new_reward", False)
        # Number of agents in the population
        self.n_agents = config.get("population_size", 4)
        # Epsilon greedy exploration for communication policy
        self.eps_communication = config.get("eps_communication", 0.1)
        # If consider all goals or only collective ones
        self.all_goals = config.get("all_goals", True)
        # Only compatible goals?
        self.only_compatible = config.get("only_compatible", False)


        # Goal space
        landmarks = 6
        if self.all_goals:
            self.goal_space = np.eye(landmarks, dtype=np.uint8).tolist()
            self.goal_space += (
                np.array(list(combinations(self.goal_space, 2))).sum(1).tolist()
            )
        else:
            individual_goals = np.eye(landmarks, dtype=np.uint8).tolist()
            self.goal_space = (
                np.array(list(combinations(individual_goals, 2))).sum(1).tolist()
            )
        self.goal_space_dim = len(self.goal_space)
        self.goal_repr_dim = landmarks

        # ZERO-SHOT CONFIG ONLY: Only allowing some goals during training
        # ZERO-SHOT ONLY WORKS WITH CENTRALIZED OR INDEPENDENT
        self.allowed_training_goals = config.get("allowed_goals", self.goal_space)
        self.n_allowed_training_goals = len(self.allowed_training_goals)

        self.episodes = 0
        self.time_steps = 0

        # Create playground
        self.playground = SingleRoom(size=(400, 400))
        zone_0 = MultiAgentRewardZone(
            reward=1e-4,
            physical_shape="rectangle",
            texture=[255, 0, 0],
            size=(100, 20),
        )
        zone_1 = MultiAgentRewardZone(
            reward=0.01,
            physical_shape="rectangle",
            texture=[0, 255, 0],
            size=(20, 100),
        )
        zone_2 = MultiAgentRewardZone(
            reward=1,
            physical_shape="rectangle",
            texture=[255, 255, 255],
            size=(100, 20),
        )
        zone_3 = MultiAgentRewardZone(
            reward=100,
            physical_shape="rectangle",
            texture=[0, 255, 255],
            size=(100, 20),
        )
        zone_4 = MultiAgentRewardZone(
            reward=10_000,
            physical_shape="rectangle",
            texture=[0, 0, 255],
            size=(20, 100),
        )
        zone_5 = MultiAgentRewardZone(
            reward=1_000_000,
            physical_shape="rectangle",
            texture=[150, 0, 200],
            size=(100, 20),
        )
        self.playground.add_element(zone_0, ((50, 390), 0))
        self.playground.add_element(zone_1, ((10, 200), 0))
        self.playground.add_element(zone_2, ((50, 10), 0))
        self.playground.add_element(zone_3, ((350, 10), 0))
        self.playground.add_element(zone_4, ((390, 200), 0))
        self.playground.add_element(zone_5, ((350, 390), 0))

        # Create agents
        self._agent_ids = set()

        agent_sampler = CoordinateSampler(
            (200, 200), area_shape="rectangle", size=(300, 300)
        )
        self.agent_sampler = agent_sampler

        # Creating each agent
        for i in range(self.n_agents):
            agent = BaseAgent(
                controller=External(),
                interactive=False,
                name=f"agent_{i}",
                texture=UniqueCenteredStripeTexture(
                    color=POPULATION_COLORS[i], color_stripe=STRIP_COLOR, size_stripe=4
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
            self._agent_ids.add(f"agent_{i}")

        # Init engine
        self.engine = Engine(
            playground=self.playground, time_limit=LARGE_GOAL_LINES_TIMELIMIT
        )

        # Define action and observation space
        actuators = agent.controller.controlled_actuators
        # Discrete action space
        act_spaces = []
        for actuator in actuators:
            if isinstance(actuator, ContinuousActuator):
                act_spaces.append(3)
            else:
                act_spaces.append(2)
        self.action_space = spaces.MultiDiscrete(act_spaces)

        # Continuous observation space + goal representation as ohe
        number_of_elements = len(self.playground.elements) + 1
        self.observation_space = spaces.Box(
            low=np.hstack(
                (
                    np.array(
                        [[0, -2 * np.pi] for i in range(number_of_elements)]
                    ).flatten(),
                    np.zeros(self.goal_repr_dim),
                )
            ),
            high=np.hstack(
                (
                    np.array(
                        [[1, 2 * np.pi] for i in range(number_of_elements)]
                    ).flatten(),
                    np.ones(self.goal_repr_dim),
                )
            ),
            dtype=np.float64,
        )

        # Mapping to keep consistent coordinates of observations for the same objects
        # Elements will have the first coordinates and then the other agent
        for j, agent in enumerate(self.playground.agents):
            agent.COORDINATE_MAP = {
                element: 2 * i for i, element in enumerate(self.playground.elements)
            }
            for n in range(1, self.n_agents):
                agent.COORDINATE_MAP[self.playground.agents[j - n].parts[0]] = (
                    len(self.playground.elements) * 2
                )
        # List of active agents, agents can exit early if completed their goal
        self._active_agents = self.playground.agents.copy()
        # Track the whole population
        self.population = self.playground.agents.copy()

    def compute_rewards(self):
        """
        If goal is individual, the agent must solve it by itself.
        If goal is collective, it doesn't matter which of the two
        goal zones an agent is touching (since both of them are
        required)
        """
        individual_achieved_goals = {
            agent.name: np.zeros(self.goal_repr_dim, dtype=int)
            for agent in self.playground.agents
        }
        assert len(individual_achieved_goals) == 2
        agent_a = self.playground.agents[0].name
        agent_b = self.playground.agents[1].name
        rewards = {}
        dones = {}
        info = {}
        # Computing individual achieved goals
        for agent in self.playground.agents:
            # Hack for identifying which goal is being activated by this agent
            if agent.reward:
                if agent.reward < 0.01:
                    agent.reward = 1
                elif agent.reward < 1:
                    agent.reward = 2
                elif agent.reward < 100:
                    agent.reward = 3
                elif agent.reward < 10_000:
                    agent.reward = 4
                elif agent.reward < 1_000_000:
                    agent.reward = 5
                else:
                    agent.reward = 6
                individual_achieved_goals[agent.name][agent.reward - 1] = 1
        # Computing collective goal
        collective_achieved_goal = np.bitwise_or.reduce(
            np.vstack(
                [
                    individual_achieved_goals[agent_a],
                    individual_achieved_goals[agent_b],
                ]
            ),
            axis=0,
        )
        # Checking if achieved goal is desired goal (only for active agents)
        for agent in self._active_agents:
            if not self.new_reward:
                # original reward scheme
                if (
                    np.sum(agent.goal) > 1
                    and np.all(agent.goal == collective_achieved_goal)
                ) or (np.all(agent.goal == individual_achieved_goals[agent.name])):
                    reward = 1
                else:
                    reward = 0
            else:
                # new reward scheme (individual goals are consistent with collective ones)
                if (
                    np.sum(agent.goal) > 1
                    and np.all(agent.goal == collective_achieved_goal)
                ) or (
                    (np.all(agent.goal == individual_achieved_goals[agent_a]))
                    or (np.all(agent.goal == individual_achieved_goals[agent_b]))
                ):
                    reward = 1
                else:
                    reward = 0
            rewards[agent.name] = reward
            done = bool(reward) or self.playground.done or not self.engine.game_on
            dones[agent.name] = done
            # logging which goal line the agent achieved (-1 means no goal line)
            info[agent.name] = {"goal_line": agent.reward - 1}

            if done and agent.message is not None:
                info[agent.name]["message"] = agent.message

        # Agents that are done are deleted from the list of active agents
        [
            self._active_agents.remove(agent)
            for agent in self._active_agents
            if dones[agent.name]
        ]
        dones["__all__"] = all(dones.values())
        return rewards, dones, info
