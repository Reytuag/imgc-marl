import os
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
from simple_playgrounds.element.elements.activable import (
    OpenCloseSwitch,
    RewardOnActivation,
)
from simple_playgrounds.element.elements.basic import Wall
from simple_playgrounds.engine import Engine
from simple_playgrounds.playground.layouts import LineRooms, SingleRoom

font = cv2.FONT_HERSHEY_SIMPLEX

SIMPLE_PLAYGROUND = (300, 300)
SIMPLE_TIMELIMIT = 100
GOAL_LINES_TIMELIMIT = 250
NEW_ENV_TIMELIMIT = 250
LARGE_GOAL_LINES_TIMELIMIT = 500


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
                if isinstance(actuator, ContinuousActuator):
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

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config):
        super().__init__()

        # If action space is continuous or discrete
        self.continuous = config["continuous"]
        # If agents should sample goals centralized or decentralized
        self.centralized = config.get("centralized", False)
        # If use learning progress or not. learning_progress is the epsilon value in exploration (0 acts as a flag for not using LP)
        self.learning_progress = config.get("learning_progress", 0)
        # If use joint learning progress or not.
        self.joint_learning_progress = config.get("joint_learning_progress", 0)
        # Number of episodes for updating LP
        self.update_lp = config.get("update_lp", 500)
        # If goal is fixed or might be updated
        self.fixed_goal = False
        # If policies are conditioned on both goals
        self.double_condition = config.get("double_condition", False)
        # If independent agents might get aligned sometimes
        self.alignment_percentage = config.get("alignment_percentage", 0.0)
        # If consider new reward scheme (individual == collective)
        self.new_reward = config.get("new_reward", False)
        # If agents will be blind to each other
        self.blind_agents = config.get("blind_agents", False)

        # Goal space
        self.goal_space = [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
        self.goal_space_dim = len(self.goal_space)
        self.goal_repr_dim = 3
        # Only allowing some goals during training to test generalization
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

        # Agent 0
        agent_0 = BaseAgent(
            controller=External(),
            interactive=False,
            name="agent_0",
            texture=UniqueCenteredStripeTexture(
                color=(0, 200, 0), color_stripe=(0, 0, 200), size_stripe=4
            ),
        )
        agent_0.learning_progress = {
            "".join(str(t) for t in goal): 0.0 for goal in self.goal_space
        }
        agent_0.competence = {
            "".join(str(t) for t in goal): 0.0 for goal in self.goal_space
        }
        agent_0.reward_list = {
            "".join(str(t) for t in goal): [] for goal in self.goal_space
        }
        agent_0.joint_learning_progress = np.zeros(
            (self.goal_space_dim, self.goal_space_dim)
        )
        agent_0.joint_competence = np.zeros((self.goal_space_dim, self.goal_space_dim))
        agent_0.joint_reward_list = {
            i: {i: [] for i in range(self.goal_space_dim)}
            for i in range(self.goal_space_dim)
        }

        # Agent 1
        agent_1 = BaseAgent(
            controller=External(),
            interactive=False,
            name="agent_1",
            texture=UniqueCenteredStripeTexture(
                color=(0, 0, 200), color_stripe=(0, 200, 0), size_stripe=4
            ),
        )
        agent_1.learning_progress = {
            "".join(str(t) for t in goal): 0.0 for goal in self.goal_space
        }
        agent_1.competence = {
            "".join(str(t) for t in goal): 0.0 for goal in self.goal_space
        }
        agent_1.reward_list = {
            "".join(str(t) for t in goal): [] for goal in self.goal_space
        }
        agent_1.joint_learning_progress = np.zeros(
            (self.goal_space_dim, self.goal_space_dim)
        )
        agent_1.joint_competence = np.zeros((self.goal_space_dim, self.goal_space_dim))
        agent_1.joint_reward_list = {
            i: {i: [] for i in range(self.goal_space_dim)}
            for i in range(self.goal_space_dim)
        }

        # Add agents to the playground
        ignore_elements = [agent_0.parts, agent_0.base_platform]
        if self.blind_agents:
            ignore_elements += [agent_1.parts, agent_1.base_platform]
        agent_0.add_sensor(
            PerfectSemantic(
                agent_0.base_platform,
                invisible_elements=ignore_elements,
                min_range=0,
                max_range=400,
                name="sensor",
                normalize=True,
            )
        )
        self.playground.add_agent(agent_0, agent_sampler)

        self._agent_ids.add("agent_0")
        if not self.blind_agents:
            ignore_elements = [agent_1.parts, agent_1.base_platform]
        agent_1.add_sensor(
            PerfectSemantic(
                agent_1.base_platform,
                invisible_elements=ignore_elements,
                min_range=0,
                max_range=400,
                name="sensor",
                normalize=True,
            )
        )
        self.playground.add_agent(agent_1, agent_sampler)
        self._agent_ids.add("agent_1")

        # Init engine
        self.engine = Engine(
            playground=self.playground, time_limit=GOAL_LINES_TIMELIMIT
        )

        # Define action and observation space
        actuators = agent_0.controller.controlled_actuators
        if not self.continuous:
            # Discrete action space
            act_spaces = []
            for actuator in actuators:
                if isinstance(actuator, ContinuousActuator):
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
        # if double condition (we condition on both goals) obs space + other goal ohe + own goal ohe
        if self.blind_agents:
            number_of_elements = len(self.playground.elements)
        else:
            number_of_elements = len(self.playground.elements) + 1
        if not self.double_condition:
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
        else:
            self.observation_space = spaces.Box(
                low=np.hstack(
                    (
                        np.array(
                            [[0, -2 * np.pi] for i in range(number_of_elements)]
                        ).flatten(),
                        np.zeros(self.goal_repr_dim),
                        np.zeros(self.goal_repr_dim),
                    )
                ),
                high=np.hstack(
                    (
                        np.array(
                            [[1, 2 * np.pi] for i in range(number_of_elements)]
                        ).flatten(),
                        np.ones(self.goal_repr_dim),
                        np.ones(self.goal_repr_dim),
                    )
                ),
                dtype=np.float64,
            )

        # Mapping to keep consistent coordinates of observations for the same objects
        # Elements will have the first coordinates and then the agent
        for j, agent in enumerate(self.playground.agents):
            agent.COORDINATE_MAP = {
                element: 2 * i for i, element in enumerate(self.playground.elements)
            }
            if not self.blind_agents:
                agent.COORDINATE_MAP[self.playground.agents[j - 1].parts[0]] = (
                    len(self.playground.elements) * 2
                )
        # List of active agents, agents can exit early if completed their goal
        self._active_agents = self.playground.agents.copy()

    def process_obs(self):
        """Process observations to match RLlib API (a dict with obs for each agent) and append goal"""
        obs = dict()
        for agent in self._active_agents:
            agent_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            # append own agents goal at the end of the obs
            agent_obs[-self.goal_repr_dim :] = agent.goal
            if self.double_condition:
                # Append other's agent goal before own goal
                other_agent = [a for a in self.playground.agents if a != agent][0]
                agent_obs[
                    -2 * self.goal_repr_dim : -self.goal_repr_dim
                ] = other_agent.goal
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
            "agent_0": np.zeros(self.goal_repr_dim, dtype=int),
            "agent_1": np.zeros(self.goal_repr_dim, dtype=int),
        }
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
                    individual_achieved_goals["agent_0"],
                    individual_achieved_goals["agent_1"],
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
                    (np.all(agent.goal == individual_achieved_goals["agent_0"]))
                    or (np.all(agent.goal == individual_achieved_goals["agent_1"]))
                ):
                    reward = 1
                else:
                    reward = 0
            rewards[agent.name] = reward
            done = bool(reward) or self.playground.done or not self.engine.game_on
            dones[agent.name] = done
            # logging which goal line the agent achieved (-1 means no goal line)
            info[agent.name] = {"goal_line": agent.reward - 1}
            if done and not self.fixed_goal:
                # If self.fixed_goal we are in evaluation mode, and don't want to update LP
                agent_goal_name = "".join(str(t) for t in agent.goal)
                agent.reward_list[agent_goal_name].append(reward)
                # LP
                if len(agent.reward_list[agent_goal_name]) >= self.update_lp:
                    agent.competence[agent_goal_name] = np.mean(
                        agent.reward_list[agent_goal_name][-self.update_lp :]
                    )
                    if len(agent.reward_list[agent_goal_name]) >= 2 * self.update_lp:
                        agent.learning_progress[agent_goal_name] = agent.competence[
                            agent_goal_name
                        ] - np.mean(
                            agent.reward_list[agent_goal_name][
                                -2 * self.update_lp : -self.update_lp
                            ]
                        )
                info[agent.name]["learning_progress"] = agent.learning_progress
                info[agent.name]["competence"] = agent.competence

                # Joint LP
                agent_goal = self.goal_space.index(agent.goal)
                other_agent = [a for a in self.playground.agents if a != agent][0]
                other_agent_goal = self.goal_space.index(other_agent.goal)
                agent.joint_reward_list[agent_goal][other_agent_goal].append(
                    rewards[agent.name]
                )
                if (
                    len(agent.joint_reward_list[agent_goal][other_agent_goal])
                    >= self.update_lp
                ):
                    agent.joint_competence[agent_goal][other_agent_goal] = np.mean(
                        agent.joint_reward_list[agent_goal][other_agent_goal][
                            -self.update_lp :
                        ]
                    )
                    if (
                        len(agent.joint_reward_list[agent_goal][other_agent_goal])
                        >= 2 * self.update_lp
                    ):
                        agent.joint_learning_progress[agent_goal][
                            other_agent_goal
                        ] = agent.joint_competence[agent_goal][
                            other_agent_goal
                        ] - np.mean(
                            agent.joint_reward_list[agent_goal][other_agent_goal][
                                -2 * self.update_lp : -self.update_lp
                            ]
                        )
                info[agent.name][
                    "joint_learning_progress"
                ] = agent.joint_learning_progress
                info[agent.name]["joint_competence"] = agent.joint_competence

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
        rewards, dones, info = self.compute_rewards()
        return observations, rewards, dones, info

    def reset(self):
        self.engine.reset()
        # All agents become active again
        self._active_agents = self.playground.agents.copy()
        # Each agent samples its own goal if not fixed
        if not self.fixed_goal:
            if self.centralized or np.random.random() < self.alignment_percentage:
                if (
                    self.learning_progress
                    and np.random.random() >= self.learning_progress
                ):
                    # Centralized LP strategy
                    sum_lp = [
                        l1 + l2
                        for l1, l2 in zip(
                            self.playground.agents[0].learning_progress.values(),
                            self.playground.agents[1].learning_progress.values(),
                        )
                    ]
                    scale_factor = sum([abs(lp) for lp in sum_lp])
                    if scale_factor == 0:
                        weights = [1 / self.goal_space_dim for _ in self.goal_space]
                    else:
                        weights = [abs(w) / scale_factor for w in sum_lp]
                    goal = self.goal_space[
                        np.random.choice(range(self.goal_space_dim), 1, p=weights)[0]
                    ]
                else:
                    # Centralized uniform (or e-greedy if LP)
                    # goal = self.goal_space[np.random.randint(0, self.goal_space_dim)]
                    # testing 0 shot generalization
                    goal = self.allowed_training_goals[
                        np.random.randint(0, self.n_allowed_training_goals)
                    ]
                for agent in self.playground.agents:
                    agent.goal = goal
            elif self.learning_progress:
                for agent in self.playground.agents:
                    # epsilon greedy
                    if np.random.random() < self.learning_progress:
                        agent.goal = self.goal_space[
                            np.random.randint(0, self.goal_space_dim)
                        ]
                    else:
                        # goals in lp are in the same order than in the goal space
                        # we can safely assign the weight 001 as the weight to the first int in the random sample
                        scale_factor = sum(
                            [abs(lp) for lp in agent.learning_progress.values()]
                        )
                        if scale_factor == 0:
                            weights = [1 / self.goal_space_dim for _ in self.goal_space]
                        else:
                            weights = [
                                abs(w) / scale_factor
                                for w in agent.learning_progress.values()
                            ]
                        agent.goal = self.goal_space[
                            np.random.choice(range(self.goal_space_dim), 1, p=weights)[
                                0
                            ]
                        ]
            elif self.joint_learning_progress:
                for agent in self.playground.agents:
                    # epsilon greedy
                    if np.random.random() < self.joint_learning_progress:
                        agent.goal = self.goal_space[
                            np.random.randint(0, self.goal_space_dim)
                        ]
                    else:
                        lps = np.max(abs(agent.joint_learning_progress), axis=1)
                        scale_factor = abs(lps).sum()
                        if scale_factor == 0:
                            weights = [1 / self.goal_space_dim for _ in self.goal_space]
                        else:
                            weights = abs(lps / scale_factor)
                        agent.goal = self.goal_space[
                            np.random.choice(range(self.goal_space_dim), 1, p=weights)[
                                0
                            ]
                        ]
            else:
                # independent uniform sampling

                # # Uncomment to only allow compatible goals
                # incompatible_goals = True
                # while incompatible_goals:
                #     for agent in self.playground.agents:
                #         agent.goal = self.goal_space[
                #             np.random.randint(0, self.goal_space_dim)
                #         ]
                #     if (
                #         np.bitwise_or(
                #             self.playground.agents[0].goal,
                #             self.playground.agents[1].goal,
                #         ).sum()
                #         <= 2
                #     ):
                #         incompatible_goals = False
                #
                # # Uncomment to only allow same collective goals
                # for agent in self.playground.agents:
                #     agent.goal = self.goal_space[
                #         np.random.randint(0, self.goal_space_dim)
                #     ]
                # if np.sum(self.playground.agents[0].goal) > 1:
                #     self.playground.agents[1].goal = self.playground.agents[0].goal
                # elif np.sum(self.playground.agents[1].goal) > 1:
                #     self.playground.agents[0].goal = self.playground.agents[1].goal
                #
                #
                # Uncomment to allow all gols during training
                for agent in self.playground.agents:
                    # agent.goal = self.goal_space[
                    #     np.random.randint(0, self.goal_space_dim)
                    # ]
                    # testing 0 shot
                    agent.goal = self.allowed_training_goals[
                        np.random.randint(0, self.n_allowed_training_goals)
                    ]

        self.engine.elapsed_time = 0
        self.episodes += 1
        self.engine.update_observations()
        self.time_steps = 0
        observations = self.process_obs()
        return observations

    def set_external_goal(
        self, external_goals: Dict[str, int] = None, fix_goal: bool = True
    ):
        # Not sample goal on each reset
        self.fixed_goal = fix_goal
        for agent in self.playground.agents:
            agent.goal = self.goal_space[external_goals[agent.name]]

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


class NewEnv(GoalLinesEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config):
        super(MultiAgentEnv, self).__init__()

        # If agents should sample goals centralized or decentralized
        self.centralized = config.get("centralized", False)
        # If goal is fixed or might be updated
        self.fixed_goal = False
        # If policies are conditioned on both goals
        self.double_condition = config.get("double_condition", False)
        # If independent agents might get aligned sometimes
        self.alignment_percentage = config.get("alignment_percentage", 0.0)
        # If give reward based on subgoals instead of collective goal
        self.subgoal = config.get("subgoal", False)

        # Goal space
        self.goal_space = [[1, 0], [0, 1]]
        self.goal_space_dim = len(self.goal_space)
        self.goal_repr_dim = 2

        self.episodes = 0
        self.time_steps = 0

        # Create playground
        self.playground = LineRooms(
            size=(450, 100),
            number_rooms=3,
            doorstep_size=50,
            random_doorstep_position=False,
        )

        room = self.playground.grid_rooms[0][1]
        left_door = room.doorstep_left.generate_door()
        right_door = room.doorstep_right.generate_door()
        self.playground._left_door = left_door
        self.playground._right_door = right_door
        self.playground.add_element(left_door)
        self.playground.add_element(right_door)

        left_switch = OpenCloseSwitch(door=left_door, size=(10, 10))
        self.playground.add_element(
            left_switch, room.get_random_position_on_wall("left", left_switch)
        )
        right_switch = OpenCloseSwitch(door=right_door, size=(10, 10))
        self.playground.add_element(
            right_switch, room.get_random_position_on_wall("right", right_switch)
        )

        left_zone = MultiAgentRewardZone(
            reward=1, physical_shape="rectangle", texture=[255, 0, 0], size=(25, 25)
        )

        right_zone = MultiAgentRewardZone(
            reward=100, physical_shape="rectangle", texture=[255, 0, 0], size=(25, 25)
        )

        self.playground.add_element(left_zone, ((25, 50), 0))
        self.playground.add_element(right_zone, ((425, 50), 0))

        # Add agents
        self._agent_ids = set()

        # Agent 0 (the one that can open doors)
        agent = BaseAgent(
            controller=External(),
            interactive=True,
            name="agent_0",
            texture=UniqueCenteredStripeTexture(
                color=(0, 200, 0), color_stripe=(0, 0, 200), size_stripe=4
            ),
        )
        ignore_parts_of_playground = [
            w for w in self.playground.elements if (isinstance(w, Wall))
        ]

        ignore_elements = [
            agent.parts,
            agent.base_platform,
        ] + ignore_parts_of_playground
        agent.add_sensor(
            PerfectSemantic(
                agent.base_platform,
                invisible_elements=ignore_elements,
                min_range=0,
                max_range=450,
                name="sensor",
                normalize=True,
            )
        )
        self.playground.add_agent(agent, ((225, 25), 3.14 / 2))
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
        ignore_elements = [
            agent.parts,
            agent.base_platform,
        ] + ignore_parts_of_playground
        agent.add_sensor(
            PerfectSemantic(
                agent.base_platform,
                invisible_elements=ignore_elements,
                min_range=0,
                max_range=450,
                name="sensor",
                normalize=True,
            )
        )
        self.playground.add_agent(agent, ((225, 75), 3.14 / 2))
        self._agent_ids.add("agent_1")

        # Init engine
        self.engine = Engine(playground=self.playground, time_limit=NEW_ENV_TIMELIMIT)

        # Define action and observation space
        actuators = self.playground.agents[0].controller.controlled_actuators
        # Discrete action space
        act_spaces = []
        for actuator in actuators:
            if isinstance(actuator, ContinuousActuator):
                act_spaces.append(3)
            else:
                act_spaces.append(2)
        self.action_space = spaces.MultiDiscrete(act_spaces)

        # Continuous observation space + goal representation as ohe
        # if double condition (we condition on both goals) obs space + other goal ohe + own goal ohe
        number_of_elements = (
            1 + len(self.playground.elements) - len(ignore_parts_of_playground)
        )
        if not self.double_condition:
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
        else:
            self.observation_space = spaces.Box(
                low=np.hstack(
                    (
                        np.array(
                            [[0, -2 * np.pi] for i in range(number_of_elements)]
                        ).flatten(),
                        np.zeros(self.goal_repr_dim),
                        np.zeros(self.goal_repr_dim),
                    )
                ),
                high=np.hstack(
                    (
                        np.array(
                            [[1, 2 * np.pi] for i in range(number_of_elements)]
                        ).flatten(),
                        np.ones(self.goal_repr_dim),
                        np.ones(self.goal_repr_dim),
                    )
                ),
                dtype=np.float64,
            )

        # Mapping to keep consistent coordinates of observations for the same objects
        # Elements will have the first coordinates and then the agent
        valid_elements = [
            element
            for element in self.playground.elements
            if not element in ignore_parts_of_playground
        ]
        for j, agent in enumerate(self.playground.agents):
            agent.COORDINATE_MAP = {
                element: 2 * i for i, element in enumerate(valid_elements)
            }
            agent.COORDINATE_MAP[self.playground.agents[j - 1].parts[0]] = (
                len(valid_elements) * 2
            )
        # List of active agents, agents can exit early if completed their goal
        self._active_agents = self.playground.agents.copy()

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

    def compute_rewards(self):
        """
        Goals:
        0: left, 1: right
        """
        rewards = {}
        dones = {}
        info = {agent.name: {} for agent in self._active_agents}

        agent_1_observed_reward = self.playground.agents[1].reward
        for agent in self._active_agents:
            reward = 0
            if self.subgoal:
                if agent.name == "agent_0":
                    if (
                        agent.goal == self.goal_space[0]
                        and self.playground._left_door.opened
                    ):
                        reward = 1
                    elif (
                        agent.goal == self.goal_space[1]
                        and self.playground._right_door.opened
                    ):
                        reward = 1
                elif agent.name == "agent_1":
                    if agent.goal == self.goal_space[0] and (
                        0 < agent_1_observed_reward < 100
                    ):
                        reward = 1
                    elif agent.goal == self.goal_space[1] and (
                        100 < agent_1_observed_reward
                    ):
                        reward = 1
            else:
                if agent.goal == self.goal_space[0] and (
                    0 < agent_1_observed_reward < 100
                ):
                    reward = 1
                elif agent.goal == self.goal_space[1] and (
                    100 < agent_1_observed_reward
                ):
                    reward = 1
            rewards[agent.name] = reward
            done = bool(reward) or self.playground.done or not self.engine.game_on
            dones[agent.name] = done

        dones["__all__"] = all(dones.values())
        # Agents that are done are deleted from the list of active agents
        [
            self._active_agents.remove(agent)
            for agent in self._active_agents
            if dones[agent.name]
        ]
        return rewards, dones, info

    def reset(self):
        self.engine.reset()
        # All agents become active again
        self._active_agents = self.playground.agents.copy()
        # Each agent samples its own goal if not fixed
        if not self.fixed_goal:
            if self.centralized or np.random.random() < self.alignment_percentage:
                # Centralized uniform (or e-greedy if LP)
                goal = self.goal_space[np.random.randint(0, self.goal_space_dim)]
                for agent in self.playground.agents:
                    agent.goal = goal
            else:
                # independent uniform sampling
                for agent in self.playground.agents:
                    agent.goal = self.goal_space[
                        np.random.randint(0, self.goal_space_dim)
                    ]
        self.engine.elapsed_time = 0
        self.episodes += 1
        self.engine.update_observations()
        self.time_steps = 0
        observations = self.process_obs()
        return observations


class LargeGoalLinesEnv(GoalLinesEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config):
        super(MultiAgentEnv, self).__init__()

        # If action space is continuous or discrete
        self.continuous = config["continuous"]
        # If agents should sample goals centralized or decentralized
        self.centralized = config.get("centralized", False)
        # If use learning progress or not. learning_progress is the epsilon value in exploration (0 acts as a flag for not using LP)
        self.learning_progress = config.get("learning_progress", 0)
        # If use joint learning progress or not.
        self.joint_learning_progress = config.get("joint_learning_progress", 0)
        # Number of episodes for updating LP
        self.update_lp = config.get("update_lp", 500)
        # If goal is fixed or might be updated
        self.fixed_goal = False
        # If policies are conditioned on both goals
        self.double_condition = config.get("double_condition", False)
        # If independent agents might get aligned sometimes
        self.alignment_percentage = config.get("alignment_percentage", 0.0)

        # Goal space
        landmarks = 4
        # self.goal_space = np.eye(landmarks, dtype=np.uint8).tolist()
        # self.goal_space += (
        #     np.array(list(combinations(self.goal_space, 2))).sum(1).tolist()
        # )
        individual_goals = np.eye(landmarks, dtype=np.uint8).tolist()
        self.goal_space = (
            np.array(list(combinations(individual_goals, 2))).sum(1).tolist()
        )
        self.goal_space_dim = len(self.goal_space)
        self.goal_repr_dim = landmarks

        self.episodes = 0
        self.time_steps = 0

        # Create playground
        self.playground = SingleRoom(size=(400, 400))
        zone_0 = MultiAgentRewardZone(
            reward=1,
            physical_shape="rectangle",
            texture=[255, 0, 0],
            size=(100, 20),
        )
        zone_1 = MultiAgentRewardZone(
            reward=100,
            physical_shape="rectangle",
            texture=[0, 255, 0],
            size=(100, 20),
        )
        zone_2 = MultiAgentRewardZone(
            reward=10_000,
            physical_shape="rectangle",
            texture=[255, 255, 255],
            size=(100, 20),
        )
        zone_3 = MultiAgentRewardZone(
            reward=1_000_000,
            physical_shape="rectangle",
            texture=[0, 255, 255],
            size=(100, 20),
        )
        self.playground.add_element(zone_0, ((50, 390), 0))
        self.playground.add_element(zone_1, ((50, 10), 0))
        self.playground.add_element(zone_2, ((350, 10), 0))
        self.playground.add_element(zone_3, ((350, 390), 0))

        # Add agents
        self._agent_ids = set()

        agent_sampler = CoordinateSampler(
            (200, 200), area_shape="rectangle", size=(300, 300)
        )

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
        agent.learning_progress = {
            "".join(str(t) for t in goal): 0.0 for goal in self.goal_space
        }
        agent.competence = {
            "".join(str(t) for t in goal): 0.0 for goal in self.goal_space
        }
        agent.reward_list = {
            "".join(str(t) for t in goal): [] for goal in self.goal_space
        }
        agent.joint_learning_progress = np.zeros(
            (self.goal_space_dim, self.goal_space_dim)
        )
        agent.joint_competence = np.zeros((self.goal_space_dim, self.goal_space_dim))
        agent.joint_reward_list = {
            i: {i: [] for i in range(self.goal_space_dim)}
            for i in range(self.goal_space_dim)
        }
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
        agent.learning_progress = {
            "".join(str(t) for t in goal): 0.0 for goal in self.goal_space
        }
        agent.competence = {
            "".join(str(t) for t in goal): 0.0 for goal in self.goal_space
        }
        agent.reward_list = {
            "".join(str(t) for t in goal): [] for goal in self.goal_space
        }
        agent.joint_learning_progress = np.zeros(
            (self.goal_space_dim, self.goal_space_dim)
        )
        agent.joint_competence = np.zeros((self.goal_space_dim, self.goal_space_dim))
        agent.joint_reward_list = {
            i: {i: [] for i in range(self.goal_space_dim)}
            for i in range(self.goal_space_dim)
        }
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
                if isinstance(actuator, ContinuousActuator):
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
        # if double condition (we condition on both goals) obs space + other goal ohe + own goal ohe
        number_of_elements = len(self.playground.elements) + 1
        if not self.double_condition:
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
        else:
            self.observation_space = spaces.Box(
                low=np.hstack(
                    (
                        np.array(
                            [[0, -2 * np.pi] for i in range(number_of_elements)]
                        ).flatten(),
                        np.zeros(self.goal_repr_dim),
                        np.zeros(self.goal_repr_dim),
                    )
                ),
                high=np.hstack(
                    (
                        np.array(
                            [[1, 2 * np.pi] for i in range(number_of_elements)]
                        ).flatten(),
                        np.ones(self.goal_repr_dim),
                        np.ones(self.goal_repr_dim),
                    )
                ),
                dtype=np.float64,
            )

        # Mapping to keep consistent coordinates of observations for the same objects
        # Elements will have the first coordinates and then the agent
        for j, agent in enumerate(self.playground.agents):
            agent.COORDINATE_MAP = {
                element: 2 * i for i, element in enumerate(self.playground.elements)
            }
            agent.COORDINATE_MAP[self.playground.agents[j - 1].parts[0]] = (
                len(self.playground.elements) * 2
            )
        # List of active agents, agents can exit early if completed their goal
        self._active_agents = self.playground.agents.copy()

    def compute_rewards(self):
        """
        If goal is individual, the agent must solve it by itself.
        If goal is collective, it doesn't matter which of the two
        goal zones an agent is touching (since both of them are
        required)
        """
        individual_achieved_goals = {
            "agent_0": np.zeros(self.goal_repr_dim, dtype=int),
            "agent_1": np.zeros(self.goal_repr_dim, dtype=int),
        }
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
                elif agent.reward < 1_000_000:
                    agent.reward = 3
                else:
                    agent.reward = 4
                individual_achieved_goals[agent.name][agent.reward - 1] = 1
        # Computing collective goal
        collective_achieved_goal = np.bitwise_or.reduce(
            np.vstack(
                [
                    individual_achieved_goals["agent_0"],
                    individual_achieved_goals["agent_1"],
                ]
            ),
            axis=0,
        )
        # Checking if achieved goal is desired goal (only for active agents)
        for agent in self._active_agents:
            if (
                np.sum(agent.goal) > 1
                and np.all(agent.goal == collective_achieved_goal)
            ) or (np.all(agent.goal == individual_achieved_goals[agent.name])):
                reward = 1
            else:
                reward = 0
            rewards[agent.name] = reward
            done = bool(reward) or self.playground.done or not self.engine.game_on
            dones[agent.name] = done
            # logging which goal line the agent achieved (-1 means no goal line)
            info[agent.name] = {"goal_line": agent.reward - 1}
            if done and not self.fixed_goal:
                # If self.fixed_goal we are in evaluation mode, and don't want to update LP
                agent_goal_name = "".join(str(t) for t in agent.goal)
                agent.reward_list[agent_goal_name].append(reward)
                # LP
                if len(agent.reward_list[agent_goal_name]) >= self.update_lp:
                    agent.competence[agent_goal_name] = np.mean(
                        agent.reward_list[agent_goal_name][-self.update_lp :]
                    )
                    if len(agent.reward_list[agent_goal_name]) >= 2 * self.update_lp:
                        agent.learning_progress[agent_goal_name] = agent.competence[
                            agent_goal_name
                        ] - np.mean(
                            agent.reward_list[agent_goal_name][
                                -2 * self.update_lp : -self.update_lp
                            ]
                        )
                info[agent.name]["learning_progress"] = agent.learning_progress
                info[agent.name]["competence"] = agent.competence

                # Joint LP
                agent_goal = self.goal_space.index(agent.goal)
                other_agent = [a for a in self.playground.agents if a != agent][0]
                other_agent_goal = self.goal_space.index(other_agent.goal)
                agent.joint_reward_list[agent_goal][other_agent_goal].append(
                    rewards[agent.name]
                )
                if (
                    len(agent.joint_reward_list[agent_goal][other_agent_goal])
                    >= self.update_lp
                ):
                    agent.joint_competence[agent_goal][other_agent_goal] = np.mean(
                        agent.joint_reward_list[agent_goal][other_agent_goal][
                            -self.update_lp :
                        ]
                    )
                    if (
                        len(agent.joint_reward_list[agent_goal][other_agent_goal])
                        >= 2 * self.update_lp
                    ):
                        agent.joint_learning_progress[agent_goal][
                            other_agent_goal
                        ] = agent.joint_competence[agent_goal][
                            other_agent_goal
                        ] - np.mean(
                            agent.joint_reward_list[agent_goal][other_agent_goal][
                                -2 * self.update_lp : -self.update_lp
                            ]
                        )
                info[agent.name][
                    "joint_learning_progress"
                ] = agent.joint_learning_progress
                info[agent.name]["joint_competence"] = agent.joint_competence

        # Agents that are done are deleted from the list of active agents
        [
            self._active_agents.remove(agent)
            for agent in self._active_agents
            if dones[agent.name]
        ]
        dones["__all__"] = all(dones.values())
        return rewards, dones, info


class VeryLargeGoalLinesEnv(GoalLinesEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config):
        super(MultiAgentEnv, self).__init__()

        # If action space is continuous or discrete
        self.continuous = config.get("continuous", False)
        # If agents should sample goals centralized or decentralized
        self.centralized = config.get("centralized", False)
        # If goal is fixed or might be updated
        self.fixed_goal = False
        # If policies are conditioned on both goals
        self.double_condition = config.get("double_condition", False)
        # If independent agents might get aligned sometimes
        self.alignment_percentage = config.get("alignment_percentage", 0.0)
        # Epsilon greedy exploration for communication policy
        self.eps_communication = config.get("eps_communication", 0.1)
        # If consider all goals or only collective ones
        self.all_goals = config.get("all_goals", True)
        # If consider new reward scheme (individual == collective)
        self.new_reward = config.get("new_reward", False)
        # If agents will be blind to each other
        self.blind_agents = config.get("blind_agents", False)

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
        # Only allowing some goals during training to test generalization
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

        # Add agents
        self._agent_ids = set()

        agent_sampler = CoordinateSampler(
            (200, 200), area_shape="rectangle", size=(300, 300)
        )

        # Agent 0
        agent_0 = BaseAgent(
            controller=External(),
            interactive=False,
            name="agent_0",
            texture=UniqueCenteredStripeTexture(
                color=(0, 200, 0), color_stripe=(0, 0, 200), size_stripe=4
            ),
        )
        # Agent 1
        agent_1 = BaseAgent(
            controller=External(),
            interactive=False,
            name="agent_1",
            texture=UniqueCenteredStripeTexture(
                color=(0, 0, 200), color_stripe=(0, 200, 0), size_stripe=4
            ),
        )

        ignore_elements = [agent_0.parts, agent_0.base_platform]
        if self.blind_agents:
            ignore_elements += [agent_1.parts, agent_1.base_platform]
        agent_0.add_sensor(
            PerfectSemantic(
                agent_0.base_platform,
                invisible_elements=ignore_elements,
                min_range=0,
                max_range=400,
                name="sensor",
                normalize=True,
            )
        )
        agent_0.message = None
        self.playground.add_agent(agent_0, agent_sampler)
        self._agent_ids.add("agent_0")

        if not self.blind_agents:
            ignore_elements = [agent_1.parts, agent_1.base_platform]
        agent_1.add_sensor(
            PerfectSemantic(
                agent_1.base_platform,
                invisible_elements=ignore_elements,
                min_range=0,
                max_range=400,
                name="sensor",
                normalize=True,
            )
        )
        agent_1.message = None
        self.playground.add_agent(agent_1, agent_sampler)
        self._agent_ids.add("agent_1")

        # Init engine
        self.engine = Engine(
            playground=self.playground, time_limit=LARGE_GOAL_LINES_TIMELIMIT
        )

        # Define action and observation space
        actuators = agent_0.controller.controlled_actuators
        if not self.continuous:
            # Discrete action space
            act_spaces = []
            for actuator in actuators:
                if isinstance(actuator, ContinuousActuator):
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
        # if double condition (we condition on both goals) obs space + other goal ohe + own goal ohe
        if self.blind_agents:
            number_of_elements = len(self.playground.elements)
        else:
            number_of_elements = len(self.playground.elements) + 1
        if not self.double_condition:
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
        else:
            self.observation_space = spaces.Box(
                low=np.hstack(
                    (
                        np.array(
                            [[0, -2 * np.pi] for i in range(number_of_elements)]
                        ).flatten(),
                        np.zeros(self.goal_repr_dim),
                        np.zeros(self.goal_repr_dim),
                    )
                ),
                high=np.hstack(
                    (
                        np.array(
                            [[1, 2 * np.pi] for i in range(number_of_elements)]
                        ).flatten(),
                        np.ones(self.goal_repr_dim),
                        np.ones(self.goal_repr_dim),
                    )
                ),
                dtype=np.float64,
            )

        # Mapping to keep consistent coordinates of observations for the same objects
        # Elements will have the first coordinates and then the agent
        for j, agent in enumerate(self.playground.agents):
            agent.COORDINATE_MAP = {
                element: 2 * i for i, element in enumerate(self.playground.elements)
            }
            if not self.blind_agents:
                agent.COORDINATE_MAP[self.playground.agents[j - 1].parts[0]] = (
                    len(self.playground.elements) * 2
                )
        # List of active agents, agents can exit early if completed their goal
        self._active_agents = self.playground.agents.copy()

    def compute_rewards(self):
        """
        If goal is individual, the agent must solve it by itself.
        If goal is collective, it doesn't matter which of the two
        goal zones an agent is touching (since both of them are
        required)
        """
        individual_achieved_goals = {
            "agent_0": np.zeros(self.goal_repr_dim, dtype=int),
            "agent_1": np.zeros(self.goal_repr_dim, dtype=int),
        }
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
                    individual_achieved_goals["agent_0"],
                    individual_achieved_goals["agent_1"],
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
                    (np.all(agent.goal == individual_achieved_goals["agent_0"]))
                    or (np.all(agent.goal == individual_achieved_goals["agent_1"]))
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

    def reset(self):
        self.engine.reset()
        # All agents become active again
        self._active_agents = self.playground.agents.copy()
        # Each agent samples its own goal if not fixed
        if not self.fixed_goal:
            if self.centralized or np.random.random() < self.alignment_percentage:
                # Centralized uniform (or e-greedy if LP)
                goal = self.allowed_training_goals[
                    np.random.randint(0, self.n_allowed_training_goals)
                ]
                for agent in self.playground.agents:
                    agent.goal = goal
            else:
                # independent uniform sampling

                # # Uncomment to only allow compatible goals
                # incompatible_goals = True
                # while incompatible_goals:
                #     for agent in self.playground.agents:
                #         agent.goal = self.goal_space[
                #             np.random.randint(0, self.goal_space_dim)
                #         ]
                #     if (
                #         np.bitwise_or(
                #             self.playground.agents[0].goal,
                #             self.playground.agents[1].goal,
                #         ).sum()
                #         <= 2
                #     ):
                #         incompatible_goals = False
                #
                # # Uncomment to only allow same collective goals
                # for agent in self.playground.agents:
                #     agent.goal = self.goal_space[
                #         np.random.randint(0, self.goal_space_dim)
                #     ]
                # if np.sum(self.playground.agents[0].goal) > 1:
                #     self.playground.agents[1].goal = self.playground.agents[0].goal
                # elif np.sum(self.playground.agents[1].goal) > 1:
                #     self.playground.agents[0].goal = self.playground.agents[1].goal
                #
                #
                # Uncomment to allow all gols during training
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
