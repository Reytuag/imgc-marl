import collections
import os
from itertools import combinations
from typing import Dict, List

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = ""

import cv2
import numpy as np
from gym import spaces
from imgc_marl.envs.elements.zone import MultiAgentRewardZone
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
from simple_playgrounds.engine import Engine
from simple_playgrounds.playground.layouts import SingleRoom

font = cv2.FONT_HERSHEY_SIMPLEX

SIMPLE_PLAYGROUND = (300, 300)
SIMPLE_TIMELIMIT = 100
GOAL_LINES_TIMELIMIT = 250
SCALED_REWARD = 1e-6


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
        super(MultiAgentEnv, self).__init__()

        # If action space is continuous or discrete
        self.continuous = config["continuous"]
        # If agents should sample goals centralized or decentralized
        self.centralized = config.get("centralized", False)
        # If use learning progress or not. learning_progress is the epsilon value in exploration (0 acts as a flag for not using LP)
        self.learning_progress = config.get("learning_progress", 0)
        # Number of episodes for updating LP
        self.update_lp = config.get("update_lp", 100)
        # If goal is fixed or might be updated
        self.fixed_goal = False

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

        self.episodes = 0
        self.time_steps = 0

        # Create playground
        # One room with 3 goal zones
        self.playground = SingleRoom(size=(400, 400))
        # zone_0 = MultiAgentRewardZone(
        #     reward=1,
        #     physical_shape="rectangle",
        #     texture=[255, 0, 0],
        #     size=(25, 150),
        #     name="001",
        # )
        # zone_1 = MultiAgentRewardZone(
        #     reward=100,
        #     physical_shape="rectangle",
        #     texture=[0, 0, 255],
        #     size=(25, 150),
        #     name="010",
        # )
        # zone_2 = MultiAgentRewardZone(
        #     reward=10_000,
        #     physical_shape="rectangle",
        #     texture=[255, 255, 255],
        #     size=(25, 150),
        #     name="100",
        # )
        # self.playground.add_element(zone_0, ((125, 200), 0))
        # self.playground.add_element(zone_1, ((200, 200), 0))
        # self.playground.add_element(zone_2, ((275, 200), 0))
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

        # Add agents
        self._agent_ids = set()

        # sampler1 = CoordinateSampler((200, 50), area_shape="rectangle", size=(400, 100))
        # sampler2 = CoordinateSampler(
        #     (200, 350), area_shape="rectangle", size=(400, 100)
        # )
        # sampler3 = CoordinateSampler((50, 200), area_shape="rectangle", size=(25, 200))
        # sampler4 = CoordinateSampler((350, 200), area_shape="rectangle", size=(25, 200))
        # agent_sampler = MetaSampler([sampler1, sampler2, sampler3, sampler4])
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
        # Old LP
        # agent.competence = {
        #     "".join(str(t) for t in goal): 0.0 for goal in self.goal_space
        # }
        # New LP
        agent.competence = {
            "".join(str(t) for t in goal): collections.deque(maxlen=2)
            for goal in self.goal_space
        }
        agent.reward_list = {
            "".join(str(t) for t in goal): [] for goal in self.goal_space
        }
        agent.counts = {"".join(str(t) for t in goal): 0 for goal in self.goal_space}
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
        # Old LP
        # agent.competence = {
        #     "".join(str(t) for t in goal): 0.0 for goal in self.goal_space
        # }
        # New LP
        agent.competence = {
            "".join(str(t) for t in goal): collections.deque(maxlen=2)
            for goal in self.goal_space
        }
        agent.reward_list = {
            "".join(str(t) for t in goal): [] for goal in self.goal_space
        }
        agent.counts = {"".join(str(t) for t in goal): 0 for goal in self.goal_space}
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

    def process_obs(self):
        """Process observations to match RLlib API (a dict with obs for each agent) and append goal"""
        obs = dict()
        for agent in self._active_agents:
            agent_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            # append agents goal at the end of the obs
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
                agent.counts[agent_goal_name] += 1
                # OLD LP
                # agent.learning_progress[agent_goal_name] = (
                #     rewards[agent.name] - agent.competence[agent_goal_name]
                # ) / agent.counts[agent_goal_name]
                # agent.competence[agent_goal_name] += agent.learning_progress[
                #     agent_goal_name
                # ]
                # NEW LP
                agent.reward_list[agent_goal_name].append(rewards[agent.name])
                if agent.counts[agent_goal_name] % self.update_lp == 0:
                    assert len(agent.reward_list[agent_goal_name]) == self.update_lp
                    agent.competence[agent_goal_name].append(
                        np.mean(agent.reward_list[agent_goal_name])
                    )
                    agent.learning_progress[agent_goal_name] = (
                        agent.competence[agent_goal_name][-1]
                        - agent.competence[agent_goal_name][0]
                    )
                    agent.reward_list[agent_goal_name] = []
                info[agent.name]["learning_progress"] = agent.learning_progress
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
            if self.centralized:
                goal = self.goal_space[np.random.randint(0, self.goal_space_dim)]
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


class ScaledGoalLinesEnv(GoalLinesEnv):
    """ScaledGoalLinesEnv for multiple agents. Goal conditioned.
    There are different goal areas to be targeted as objectives"""

    def __init__(self, config):
        super(MultiAgentEnv, self).__init__()

        # If action space is continuous or discrete
        self.continuous = config["continuous"]
        # If agents should sample goals centralized or decentralized
        self.centralized = config.get("centralized", False)
        # If goal is fixed or might be updated
        self.fixed_goal = False

        self.episodes = 0
        self.time_steps = 0

        # Goal space
        self.goal_space = np.eye(9, dtype=np.uint8).tolist()
        self.goal_space += (
            np.array(list(combinations(self.goal_space, 2))).sum(1).tolist()
        )
        self.goal_space_dim = len(self.goal_space)
        self.goal_repr_dim = 9

        # Create playground
        # One room with 9 goal zones
        self.playground = SingleRoom(size=(500, 700))
        zone_0 = MultiAgentRewardZone(
            reward=SCALED_REWARD,
            physical_shape="rectangle",
            texture=[255, 0, 0],
            size=(50, 100),
        )
        zone_1 = MultiAgentRewardZone(
            reward=SCALED_REWARD * 11,
            physical_shape="rectangle",
            texture=[0, 0, 255],
            size=(50, 100),
        )
        zone_2 = MultiAgentRewardZone(
            reward=SCALED_REWARD * (11**2),
            physical_shape="rectangle",
            texture=[255, 255, 255],
            size=(50, 100),
        )
        zone_3 = MultiAgentRewardZone(
            reward=SCALED_REWARD * (11**3),
            physical_shape="rectangle",
            texture=[255, 0, 0],
            size=(50, 100),
        )
        zone_4 = MultiAgentRewardZone(
            reward=SCALED_REWARD * (11**4),
            physical_shape="rectangle",
            texture=[0, 0, 255],
            size=(50, 100),
        )
        zone_5 = MultiAgentRewardZone(
            reward=SCALED_REWARD * (11**5),
            physical_shape="rectangle",
            texture=[255, 255, 255],
            size=(50, 100),
        )
        zone_6 = MultiAgentRewardZone(
            reward=SCALED_REWARD * (11**6),
            physical_shape="rectangle",
            texture=[255, 0, 0],
            size=(50, 100),
        )
        zone_7 = MultiAgentRewardZone(
            reward=SCALED_REWARD * (11**7),
            physical_shape="rectangle",
            texture=[0, 0, 255],
            size=(50, 100),
        )
        zone_8 = MultiAgentRewardZone(
            reward=SCALED_REWARD * (11**8),
            physical_shape="rectangle",
            texture=[255, 255, 255],
            size=(50, 100),
        )
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.playground.add_element(zone_0, ((125, 150), 0))
        self.playground.add_element(zone_1, ((250, 150), 0))
        self.playground.add_element(zone_2, ((375, 150), 0))
        self.playground.add_element(zone_3, ((125, 350), 0))
        self.playground.add_element(zone_4, ((250, 350), 0))
        self.playground.add_element(zone_5, ((375, 350), 0))
        self.playground.add_element(zone_6, ((125, 550), 0))
        self.playground.add_element(zone_7, ((250, 550), 0))
        self.playground.add_element(zone_8, ((375, 550), 0))

        self.playground.walls = [
            elem for elem in self.playground.elements if isinstance(elem, Wall)
        ]

        # Add agents
        self._agent_ids = set()

        sampler1 = CoordinateSampler((250, 50), area_shape="rectangle", size=(500, 100))
        sampler2 = CoordinateSampler(
            (250, 650), area_shape="rectangle", size=(500, 100)
        )
        sampler3 = CoordinateSampler((50, 350), area_shape="rectangle", size=(50, 700))
        sampler4 = CoordinateSampler((450, 350), area_shape="rectangle", size=(50, 700))
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
                max_range=700,
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
                max_range=700,
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
        number_of_elements = len(self.playground.walls) + 1
        self.observation_space = spaces.Box(
            low=np.hstack(
                (
                    np.array(
                        [[0, -2 * np.pi] for i in range(number_of_elements)]
                    ).flatten(),
                    np.zeros(9),
                )
            ),
            high=np.hstack(
                (
                    np.array(
                        [[1, 2 * np.pi] for i in range(number_of_elements)]
                    ).flatten(),
                    np.ones(9),
                )
            ),
            dtype=np.float64,
        )
        # Mapping to keep consistent coordinates of observations for the same objects
        # Walls will have the first coordinates and then the agent
        for j, agent in enumerate(self.playground.agents):
            agent.COORDINATE_MAP = {
                wall: 2 * i for i, wall in enumerate(self.playground.walls)
            }
            agent.COORDINATE_MAP[self.playground.agents[j - 1].parts[0]] = 8

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
            "agent_0": np.zeros(9, dtype=int),
            "agent_1": np.zeros(9, dtype=int),
        }
        rewards = {}
        dones = {}
        info = {}
        # Computing individual achieved goals
        for agent in self.playground.agents:
            # Hack for identifying which goal is being activated by this agent
            if agent.reward:
                if agent.reward < SCALED_REWARD * 10:
                    agent.reward = 1
                elif agent.reward < SCALED_REWARD * (10**2):
                    agent.reward = 2
                elif agent.reward < SCALED_REWARD * (10**3):
                    agent.reward = 3
                elif agent.reward < SCALED_REWARD * (10**4):
                    agent.reward = 4
                elif agent.reward < SCALED_REWARD * (10**5):
                    agent.reward = 5
                elif agent.reward < SCALED_REWARD * (10**6):
                    agent.reward = 6
                elif agent.reward < SCALED_REWARD * (10**7):
                    agent.reward = 7
                elif agent.reward < SCALED_REWARD * (10**8):
                    agent.reward = 8
                else:
                    agent.reward = 9
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
            dones[agent.name] = (
                bool(reward) or self.playground.done or not self.engine.game_on
            )
            # logging which goal line the agent achieved (-1 means no goal line)
            info[agent.name] = {"goal_line": agent.reward - 1}
        # Agents that are done are deleted from the list of active agents
        [
            self._active_agents.remove(agent)
            for agent in self._active_agents
            if dones[agent.name]
        ]
        dones["__all__"] = all(dones.values())
        return rewards, dones, info
