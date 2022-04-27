import gym
import numpy as np
from gym import spaces
from simple_playgrounds.agent.actuators import ContinuousActuator
from simple_playgrounds.agent.agents import BaseAgent
from simple_playgrounds.agent.controllers import External
from simple_playgrounds.common.position_utils import CoordinateSampler
from simple_playgrounds.device.sensors.semantic import PerfectSemantic
from simple_playgrounds.element.elements.activable import RewardOnActivation
from simple_playgrounds.element.elements.basic import Wall
from simple_playgrounds.engine import Engine
from simple_playgrounds.playground.layouts import SingleRoom

PLAYGROUND_SIZE = (200, 200)
TIMELIMIT = 500


class SimpleEnv(gym.Env):
    """Simple Environment that follows gym interface using spg: one agent, single goal.
    Spawns an object that gives a reward to the agent when activated.
    """

    def __init__(self):
        super(SimpleEnv, self).__init__()

        self.episodes = 0
        self.time_steps = 0

        # Create playground
        # Minimal environment with 1 room and 1 goal
        self.playground = SingleRoom(
            size=PLAYGROUND_SIZE,
        )
        room = self.playground.grid_rooms[0][0]
        center_area, size_area = room.get_partial_area("up-left")
        spawn_area_fountain = CoordinateSampler(
            center_area, area_shape="rectangle", size=size_area
        )
        fountain = RewardOnActivation(
            1,
        )
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
                max_range=max(PLAYGROUND_SIZE),
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
        self.engine = Engine(playground=self.playground, time_limit=TIMELIMIT)

        # Define action and observation space
        # Discrete action space
        actuators = self.agent.controller.controlled_actuators
        act_spaces = []
        for actuator in actuators:
            if isinstance(actuators, ContinuousActuator):
                act_spaces.append(3)
            else:
                act_spaces.append(2)
        self.action_space = spaces.MultiDiscrete(act_spaces)

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
        info = {}
        actuators = self.agent.controller.controlled_actuators
        for actuator, act in zip(actuators, action):
            if isinstance(actuators, ContinuousActuator):
                actions_dict[actuator] = [-1, 0, 1][act]
            else:
                actions_dict[actuator] = [0, 1][act]
        self.engine.step({self.agent: actions_dict})
        self.engine.update_observations()
        reward = self.agent.reward
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

    def render(self, mode):
        return (255 * self.engine.generate_agent_image(self.agent)).astype(np.uint8)

    def close(self):
        self.engine.terminate()
