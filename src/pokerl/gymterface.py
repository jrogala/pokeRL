from dataclasses import dataclass, field
from gymnasium import Env, spaces
import numpy as np
from numpy import dtype, ndarray, uint8
from pyboy import WindowEvent
from enum import Enum
from random import choice

from typing import Any
from pokerl.pyboyterface import PyBoyInterface


class GameboyAction(Enum):
    """
    An enum representing the possible actions that can be taken by the agent.
    """
    UP = (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP)
    DOWN = (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN)
    LEFT = (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT)
    RIGHT = (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)
    A = (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A)
    B = (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B)
    START = (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START)
    SELECT = (WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT)
    NOTHING = (WindowEvent.PASS, WindowEvent.PASS)
    
@dataclass
class PyBoyGym(Env, PyBoyInterface):
    """A Gym environment for Pokemon Blue."""

    def __post_init__(self):
        super().__post_init__()
        self.action_space = spaces.Discrete(len(GameboyAction))
        self.action_space_convertissor = [
            GameboyAction.UP,
            GameboyAction.DOWN,
            GameboyAction.LEFT,
            GameboyAction.RIGHT,
            GameboyAction.A,
            GameboyAction.B,
            GameboyAction.START,
            GameboyAction.SELECT,
            GameboyAction.NOTHING,
        ]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(144, 160, 1), dtype=np.uint8
        )
        self.reward = 0

    def step(self, action: int):
        """Take a step in the environment.

        Args:
            action (int): The action to take in the environment.

        Returns:
            observation (ndarray): The current observation of the environment.
            reward (float): The reward obtained from the previous action.
            done (bool): Whether the episode is done or not.
            info (dict): Additional information about the step.
        """
        action = self.action_space_convertissor[action]
        self._logger.debug(f"Step: {action}")
        self.send_input(GameboyAction(action).value[0])
        self.tick()
        self.send_input(GameboyAction(action).value[1])
        observation = self._get_observation()
        rewardDelta = self._get_reward_delta()
        self.reward += rewardDelta
        done = self._get_done()
        info = self._get_info()
        return observation, rewardDelta, done, info
    
    def reset(self):
        """Reset the environment.

        Returns:
            observation (ndarray): The initial observation of the environment.
        """
        self.reset_game()
        return self._get_observation()
    
    def render(self) -> ndarray[Any, dtype[uint8]]:
        return self.screen_image()
    
    def close(self):
        """Close the environment."""
        self.pyboy.stop()
    
    def _get_observation(self):
        """Get the current observation of the environment."""
        return self.screen.screen_ndarray()[:, :, 0]

    def _get_reward_delta(self) -> float:
        """Get the reward obtained from the previous action."""
        return 0
        raise NotImplementedError
    
    def _get_done(self) -> bool:
        """Check whether the episode is done or not."""
        return False
        raise NotImplementedError
    
    def _get_info(self) -> dict[str, Any]:
        """Get additional information about the step."""
        info = {}
        return info
        raise NotImplementedError