from dataclasses import dataclass, field
from gymnasium import Env, Space, spaces
from pyboy import WindowEvent
from enum import Enum
from random import choice

from typing import Any
from pokerl.blueterface import PokemonBlueInterface


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

class PokemonBlueGym(Env, PokemonBlueInterface):   
    def __init__(self):
        super().__init__()
        self.screen = self.pyboy.botsupport_manager().screen()
        self.action_space = spaces.Discrete(len(GameboyAction))

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
        self.send_input(GameboyAction(action).value[0])
        self.tick()
        self.send_input(GameboyAction(action).value[1])
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        return observation, reward, done, info
    
    def reset(self):
        """Reset the environment.

        Returns:
            observation (ndarray): The initial observation of the environment.
        """
        if not self._started:
            self.game_wrapper.start_game(**self._kwargs)
            self._started = True
        else:
            self.game_wrapper.reset_game()
        return self._get_observation()
    
    def render():
        pass
    
    def close(self):
        """Close the environment."""
        self.pyboy.stop()
    
    def _get_observation(self):
        """Get the current observation of the environment.

        Returns:
            observation (ndarray): The current observation of the environment.
        """
        return self.screen.screen_ndarray()[:, :, 0]

    def _get_reward(self) -> float:
        """Get the reward obtained from the previous action.

        Returns:
            reward (float): The reward obtained from the previous action.
        """
        return 0
    
    def _get_done(self) -> bool:
        """Check whether the episode is done or not.

        Returns:
            done (bool): Whether the episode is done or not.
        """
        return False
    
    def _get_info(self) -> dict[str, Any]:
        """Get additional information about the step.

        Returns:
            info (dict): Additional information about the step.
        """
        return {}
    