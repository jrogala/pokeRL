import os
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import pyboy
from gymnasium import Env, spaces
from numpy import dtype, ndarray, uint8
from pyboy import WindowEvent

current_folder = Path(os.path.dirname(os.path.realpath(__file__)), "../..")


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
class PyBoyGym(Env):
    rom_name: str = field(default="", init=True)
    interactive: bool = field(default=False, init=True)

    def __post_init__(self):
        self.rom_path = str(Path(current_folder, "rom", self.rom_name))
        self.pyboy = pyboy.PyBoy(
            self.rom_path,
            game_wrapper=True,
            window_type="SDL2" if self.interactive else "headless",
        )
        self.game_wrapper = self.pyboy.game_wrapper()
        self.pyboy.set_emulation_speed(1 if self.interactive else 0)
        self.screen = self.pyboy.botsupport_manager().screen()

        self._tick = 0
        self._started = False
        self._logger = getLogger(__name__)
        self._logger.setLevel("DEBUG")

        self.action_space = spaces.Discrete(len(GameboyAction))
        self.action_space_convertissor = [
            GameboyAction.NOTHING,
            GameboyAction.UP,
            GameboyAction.DOWN,
            GameboyAction.LEFT,
            GameboyAction.RIGHT,
            GameboyAction.A,
            GameboyAction.B,
            GameboyAction.START,
            GameboyAction.SELECT,
        ]
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 1), dtype=np.uint8)

    def play(self):
        try:
            while not self.tick():
                pass
        finally:
            self.close()

    def play_debug(self):
        raise NotImplementedError

    def start_game(self):
        self._logger.debug("Starting game")
        self.game_wrapper.start_game()
        self._started = True

    def reset_game(self):
        self._tick = 0
        if not self._started:
            self.start_game()
        else:
            self._logger.debug("Resetting game")
            self.game_wrapper.reset_game()

    def send_input(self, button: WindowEvent):
        self._logger.debug(f"Sending input: {button}")
        self.pyboy.send_input(button)

    def tick(self):
        self._tick += 1
        self._logger.debug(f"Tick: {self._tick}")
        self.pyboy.tick()

    def screen_image(self):
        # Todo: Fix this
        return self.screen.screen_ndarray()[:, :, 0]

    def close(self):
        self._logger.debug("Closing")
        self.pyboy.stop()

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
        rewardDelta = self._get_reward()
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

    def _get_observation(self):
        """Get the current observation of the environment."""
        return self.screen.screen_ndarray()[:, :, 0]

    def _get_reward(self) -> float:
        """Get the reward obtained from the previous action."""
        return 0

    def _get_done(self) -> bool:
        """Check whether the episode is done or not."""
        return False

    def _get_info(self) -> dict[str, Any]:
        """Get additional information about the step."""
        info = {}
        return info
