import os
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import pyboy  # type: ignore
from gymnasium import Env, spaces
from pyboy import WindowEvent

from pokerl.env.rewards.basic_reward import basic_reward

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
        self.pyboy.set_emulation_speed(1 if self.interactive else 0)
        self.screen = self.pyboy.botsupport_manager().screen()

        self._tick = 0
        self._started = False
        self._logger = getLogger(__name__)
        self._logger.setLevel("DEBUG")
        self.state_file = None

        self.action_space = spaces.Discrete(len(GameboyAction))
        self.action_space_convertissor = (
            GameboyAction.NOTHING,
            GameboyAction.UP,
            GameboyAction.DOWN,
            GameboyAction.LEFT,
            GameboyAction.RIGHT,
            GameboyAction.A,
            GameboyAction.B,
            GameboyAction.START,
            GameboyAction.SELECT,
        )
        self.observation_space = spaces.Box(low=-255, high=255, shape=(144, 160, 3), dtype=np.int16)
        self.reward_range = (0, 0)
        self.current_state = None

    def play(self):
        """Play the game."""
        try:
            while not self.tick():
                pass
        finally:
            self.close()

    def _start_game(self):
        """Start the game."""
        self._logger.debug("Starting game")
        self._started = True

    def _send_input(self, button: WindowEvent):
        """Send input to the gameboy."""
        self._logger.debug("Sending input: %s", button)
        self.pyboy.send_input(button)

    def tick(self):
        """Make a tick"""
        self._tick += 1
        self._logger.debug("Tick: %s", self._tick)
        self.pyboy.tick()

    def screen_image(self):
        """Get the current screen image."""
        return self.screen.screen_ndarray()

    def close(self):
        """Close the gameboy."""
        self._logger.debug("Closing")
        self.pyboy.stop()

    def step(self, action: int):
        """Make a step."""
        action_gameboy = self.action_space_convertissor[action]
        self._logger.debug("Step: %s", action_gameboy)
        self._send_input(action_gameboy.value[0])
        self.tick()
        self._send_input(action_gameboy.value[1])
        observation = self.screen_image()
        truncated = self._get_done()
        terminated = False
        info = self.get_info()
        new_state = {"observation": observation, "info": info}
        reward_delta = self._get_reward(self.current_state, new_state)
        self.current_state = new_state
        return observation, reward_delta, truncated, terminated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment.

        Returns:
            observation (ndarray): The initial observation of the environment.
        """
        super().reset()
        if self._started:
            self.pyboy.load_state(self.state_file)
        else:
            self.state_file = open(self.rom_path + ".state", "r+b")
            self.pyboy.save_state(self.state_file)
            self.state_file.close()
        self._tick = 0
        self._logger.debug("Resetting game")
        return self.screen_image(), self.get_info()

    def _get_reward(self, current_state, next_state) -> float:
        """Get the reward obtained from the previous action."""
        return basic_reward(current_state, next_state)

    def _get_done(self) -> bool:
        """Check whether the episode is done or not."""
        return False

    def get_info(self) -> dict[str, Any]:
        """Get additional information about the step."""
        info: dict = {
            "self.tick": self._tick,
        }
        return info
