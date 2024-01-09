from dataclasses import dataclass, field
import pyboy
from pyboy import WindowEvent
from pathlib import Path
import os
import gymnasium as gym
from ray import init
from logging import getLogger

current_folder = Path(os.path.dirname(os.path.realpath(__file__)), "../..")
_logger = getLogger(__name__)
@dataclass
class PyBoyInterface:
    rom_name: str = field(default="", init=True)
    interactive: bool = field(default=False, init=True, repr=False)

    def __post_init__(self):
        self.rom_path = str(Path(current_folder, "rom", self.rom_name))
        self.pyboy = pyboy.PyBoy(
            self.rom_path,
            game_wrapper=True,
            window_type="SDL2" if self.interactive else "headless",
        )
        self.game_wrapper = self.pyboy.game_wrapper()
        self.pyboy.set_emulation_speed(1 if self.interactive else 0)

        self._tick = 0

    def play(self):
        try:
            while not self.tick():
                pass
        finally:
            self.close()

    def play_debug(self):
        raise NotImplementedError

    def reset_game(self):
        _logger.debug("Resetting game")
        self.game_wrapper.reset_game()

    def send_input(self, button: WindowEvent):
        _logger.debug(f"Sending input: {button}")
        self.pyboy.send_input(button)
    
    def tick(self):
        self._tick += 1
        _logger.debug(f"Tick: {self._tick}")
        self.pyboy.tick()

    def screen_image(self):
        # Todo: Fix this
        return self.pyboy.screen_image()

    def close(self):
        _logger.debug("Closing")
        self.pyboy.stop()
