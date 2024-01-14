from dataclasses import dataclass, field
import pyboy
from pyboy import WindowEvent
from pathlib import Path
import os
from logging import getLogger

current_folder = Path(os.path.dirname(os.path.realpath(__file__)), "../..")
@dataclass
class PyBoyInterface:
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
            self.game_wrapper.start_game()

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
