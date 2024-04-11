import os
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import pyboy
from gymnasium import Env, spaces

from pokerl.env.settings import Pokesettings
from pokerl.env.tool import game_coord_to_global_coord, pprint_info

current_folder = Path(os.path.dirname(os.path.realpath(__file__)), "../..")


class GameboyAction(Enum):
    """
    An enum representing the possible actions that can be taken by the agent.
    """

    NOTHING = ""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    A = "a"
    B = "b"
    START = "start"
    SELECT = "select"


@dataclass
class PokemonHelper:
    pyboy: pyboy.PyBoy

    def get_level_pokemon(self, pokemon_index: int) -> int:
        """Get a list of pokemon level"""
        return self.pyboy.memory[Pokesettings.pokemon_level[pokemon_index]]

    def get_hp_pokemon(self, pokemon_index: int) -> int:
        """Get the HP of a pokemon given its index"""
        add = Pokesettings.pokemon_hp[pokemon_index]
        return 256 * self.pyboy.memory[add] + self.pyboy.memory[add + 1]

    def get_max_hp_pokemon(self, pokemon_index: int) -> int:
        """Get the MAX HP of a pokemon given its index"""
        add = Pokesettings.pokemon_max_hp[pokemon_index]
        return 256 * self.pyboy.memory[add] + self.pyboy.memory[add + 1]

    def get_ennemy_hp(self) -> int:
        add = Pokesettings.ennemy_hp
        return 256 * self.pyboy.memory[add] + self.pyboy.memory[add + 1]

    def get_badges(self) -> int:
        """Get the badges count"""
        return self.pyboy.memory[Pokesettings.badges].bit_count()

    def get_owned_pokemon(self) -> list[int]:
        """Get the owned pokemon"""
        return np.array([self.pyboy.memory[i] for i in Pokesettings.owned_pokemon])

    def get_player_position(self):
        """Get the player position"""
        x = self.pyboy.memory[Pokesettings.position[0]]
        y = self.pyboy.memory[Pokesettings.position[1]]
        tyle = self.pyboy.memory[Pokesettings.map_address]
        return np.array([x, y, tyle])

    def get_combat_turn(self) -> int:
        return self.pyboy.memory[Pokesettings.combat_turn]

    def get_info(self) -> dict[str, Any]:
        pos = self.get_player_position()
        poke_info = {
            "pokemon_level": np.array([self.get_level_pokemon(i) for i in range(6)]),
            "badges": np.array(self.get_badges()),
            "position": np.array(pos),
            "absolute_position": np.array(game_coord_to_global_coord(*pos)),
            "owned_pokemon": np.array(self.get_owned_pokemon()),
            "start_combat": self.get_combat_turn() == 0,
        }
        return poke_info


@dataclass
class PokemonBlueEnv(Env):
    rom_name: str = field(default=Pokesettings.rom_name, init=True)
    save_state: str = field(default="starter_feu", init=True)
    interactive: bool = field(default=False, init=True)

    def __post_init__(self):
        self.rom_path = str(Path(current_folder, "rom", self.rom_name))
        self.save_state_path = str(Path(current_folder, "states", f"{self.save_state}.state"))
        self.pyboy = pyboy.PyBoy(
            self.rom_path,
            window="SDL2" if self.interactive else "null",
        )
        self.pyboy.set_emulation_speed(1 if self.interactive else 0)
        self._tick = 0
        self._started = False
        self._logger = getLogger(__name__)
        self._logger.setLevel("DEBUG")
        with open(Path(self.save_state_path), "rb") as f:
            self.pyboy.load_state(f)
        # give access of pyboy to helper
        self.helper = PokemonHelper(self.pyboy)
        # gymnasium
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
        self.last_reward = 0
        self.current_state = None
        super().__init__()

    def play(self):
        """Play the game."""
        try:
            while self.pyboy.tick():
                obs, reward, truncated, terminated, info = self.step(0)
                pprint_info(info)
                pass
        finally:
            self.close()

    def _start_game(self):
        """Start the game."""
        self._logger.debug("Starting game")
        self._started = True

    def _send_input(self, button: str):
        """Send input to the gameboy."""
        self._logger.debug("Sending input: %s", button)
        self.pyboy.button(button)

    def screen_image(self):
        """Get the current screen image."""
        return self.pyboy.screen.ndarray[:, :, :3]  # RGBA -> RGB

    def get_info(self) -> dict[str, Any]:
        """Get additional information about the step."""
        info: dict = {
            "tick": self._tick,
            **self.helper.get_info(),
        }
        return info

    def close(self):
        """Close the gameboy."""
        self._logger.debug("Closing")
        self.pyboy.stop()

    def step(self, action: int):
        """Make a step."""
        # pokemon
        action_gameboy = self.action_space_convertissor[action].value
        if action_gameboy != GameboyAction.NOTHING.value:
            self.pyboy.button(action_gameboy)
        if self.interactive:
            (self.pyboy.tick(1) for _ in range(24))
        else:
            self.pyboy.tick(24, True)  # render only last frame
        self._tick += 1
        observation = self.screen_image()
        truncated = False
        terminated = False
        reward = 0
        info = self.get_info()
        return observation, reward, truncated, terminated, info

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
        if self.save_state:
            with open(self.save_state_path, "rb") as f:
                self.pyboy.load_state(f)
        self._tick = 0
        self._logger.debug("Resetting game")
        return self.screen_image(), self.get_info()

    @property
    def unwrapped(self):
        """Returns the base non-wrapped environment.

        Returns:
            Env: The base non-wrapped :class:`gymnasium.Env` instance
        """
        return self
