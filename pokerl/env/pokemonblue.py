from dataclasses import dataclass

import numpy as np

from pokerl.env.pyboygym import PyBoyGym
from pokerl.env.settings import Pokesettings
from pokerl.env.tool import game_coord_to_global_coord


@dataclass
class PokemonBlueEnv(PyBoyGym):
    """Pokemon blue environment"""

    def __post_init__(self):
        """Load default rom and set reward"""
        self.rom_name = Pokesettings.rom_name
        super().__post_init__()

    def step(self, action: int):
        """Make a step."""
        action_gameboy = self.action_space_convertissor[action]
        self._logger.debug("Step: %s", action_gameboy)
        if self.interactive:
            self.pyboy._rendering(False) #Disable rendering for speedup
        self._send_input(action_gameboy.value[0])
        self.tick()
        self._send_input(action_gameboy.value[1])
        for _ in range(24-3):
            self.tick()
        if self.interactive:
            self.pyboy._rendering(True)
        observation = self.screen_image()
        truncated = self.get_done()
        terminated = False
        info = self.get_info()
        reward = self.get_reward()
        return observation, reward, truncated, terminated, info

    def get_level_pokemon(self, pokemon_index: int) -> int:
        """Get a list of pokemon level"""
        return self.pyboy.get_memory_value(Pokesettings.pokemon_level[pokemon_index])

    def get_badges(self) -> int:
        """Get the badges count"""
        return self.pyboy.get_memory_value(Pokesettings.badges).bit_count()

    def get_owned_pokemon(self) -> list[int]:
        """Get the owned pokemon"""
        return np.array([self.pyboy.get_memory_value(i) for i in Pokesettings.owned_pokemon])

    def get_player_position(self):
        """Get the player position"""
        x = self.pyboy.get_memory_value(Pokesettings.position[0])
        y = self.pyboy.get_memory_value(Pokesettings.position[1])
        tyle = self.pyboy.get_memory_value(Pokesettings.map_address)
        return np.array([x, y, tyle])

    def get_info(self):
        info = super().get_info()
        pos = self.get_player_position()
        poke_info = {
            "pokemon_level": np.array([self.get_level_pokemon(i) for i in range(6)]),
            "badges": np.array(self.get_badges()),
            "position": np.array(pos),
            "absolute_position": np.array(game_coord_to_global_coord(*pos)),
            "owned_pokemon": np.array(self.get_owned_pokemon()),
        }
        info.update(poke_info)
        return info
