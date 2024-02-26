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

    def tick(self):
        """Tick is 24 gameboy tick (each movement take 24 gameboy tick)"""
        for _ in range(24):
            super().tick()
        self._tick += 1
        self._logger.debug("Tick: %i", self._tick)
        return self.get_info()

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
