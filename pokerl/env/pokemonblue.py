from dataclasses import dataclass

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

    def get_level_pokemon(self, pokemon_index: int) -> int:
        """Get a list of pokemon level"""
        return self.pyboy.get_memory_value(Pokesettings.pokemon_level[pokemon_index])

    def get_badges(self) -> int:
        """Get the badges count"""
        return self.pyboy.get_memory_value(Pokesettings.badges).bit_count()

    def get_player_position(self):
        """Get the player position"""
        x = self.pyboy.get_memory_value(Pokesettings.position[0])
        y = self.pyboy.get_memory_value(Pokesettings.position[1])
        tyle = self.pyboy.get_memory_value(Pokesettings.map_address)
        return (x, y, tyle)

    def get_info(self):
        info = super().get_info()
        pos = self.get_player_position()
        poke_info = {
            "level_pokemon": [self.get_level_pokemon(i) for i in range(6)],
            "badges": self.get_badges(),
            "position": pos,
            "absolute_position": game_coord_to_global_coord(*pos)
        }
        info.update(poke_info)
        return info

def play():
    """Play pokemon blue"""
    PokemonBlueEnv(interactive=True).play()
