from dataclasses import dataclass, field

from pokerl.pyboyterface import PyBoyInterface
from pokerl.settings import Pokesettings
import os
from pyboy.plugins.game_wrapper_pokemon_gen1 import GameWrapperPokemonGen1
@dataclass
class PokemonBlueInterface(PyBoyInterface):
    """
    A class representing the interface for interacting with the Pokemon Blue game.
    """   
    def __post_init__(self):
        """
        Initializes the PokemonBlueInterface object.
        """
        self.rom_name = Pokesettings.rom_name.value
        super().__post_init__()
        self.pokemon_level = Pokesettings.pokemon_level.value

    def get_level_pokemon(self, pokemon_index: int) -> int:
        """
        Retrieves the level of a specific Pokemon.

        Parameters:
        - pokemon_index (int): The index of the Pokemon.

        Returns:
        - int: The level of the Pokemon.
        """
        return self.pyboy.get_memory_value(self.pokemon_level[pokemon_index])
    
    def get_badges(self) -> int:
        """
        Retrieves the number of badges obtained by the player.

        Returns:
        - int: The number of badges.
        """
        return self.pyboy.get_memory_value(Pokesettings.badges.value).bit_count()

    def get_player_position(self):
        """
        Retrieves the position of the player. TODO: fix this.

        Returns:
        - (tile, x, y): 
        """
        return self.pyboy.get_memory_value(Pokesettings.position.value)

    def play_debug(self):
        """
        Plays the game in debug mode, displaying player position, badges, and Pokemon levels.
        """
        while not self.pyboy.tick():
            print(
                (
                    f"Player position: {self.get_player_position()}",
                    f"Badges: {self.get_badges()}",
                    f"Pokemon levels: {[self.get_level_pokemon(i) for i in range(6)]}"
                ), end="\r" ,flush=True,
            )
        self.pyboy.stop()

def main():
    pokemonBlueFilename = Pokesettings.rom_name.value
    pokemonBlue = PokemonBlueInterface(pokemonBlueFilename, interactive=True)
    pokemonBlue.play_debug()

if __name__ == "__main__":
    main()
