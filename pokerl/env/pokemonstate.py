from abc import ABC, abstractmethod


class PokemonState(ABC):
    @abstractmethod
    def get_level_pokemon(self, pokemon_index: int) -> int:
        pass

    @abstractmethod
    def get_badges(self) -> int:
        pass

    @abstractmethod
    def get_player_position(self):
        pass
