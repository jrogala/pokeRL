from dataclasses import dataclass

from pokerl.env.pyboygym import PyBoyGym
from pokerl.env.settings import Pokesettings


@dataclass
class PokemonBlueEnv(PyBoyGym):
    def __post_init__(self):
        self.rom_name = Pokesettings.rom_name.value
        super().__post_init__()
        self.pokemon_level = Pokesettings.pokemon_level.value

    def get_level_pokemon(self, pokemon_index: int) -> int:
        return self.pyboy.get_memory_value(self.pokemon_level[pokemon_index])

    def get_badges(self) -> int:
        return self.pyboy.get_memory_value(Pokesettings.badges.value).bit_count()

    def get_player_position(self):
        return self.pyboy.get_memory_value(Pokesettings.position.value)

    def play_debug(self):
        while not self.pyboy.tick():
            print(
                (
                    f"Player position: {self.get_player_position()}",
                    f"Badges: {self.get_badges()}",
                    f"Pokemon levels: {[self.get_level_pokemon(i) for i in range(6)]}",
                ),
                end="\r",
                flush=True,
            )
        self.pyboy.stop()

    def _get_reward_delta(self) -> float:
        if self.get_level_pokemon(0) > 7:
            return 1
        return 0

    def get_reward(self) -> float:
        return max(0, self.get_level_pokemon(0) - 7)

    def tick(self):
        for _ in range(24):
            self._logger.debug(f"Tick: {self._tick}")
            self.pyboy.tick()
            self._tick += 1


def play():
    pokemonBlue = PokemonBlueEnv(interactive=True)
    pokemonBlue.play_debug()
