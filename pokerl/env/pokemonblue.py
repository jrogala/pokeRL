from dataclasses import dataclass, field

from pokerl.env.pokemonstate import PokemonState
from pokerl.env.pyboygym import PyBoyGym
from pokerl.env.rewards.reward import Reward, RewardFunction
from pokerl.env.settings import Pokesettings


@dataclass
class PokemonBlueEnv(PyBoyGym, PokemonState):
    reward_list: list[RewardFunction] = field(default_factory=list, init=True)

    def __post_init__(self):
        # Load default rom and set reward
        self.rom_name = Pokesettings.rom_name.value
        super().__post_init__()
        self.reward = Reward(env=self, list_reward_class=self.reward_list)

    def _get_reward(self) -> float:
        # Add reward. Default is 0
        return self.reward.get_reward()

    def tick(self):
        # Tick is 24 gameboy tick (each movement take 24 gameboy tick)
        for _ in range(24):
            self._logger.debug(f"Tick: {self._tick}")
            self.pyboy.tick()
            self._tick += 1

    # PokemonState methods
    def get_level_pokemon(self, pokemon_index: int) -> int:
        return self.pyboy.get_memory_value(self.pokemon_level[pokemon_index])

    def get_badges(self) -> int:
        return self.pyboy.get_memory_value(Pokesettings.badges.value).bit_count()

    def get_player_position(self):
        return self.pyboy.get_memory_value(Pokesettings.position.value)

def play():
    pokemonBlue = PokemonBlueEnv(interactive=True)
    pokemonBlue.play_debug()
