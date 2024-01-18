from dataclasses import dataclass, field

from pokerl.env.pokemonblue import PokemonBlueEnv


@dataclass
class LevelPokemon:
    pokemonBlueEnv: PokemonBlueEnv
    last_level_pokemon: list[int] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.last_level_pokemon = [0 for _ in range(6)]

    def _get_reward(self) -> float:
        reward = 0
        for i in range(6):
            pokemon_level = self.pokemonBlueEnv.get_level_pokemon(i)
            if pokemon_level != self.last_level_pokemon[i]:
                reward += pokemon_level - self.last_level_pokemon[i]
                self.last_level_pokemon[i] = pokemon_level
        return reward
