import numpy as np
from gymnasium import Wrapper

from pokerl.env.pokemonblue import PokemonBlueEnv


class WrapperLevel(Wrapper):
    """Wrapper for reward based on pokemon level"""
    def __init__(self, env: PokemonBlueEnv):
        if not isinstance(env, PokemonBlueEnv):
            raise TypeError("env must be an instance of PokemonBlueEnv")
        super().__init__(env)
        self.reward_range = (0, np.inf)

    def step(self, action):
        observation, _, truncated, terminated, info = self.env.step(action)
        reward = sum(info["level_pokemon"])
        return observation, reward, truncated, terminated, info
