import numpy as np
from gymnasium import Wrapper

from pokerl.env.pokemonblue import PokemonBlueEnv


class RewardLevel(Wrapper):
    """
    Reward for increasing the level of the pokemon.
    """

    def __init__(self, env: PokemonBlueEnv, lambda_: float = 1.0):
        super().__init__(env)
        self.reward_range = (-np.inf, np.inf)
        self.lambda_ = lambda_
        self.level_pokemon = [env.get_level_pokemon(i) for i in range(6)]

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        new_level_pokemon = [self.env.get_level_pokemon(i) for i in range(6)]
        reward += sum([new - old for new, old in zip(new_level_pokemon, self.level_pokemon)]) * self.lambda_
        return observation, reward, truncated, terminated, info


class RewardPositionExploration(Wrapper):
    """
    Reward for exploring new positions.
    """

    def __init__(self, env: PokemonBlueEnv, lambda_: float = 1.0):
        super().__init__(env)
        self.reward_range = (-np.inf, np.inf)
        self.history: set[tuple[int, int]] = set()
        self.lambda_ = lambda_

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        abs_pos_str = str(info["absolute_position"])
        if abs_pos_str not in self.history:
            reward += 1 * self.lambda_
            self.history.add(abs_pos_str)
        return observation, reward, truncated, terminated, info

class RewardDecreasingSteps(Wrapper):
    """
    Negative Reward for decreasing the number of steps.
    """

    def __init__(self, env: PokemonBlueEnv, lambda_: float = 1.0):
        super().__init__(env)
        self.reward_range = (-np.inf, np.inf)
        self.lambda_ = lambda_

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        reward -= 1 * self.lambda_
        return observation, reward, truncated, terminated, info

class RewardDecreasingNoChange(Wrapper):
    """
    Negative Reward when info is the same between steps
    """

    def __init__(self, env: PokemonBlueEnv, lambda_: float = 1.0):
        super().__init__(env)
        self.reward_range = (-np.inf, np.inf)
        self.lambda_ = lambda_
        self.info = None

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        if info == self.info:
            reward -= 1 * self.lambda_
        self.info = info
        return observation, reward, truncated, terminated, info
