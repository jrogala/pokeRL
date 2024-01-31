from typing import Any
import numpy as np
from gymnasium import Wrapper, spaces
from gymnasium.spaces.utils import flatten

from pokerl.env.pokemonblue import PokemonBlueEnv


class RewardLevel(Wrapper):
    """Wrapper for reward based on pokemon level"""
    def __init__(self, env: PokemonBlueEnv):
        super().__init__(env)
        self.reward_range = (0, np.inf)

    def step(self, action):
        observation, _, truncated, terminated, info = self.env.step(action)
        reward = sum(info["level_pokemon"])
        return observation, reward, truncated, terminated, info

class RewardPositionExploriation(Wrapper):
    """Wrapper for reward based on pokemon level"""
    def __init__(self, env: PokemonBlueEnv):
        super().__init__(env)
        self.reward_range = (-np.inf, np.inf)
        self.history = set()

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        reward -= 1
        abs_pos_str = str(info["absolute_position"])
        if abs_pos_str not in self.history:
            reward += 100
            self.history.add(abs_pos_str)
        return observation, reward, truncated, terminated, info

class PositionObservation(Wrapper):
    """Wrapper for reward based on pokemon level"""
    def __init__(self, env: PokemonBlueEnv):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(
                low=self.env.observation_space.low,
                high=self.env.observation_space.high,
                shape=self.observation_space.shape,
                dtype=np.uint8
            ),
            "position": spaces.Box(low=-255, high=255, shape=(2,), dtype=np.uint8),
        })

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        self.observation_space.contains(observation)
        observation = {
            "screen": observation,
            "position": info["absolute_position"]
        }
        return observation, reward, truncated, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.array, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        if not self.observation_space.contains(observation):
            raise ValueError(f"Observation {observation.shape} is not contained in the observation space")
        observation = {
            "screen": observation,
            "position": np.zeros((2,), dtype=np.uint8)
        }
        return observation, info
