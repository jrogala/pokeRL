from typing import Any

import numpy as np
from gymnasium import Env, Wrapper

from pokerl.env import PokemonBlueEnv


class StopAtPokemon(Wrapper):
    """Wrapper to stop the game when a pokemon is encountered"""

    def __init__(self, env: Env):
        super().__init__(env)

    def step(self, action) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, reward, truncated, terminated, info = self.env.step(action)
        if info["pokemon_level"].max() > 0:
            terminated = True
            return observation, reward, truncated, terminated, info
        return observation, reward, truncated, terminated, info


class RewardStopCheckpoint(Wrapper):
    """
    Checkpoint the reward value.
    """

    def __init__(self, env: PokemonBlueEnv):
        super().__init__(env)
        self.checkpointReward = {
            str(np.array([6, 5, 0])): True,
        }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, info

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        if self.checkpointReward.get(str(info["position"])) is not None:
            return observation, reward, truncated, True, info
        return observation, reward, truncated, terminated, info
