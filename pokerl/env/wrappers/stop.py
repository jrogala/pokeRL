from typing import Any

from gymnasium import Env, Wrapper


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
