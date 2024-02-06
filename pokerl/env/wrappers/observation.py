from typing import Any

import numpy as np
from gymnasium import Env, Wrapper, spaces


class PositionObservation(Wrapper):
    """Wrapper for reward based on pokemon level"""

    def __init__(self, env: Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise ValueError(f"Observation space {env.observation_space} is not a Box")
        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(
                    low=env.observation_space.low,
                    high=env.observation_space.high,
                    shape=env.observation_space.shape,
                    dtype=np.int16,
                ),
                "position": spaces.Box(low=(-255), high=255, shape=(2,), dtype=np.int16),
            }
        )

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        observation = {"screen": observation, "position": info["absolute_position"]}
        return observation, reward, truncated, terminated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        observation = {"screen": observation, "position": np.zeros((2,), dtype=np.uint8)}
        if not self.observation_space.contains(observation):
            raise ValueError(f"Observation {observation} is not contained in the observation space")
        return observation, info
