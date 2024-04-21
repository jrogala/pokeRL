from typing import Any

import numpy as np
from gymnasium import Env, Wrapper, spaces


class ObservationDict(Wrapper):
    """
    Preprocess the observation space to be a dict

    This will separate the screen in the dict for adding more information
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = spaces.Dict({"screen": env.observation_space})

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        return {"screen": observation}, reward, truncated, terminated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        return {"screen": observation}, info


class ObservationAddPosition(Wrapper):
    """Wrapper for reward based on pokemon level"""

    def __init__(self, env: Env):
        super().__init__(env)
        if isinstance(env.observation_space, spaces.Dict):
            # We add the position to the observation space dict
            d_obs_space = env.observation_space.spaces
            d_obs_space = {**d_obs_space, "position": spaces.Box(low=(-1), high=1, shape=(2,), dtype=np.float16)}
            self.observation_space = spaces.Dict(d_obs_space)
        else:
            raise Exception("You should wrap your env in ObservationDict before using ObservationAddPosition")

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        observation |= {"position": info["absolute_position"]}
        observation["position"] = observation["position"] / 255
        return observation, reward, truncated, terminated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        observation |= {"position": np.zeros((2,), dtype=np.float16)}
        return observation, info


class ObservationAddPokemonLevel(Wrapper):
    """Wrapper for reward based on pokemon level"""

    def __init__(self, env: Env):
        super().__init__(env)
        if isinstance(env.observation_space, spaces.Dict):
            # We add the position to the observation space dict
            d_obs_space = env.observation_space.spaces
            d_obs_space = {**d_obs_space, "pokemon_level": spaces.Box(low=0, high=255, shape=(6,), dtype=np.uint8)}
            self.observation_space = spaces.Dict(d_obs_space)
        else:
            raise Exception("You should wrap your env in ObservationDict before using ObservationAddPokemonLevel")

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        observation |= {"pokemon_level": info["pokemon_level"]}
        return observation, reward, truncated, terminated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        observation |= {"pokemon_level": np.zeros((6,), dtype=np.uint8)}
        return observation, info


class ObservationRemoveScreen(Wrapper):
    """Wrapper for reward based on pokemon level"""

    def __init__(self, env: Env):
        super().__init__(env)
        if isinstance(env.observation_space, spaces.Dict):
            # We add the position to the observation space dict
            d_obs_space = env.observation_space.spaces
            d_obs_space.pop("screen")
            self.observation_space = spaces.Dict(d_obs_space)
            self.action_space = env.action_space
        else:
            raise Exception("You should wrap your env in ObservationDict before using ObservationAddPosition")

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        observation.pop("screen")
        return observation, reward, truncated, terminated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        observation.pop("screen")
        return observation, info
