from typing import Any

from gymnasium import Env, Wrapper, spaces


class ppFlattenInfo(Wrapper):
    """Wrapper flatten all info except the observation"""

    def __init__(self, env: Env):
        super().__init__(env)
        if isinstance(env.observation_space, spaces.Dict):
            # We flatten the observation space dict
            observation_observation_space = env.observation_space.spaces["screen"]
            info_observation_space = env.observation_space.spaces.copy()
            info_observation_space.pop("screen")
            self.spaces_observation_space = spaces.Dict(info_observation_space)
            flatten_observation_space = spaces.flatten_space(self.spaces_observation_space)
            self.observation_space = spaces.Dict(
                {"screen": observation_observation_space, "info": flatten_observation_space}
            )

    def step(self, action: Any) -> Any:
        observation, reward, truncated, terminated, info = self.env.step(action)
        screen = observation["screen"]
        obs_info = {k: v for k, v in info.items() if k != "screen"}
        obs_info_flat = spaces.flatten(self.spaces_observation_space, obs_info)
        obs = {"screen": screen, "info": obs_info_flat}
        return obs, reward, truncated, terminated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        screen = observation["screen"]
        obs_info = {k: v for k, v in info.items() if k != "screen"}
        obs_info_flat = spaces.flatten(self.spaces_observation_space, obs_info)
        obs = {"screen": screen, "info": obs_info_flat}
        return obs, info
