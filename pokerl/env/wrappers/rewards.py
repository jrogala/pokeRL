from collections import deque
from typing import Any

import numpy as np
from gymnasium import Wrapper

from pokerl.env import PokemonBlueEnv


class RewardIncreasingPokemonLevel(Wrapper):
    """
    Reward for increasing the level of the pokemon.
    """

    def __init__(self, env: PokemonBlueEnv, lambda_: float = 1.0):
        super().__init__(env)
        self.reward_range = (-np.inf, np.inf)
        self.lambda_ = lambda_
        self.level_pokemon = None

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        if self.level_pokemon is None:
            self.level_pokemon = info["pokemon_level"]
        new_level_pokemon = [info["pokemon_level"][i] for i in range(6)]
        for i in range(6):
            if new_level_pokemon[i] > self.level_pokemon[i]:
                reward += 1 * self.lambda_
        return observation, reward, truncated, terminated, info


class RewardIncreasingPositionExploration(Wrapper):
    """
    Reward for exploring new positions.
    """

    def __init__(self, env: PokemonBlueEnv, lambda_: float = 1.0):
        super().__init__(env)
        self.reward_range = (-np.inf, np.inf)
        self.history: set[tuple[int, int]] = set()
        self.lambda_ = lambda_

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, np.ndarray]]:
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
    Negative Reward when info and screen are the same between steps
    """

    def __init__(self, env: PokemonBlueEnv, lambda_: float = 1.0):
        super().__init__(env)
        self.reward_range = (-np.inf, np.inf)
        self.lambda_ = lambda_
        self.last_info = None

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        if self.last_info is None:
            self.last_info = info
            return observation, reward, truncated, terminated, info
        for k in info:
            if (np.array(info[k]) != np.array(self.last_info[k])).any() and k != "tick":
                self.last_info = info
                # There is a difference, no negative reward
                return observation, reward, truncated, terminated, info
        reward -= 1 * self.lambda_
        self.last_info = info
        return observation, reward, truncated, terminated, info


class RewardIncreasingCapturePokemon(Wrapper):
    """
    Positive Reward when capturing a new pokemon
    """

    def __init__(self, env: PokemonBlueEnv, lambda_: float = 1.0):
        super().__init__(env)
        self.reward_range = (-np.inf, np.inf)
        self.lambda_ = lambda_
        self.owned_pokemon = None

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        if self.owned_pokemon is None:
            self.owned_pokemon = info["owned_pokemon"]
        if (self.owned_pokemon != info["owned_pokemon"]).any():
            reward += 1 * self.lambda_
            self.owned_pokemon = info["owned_pokemon"]
        return observation, reward, truncated, terminated, info


class RewardIncreasingBadges(Wrapper):
    """
    Positive Reward when getting a new badge
    """

    def __init__(self, env: PokemonBlueEnv, lambda_: float = 1.0):
        super().__init__(env)
        self.reward_range = (-np.inf, np.inf)
        self.lambda_ = lambda_
        self.badges = None

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        if self.badges is None:
            self.badges = info["badges"]
        if self.badges != info["badges"]:
            reward += 1 * self.lambda_
            self.badges = info["badges"]
        return observation, reward, truncated, terminated, info


class RewardToInfo(Wrapper):
    """
    Send reward value to info.
    """

    def __init__(self, env: PokemonBlueEnv):
        super().__init__(env)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        info["rewardDelta"] = 0
        info["reward"] = 0
        return observation, info

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        info["rewardDelta"] = reward
        info["reward"] += reward
        return observation, reward, truncated, terminated, info


class RewardHistoryToInfo(Wrapper):
    """
    Send reward value to info.
    """

    def __init__(self, env: PokemonBlueEnv, history_size: int = 10):
        super().__init__(env)
        self.history_size = history_size
        self.rewardHistory = deque(maxlen=history_size)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.rewardHistory.clear()
        info["rewardHistory"] = self.rewardHistory
        return observation, info

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        self.rewardHistory.append(reward)
        info["rewardHistory"] = self.rewardHistory
        return observation, reward, truncated, terminated, info


class RewardCheckpoint(Wrapper):
    """
    Checkpoint the reward value.
    """

    def __init__(self, env: PokemonBlueEnv, lambda_: float = 1.0):
        super().__init__(env)
        self.checkpointReward = {
            str(np.array([6, 5, 0])): lambda_,
            str(np.array([1, 10, 0])): lambda_,
            str(np.array([31, 20, 1])): lambda_,
            str(np.array([[20, 29, 1]])): lambda_,
        }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, info

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        if self.checkpointReward.get(str(info["position"])) is not None:
            reward += self.checkpointReward[str(info["position"])]
            self.checkpointReward.pop(str(info["position"]))
        return observation, reward, truncated, terminated, info


class RewardIncreasingLandedAttack(Wrapper):
    """
    Positive Reward when hitting an ennemy pokemon
    """

    def __init__(self, env: PokemonBlueEnv, lambda_: float = 1.0):
        super().__init__(env)
        self.reward_range = (0, np.inf)
        self.lambda_ = lambda_
        self.ennemy_hp = None

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        if self.ennemy_hp is None:
            self.ennemy_hp = self.env.unwrapped.helper.get_ennemy_hp()
        new_ennemy_hp = self.env.unwrapped.helper.get_ennemy_hp()
        if self.ennemy_hp > new_ennemy_hp:
            reward += (self.ennemy_hp - new_ennemy_hp) * self.lambda_
        self.ennemy_hp = new_ennemy_hp
        return observation, reward, truncated, terminated, info


class RewardDecreasingLostBattle(Wrapper):
    """
    Negative Reward when all pokemon have no hp left
    """

    def __init__(self, env: PokemonBlueEnv, lambda_: float = 1.0):
        super().__init__(env)
        self.reward_range = (-np.inf, 0)
        self.lambda_ = lambda_

    def step(self, action):
        observation, reward, truncated, terminated, info = self.env.step(action)
        if all(self.env.unwrapped.helper.get_hp_pokemon(i) == 0 for i in range(6)):
            reward -= 1 * self.lambda_
        return observation, reward, truncated, terminated, info
