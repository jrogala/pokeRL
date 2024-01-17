from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import gymnasium as gym
from collections import deque

@dataclass
class RewardFunction(ABC):
    env: gym.Env
    
    @abstractmethod
    def get_reward(self):
        pass

@dataclass
class Reward():
    env: gym.Env
    list_reward_class: list[RewardFunction]
    
    list_reward_function: list[RewardFunction] = field(init=False, repr=True, default_factory=list)
    
    def __post_init__(self):
        for reward_class in self.list_reward_class:
            self.list_reward_function.append(reward_class(self.env))

    def get_reward(self):
        reward = 0
        for reward_function in self.list_reward_function:
            reward += reward_function.get_reward()
        return reward