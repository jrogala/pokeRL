from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pokerl.env.pokemonstate import PokemonState


@dataclass
class RewardFunction(ABC):
    env: PokemonState

    @abstractmethod
    def get_reward(self):
        pass


@dataclass
class Reward:
    env: PokemonState
    list_reward_class: list[RewardFunction] = field(init=True, repr=False, default_factory=list)

    list_reward_function: list[RewardFunction] = field(init=False, repr=True, default_factory=list)

    def __post_init__(self):
        for reward_class in self.list_reward_class:
            self.list_reward_function.append(reward_class(self.env))

    def get_reward(self):
        reward = 0
        for reward_function in self.list_reward_function:
            reward += reward_function.get_reward()
        return reward
