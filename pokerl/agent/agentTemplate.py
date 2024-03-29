from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym


@abstractmethod
@dataclass
class AgentTemplate(ABC):
    env: gym.Env = field(init=True, repr=True)

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def act(self, state: Any) -> int:
        pass

    @abstractmethod
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        pass
