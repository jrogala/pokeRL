from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path

@abstractmethod
@dataclass
class AgentTemplate(ABC):
    path: field(init=True, repr=True, default=None)
    
    def __post_init__(self):
        if self.path is not None:
            self.load(self.path)

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def act(self, state: Any) -> int:
        pass

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass
