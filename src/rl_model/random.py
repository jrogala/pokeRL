import random
from rl_model.agentTemplate import AgentTemplate
from dataclasses import dataclass, field
from collections import deque

class RandomAgent(AgentTemplate):
    def act(self, state):
        return random.randint(0, self.env.action_space.n - 1)

    def train(self):
        pass

@dataclass
class RandomAgentWithMemory(AgentTemplate):
    memory: deque = field(init=True, repr=True, default_factory=deque(maxlen=10000))

    def act(self, state):
        action = random.randint(0, self.env.action_space.n - 1)
        self.memory.append((state, action))
        return action

    def train(self):
        pass
