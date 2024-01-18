import random
from collections import deque
from dataclasses import dataclass, field

from pokerl.agent.agentTemplate import AgentTemplate


class RandomAgent(AgentTemplate):
    def act(self, state):
        return random.randint(0, self.env.action_space.n - 1)

    def train(self):
        pass


@dataclass
class RandomAgentWithMemory(AgentTemplate):
    memory: deque = field(init=False, repr=True, default=deque(maxlen=10000))

    def act(self, state):
        action = random.randint(0, self.env.action_space.n - 1)
        return action

    def train(self):
        pass

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        pass
