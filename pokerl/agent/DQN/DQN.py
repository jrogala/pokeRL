import functools
from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch import argmax, nn
from torch.utils.data import DataLoader, TensorDataset

from pokerl.agent.agentTemplate import AgentTemplate


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.model(x)
        return x


@dataclass
class DQNAgent(AgentTemplate):
    env: gym.Env

    BATCH_SIZE: int = 128
    GAMMA: float = 0.99
    EPS_START: float = 1.0
    EPS_END: float = 0.01
    EPS_DECAY: float = 0.995
    TARGET_UPDATE: int = 10
    EPSILON: float = 0.1

    memory: deque = field(default_factory=deque, init=False)

    def __post_init__(self):
        observation_shape = functools.reduce(lambda x, y: x * y, self.env.observation_space.shape)
        action_dim = self.env.action_space.n

        self.policy_net = DQN(observation_shape, action_dim)
        self.target_net = DQN(observation_shape, action_dim)

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.criterion = nn.MSELoss()

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def act(self, state):
        if np.random.rand() < self.EPSILON:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).flatten(0)
                q_values = self.policy_net(state)
                action = argmax(q_values).item()
        return action

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, next_state, reward, done))

    def train(self, epoch=1):
        # Don't optimize the model if we don't yet have enough memory for a batch
        if len(self.memory) < self.BATCH_SIZE:
            return
        states, actions, next_states, rewards, dones = zip(*self.memory)

        states_tensor = torch.tensor(states).flatten(1).float()
        actions_tensor = torch.tensor(actions).unsqueeze(1).float()
        next_states_tensor = torch.tensor(next_states).flatten(1).float()
        rewards_tensor = torch.tensor(rewards).unsqueeze(1).float()
        dataset = TensorDataset(states_tensor, actions_tensor, next_states_tensor, rewards_tensor)
        dataloader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        for _ in range(epoch):
            for state_batch, action_batch, next_state_batch, reward_batch in dataloader:
                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken
                state_action_values = self.policy_net(state_batch).gather(1, action_batch.long())

                # Compute V(s_{t+1}) for all next states.
                next_state_values = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)

                # Compute the expected Q values
                expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

                # Compute Huber loss
                loss = self.criterion(state_action_values, expected_state_action_values)

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
