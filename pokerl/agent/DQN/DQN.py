import functools
import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch import nn


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

    def forward(self, x):
        x = self.model(x)
        return x


@dataclass
class DQNAgent:
    env: gym.Env

    BATCH_SIZE: int = 128
    GAMMA: float = 0.99
    EPS_START: float = 1.0
    EPS_END: float = 0.01
    EPS_DECAY: float = 0.995
    TARGET_UPDATE: int = 10
    EPSILON: float = 0.1

    def __post_init__(self):
        observation_shape = functools.reduce(lambda x, y: x * y, self.env.observation_space.shape)
        action_dim = self.env.action_space.n
        self.memory = deque(maxlen=10000)

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
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()
        self.memory.append((state, action))
        return action

    def train(self):
        # Don't optimize the model if we don't yet have enough memory for a batch
        if len(self.memory) < self.BATCH_SIZE:
            return

        # Sample a batch of transitions from the memory
        transitions = random.sample(self.memory, self.BATCH_SIZE)

        # Transpose the batch to get separate arrays for states, actions, etc.
        batch = zip(*transitions)

        # Convert the batch arrays into PyTorch tensors
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = [
            torch.tensor(x, dtype=torch.float32) for x in batch
        ]
        # Debug: Print the shape of an original state to understand its structure
        print("Original state shape:", transitions[0][0].shape)

        # Flatten the batch_state
        batch_state = batch_state.view(self.BATCH_SIZE, -1)

        # Debug: Print the flattened state shape
        print("Flattened state shape:", batch_state.shape)
        batch_action = batch_action.type(torch.int64).unsqueeze(1)
        batch_done = batch_done.unsqueeze(1)

        print(batch_state.shape)
        print(transitions)

        # Get the current Q-values for the batch states and actions
        current_q_values = self.policy_net(batch_state).gather(1, batch_action)

        # Get the max predicted Q-values from the target network for next states
        max_next_q_values = self.target_net(batch_next_state).detach().max(1)[0].unsqueeze(1)

        # Calculate the expected Q-values
        expected_q_values = batch_reward + (self.GAMMA * max_next_q_values * (1 - batch_done))

        # Compute loss between current Q-values and expected Q-values
        loss = self.criterion(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
