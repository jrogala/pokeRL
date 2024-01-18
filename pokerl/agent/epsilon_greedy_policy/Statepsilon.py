from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn, optim, tensor
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Epsilon:
    epsilon: float = field(init=True, repr=True, default=0.5)
    epsilon_decay: float = field(init=True, repr=True, default=0.99)
    epsilon_min: float = field(init=True, repr=True, default=0.01)

    def get_epsilon(self, episode: int) -> float:
        """
        Returns the epsilon value for the given episode.
        """
        return max(self.epsilon_min, self.epsilon * (self.epsilon_decay**episode))


@dataclass
class StateEpsilon:
    """
    Represents a state with epsilon-greedy policy for reinforcement learning.
    """

    guessed_state: torch.Tensor = field(init=True, repr=False)

    max_epsilon: float = field(init=True, repr=True, default=1)
    loss: F = field(init=True, repr=True, default=lambda x, y: F.l1_loss(x, y, size_average=True))
    device: torch.device = field(init=True, repr=True, default=torch.device("cpu"))
    batch_size: int = field(init=True, repr=True, default=32)

    model: torch.nn.Module = field(init=False, repr=True, default=None)
    optimizer: torch.optim.Optimizer = field(init=False, repr=True, default=None)

    previous_epsilon: deque = field(init=False, repr=True, default=deque(maxlen=10))

    def __post_init__(self):
        """
        Initializes the StateEpsilon object.
        Sets the epsilon value to the base_epsilon.
        Creates the neural network model and optimizer.
        """
        self.guessed_state = self.guessed_state.float()
        self.model = nn.Sequential(
            nn.Linear(self.guessed_state.shape[0] + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.guessed_state.shape[0]),
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.previous_epsilon.append(1)

    def estimate_next_state(self, state: torch.Tensor, action: int):
        """
        Estimates the next state given the current state and action.
        Updates the guessed_state attribute with the estimated next state.
        Returns the estimated next state.
        """
        action = tensor([action]).unsqueeze(0).float()
        state = state.unsqueeze(0).float()
        with torch.no_grad():
            state_action = torch.cat((state, action), dim=1).squeeze(0).to(self.device)
            next_state_estimation = self.model(state_action)
        self.guessed_state = next_state_estimation
        return next_state_estimation

    def train(self, states: list[torch.Tensor], actions: list[int], next_states: list[torch.Tensor], epoch: int = 1):
        """
        Trains the model using the given states, actions, and next states.
        """
        states_tensor = torch.stack(states).float()
        actions_tensor = torch.tensor(actions).unsqueeze(1).float()
        next_states_tensor = torch.stack(next_states).float()
        dataset = TensorDataset(states_tensor, actions_tensor, next_states_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(epoch):
            for state_batch, action_batch, next_state_batch in dataloader:
                self.optimizer.zero_grad()
                state_action_batch = torch.cat((state_batch, action_batch), dim=1).to(self.device)
                next_state_batch = next_state_batch.to(self.device)
                next_state_estimation = self.model(state_action_batch)
                loss = self.loss(next_state_estimation, next_state_batch)
                loss.backward()
                self.optimizer.step()

    def get_epsilon(self, current_state: torch.Tensor) -> float:
        """
        Calculates and returns the epsilon value based on the error between the current state and the guessed state.
        """
        error = self.loss(current_state.to(self.device), self.guessed_state.to(self.device)).item()
        self.previous_epsilon.append(error)
        mean_error = sum(self.previous_epsilon) / len(self.previous_epsilon)
        return min(mean_error, self.max_epsilon)
