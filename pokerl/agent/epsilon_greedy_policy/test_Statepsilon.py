import torch
from pokerl.agent.epsilon_greedy_policy.Statepsilon import StateEpsilon
import torch
import random
from pokerl.tools import get_device

def test_state_epsilon():
    num_samples = 10
    guessed_state = torch.randint(0, 10, size=((100,)))
    states = [torch.randint(0, 10, size=((100,))) for _ in range((num_samples))]
    actions = [random.randint(0, 1) for _ in range(num_samples)]
    next_states = [torch.randint(0, 10, size=((100,))) for _ in range((num_samples))]

    # Initialize StateEpsilon object
    state_epsilon = StateEpsilon(guessed_state=guessed_state)

    # Test estimate_next_state method
    next_state_estimation = state_epsilon.estimate_next_state(states[0], actions[0])
    print(next_state_estimation)
    # Test train method
    state_epsilon.train(states, actions, next_states)
    print("test")

    # Test get_epsilon method
    current_state = states[0]
    epsilon = state_epsilon.get_epsilon(current_state)
    assert isinstance(epsilon, float)
    print(epsilon)


def test_training():
    num_samples = 512*10
    guessed_state = torch.randint(0, 1, size=((100,)))
    states = [torch.randint(0, 1, size=((100,))) for _ in range((num_samples))]
    actions = [random.randint(0, 1) for _ in range(num_samples)]
    next_states = [torch.randint(0, 1, size=((100,))) for _ in range((num_samples))]

    # Initialize StateEpsilon object
    state_epsilon = StateEpsilon(guessed_state=guessed_state, device=get_device(), batch_size=512)

    for i in range(10):
        state_epsilon.estimate_next_state(states[0], actions[0])
        print(i, state_epsilon.get_epsilon(states[0]*-1))
        state_epsilon.train(states, actions, next_states, epoch=10)
        

if __name__ == "__main__":
    test_training()