import torch
from rl_model.PPO.PPO import PPO
from pokerl.blueterface import PokemonBlueGym
from rl_model.tools import get_device

# Hyperparameters
num_episodes = 1000
max_timesteps = 1000


# Initialize environment and PPO agent
env = PokemonBlueGym()
device = get_device()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
ppo = PPO(state_dim, action_dim).to(device)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    for t in range(max_timesteps):
        state = torch.FloatTensor(state).to(device)
        action_int, _, _, _ = ppo.get_action_and_value(state)
        action = env.action_space[action_int.item()]
        next_state, reward, done, _ = env.step(action.item())
        env.render()
        if done:
            break
        state = next_state
    reward = env.get_reward()
    ppo.update
env.close()