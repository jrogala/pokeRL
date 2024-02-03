import numpy as np
from collections import Dict

def basic_reward(state: Dict, next_state: Dict) -> float:
    """
    A basic reward function.
    """
    cum_reward = -1
    #decrease reward if the new state is the same as the previous state
    if state == next_state:
        cum_reward -= 10
    
    if state['level_pokemon'] < next_state['level_pokemon']:
        cum_reward += 1
        if np.max(next_state['level_pokemon']) > 2 * np.median(next_state['level_pokemon']):
            cum_reward += -1

    if state['badges'] != next_state['badges']:
        cum_reward += 100

    return cum_reward


"""
Boite a idée :

Positive rewards unconditionaly given when:
- Getting a badge
- Leveling up a pokemon
- Catching a pokemon
- Winning a battle
- Evolving a pokemon

Positive rewards when first time:
- Discovering a new area/pokemon

Negative rewards unconditionaly given when:
- Losing a battle
- Useless action (same state as previous state)
- Pokemon fainting
- Pokemon hit by poison or burn


Combat rewards:
Positive rewards when:
- Landing a hit
- Landing a super effective hit
- Ennemy fainting
- Ennemy status effect (poison, burn, sleep, etc)

Negative rewards when:
- Missing a hit
- New status effect (poison, burn, sleep, etc)
- Trying to use a move that is not available
- Trying to flee a battle with a dresser
- Using a move that is not effective
- Using an object that is not available


"""         

