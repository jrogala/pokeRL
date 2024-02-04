from typing import Dict

import numpy as np


def basic_reward(current_state: Dict, next_state: Dict) -> float:
    """
    A basic reward function.
    """
    if current_state is None:
        return 0  # no reward for the first state
    reward = -1
    # decrease reward if the new state is the same as the previous state
    if (current_state["observation"].all() == next_state["observation"].all()) and (
        current_state["info"] == next_state["info"]
    ):
        reward -= 10

    if current_state["info"]["level_pokemon"] < next_state["info"]["level_pokemon"]:
        reward += 1
        if np.max(next_state["info"]["level_pokemon"]) > 2 * np.median(next_state["info"]["level_pokemon"]):
            reward += -1

    if sum(current_state["info"]["owned_pokemon"]) < sum(next_state["info"]["owned_pokemon"]):
        reward += 10

    if current_state["info"]["badges"] != next_state["info"]["badges"]:
        reward += 100

    return reward


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
