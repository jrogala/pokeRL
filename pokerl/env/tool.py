import numpy as np


def game_coord_to_global_coord(x, y, map_idx):
    # src: https://github.com/PWhiddy/PokemonRedExperiments/blob/master/visualization/BetterMapVis_script_version_FLOW_edge.py
    map_offsets = {
        # https://bulbapedia.bulbagarden.net/wiki/List_of_locations_by_index_number_(Generation_I)
        0: np.array([0, 0]),  # pallet town
        1: np.array([-10, 72]),  # viridian
        2: np.array([-10, 180]),  # pewter
        12: np.array([0, 36]),  # route 1
        13: np.array([0, 144]),  # route 2
        14: np.array([30, 172]),  # Route 3
        15: np.array([80, 190]),  # Route 4
        33: np.array([-50, 64]),  # route 22
        37: np.array([-9, 2]),  # red house first
        38: np.array([-9, 25 - 32]),  # red house second
        39: np.array([9 + 12, 2]),  # blues house
        40: np.array([25 - 4, -6]),  # oaks lab
        41: np.array([30, 47]),  # Pokémon Center (Viridian City)
        42: np.array([30, 55]),  # Poké Mart (Viridian City)
        43: np.array([30, 72]),  # School (Viridian City)
        44: np.array([30, 64]),  # House 1 (Viridian City)
        47: np.array([21, 136]),  # Gate (Viridian City/Pewter City) (Route 2)
        49: np.array([21, 108]),  # Gate (Route 2)
        50: np.array([21, 108]),  # Gate (Route 2/Viridian Forest) (Route 2)
        51: np.array([-35, 137]),  # viridian forest
        52: np.array([-10, 189]),  # Pewter Museum (floor 1)
        53: np.array([-10, 198]),  # Pewter Museum (floor 2)
        54: np.array([-21, 169]),  # Pokémon Gym (Pewter City)
        55: np.array([-19, 177]),  # House with disobedient Nidoran♂ (Pewter City)
        56: np.array([-30, 163]),  # Poké Mart (Pewter City)
        57: np.array([-19, 177]),  # House with two Trainers (Pewter City)
        58: np.array([-25, 154]),  # Pokémon Center (Pewter City)
        59: np.array([83, 227]),  # Mt. Moon (Route 3 entrance)
        60: np.array([123, 227]),  # Mt. Moon
        61: np.array([152, 227]),  # Mt. Moon
        68: np.array([65, 190]),  # Pokémon Center (Route 4)
        193: None,  # Badges check gate (Route 22)
    }
    if map_idx in map_offsets:
        offset = map_offsets[map_idx]
    else:
        offset = np.array([0, 0])
        x, y = 0, 0
    coord = offset + np.array([x, y])
    return coord
