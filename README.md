# PokeRL - A Pokemon RL Environment

## Using PokeRL

### Install PokeRL

```bash
git clone git@github.com:jrogala/pokeRL.git
cd pokeRL
poetry install
poetry shell
```

### Install ROMs

```bash
rom_handler --download --extract
```

This will download and extract the pokemon blue ROMs into the `pokeRL/roms` directory.

if url has changed, --url can be used to specify a new url. 

### Run Pokemon blue with the simulator

```bash
play
```

## Ram Addresses

[Ram link](https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map#Saved_data_(SRAM))

## Idea / TODO

Idea of rewards:

- Level of all pokemon over time.
- Badge count over time.
- Percent of life of every pokemon. (to see if AI has the same idea of us to heal pokemon everytime)
- Amount of money.
- Number of pokemon caught.
- Number of pokemon seen.
- Exploratory actions. (distance to previous step)

Idea of rewards to test:

- Leveling every pokemon vs leveling only one pokemon. (mean vs max vs L2 norm)
- Leveling pokemon vs catching pokemon. (or seeing)
- Over time constraints. vs not.

Idea of visualisation:

- Pokemon level over time
- X,Y coords over save (color: level of pokemon or any rewards metric)
- Result of battle against fixed trainer. (is it possible to emulate a battle in the emulator or using external tool like [this](https://pypi.org/project/poke-battle-sim/))
