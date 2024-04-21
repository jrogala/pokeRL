import pytest
from gymnasium.wrappers import FlattenObservation, ResizeObservation

from pokerl.env.pokemonblue import PokemonBlueEnv
from pokerl.env.pyboygym import PyBoyGym
from pokerl.env.settings import Pokesettings
from pokerl.env.wrappers import ObservationAddPokemonLevel, ObservationAddPosition, ObservationDict, ppFlattenInfo


@pytest.fixture
def pyboy_gym():
    return PyBoyGym(rom_name=Pokesettings.rom_name, interactive=False)


def test_start_stop(pyboy_gym: PyBoyGym):
    """Test start and stop of PyBoyGym env"""
    pyboy_gym.reset()
    pyboy_gym.tick()
    pyboy_gym.close()


def test_pressing_a_button(pyboy_gym: PyBoyGym):
    """Test pressing a button of PyBoyGym env"""
    pyboy_gym.reset()
    pyboy_gym.step(0)
    pyboy_gym.tick()
    pyboy_gym.close()


@pytest.fixture
def pokemon_blue():
    """Return a PokemonBlueEnv instance"""
    return PokemonBlueEnv(rom_name=Pokesettings.rom_name, interactive=False)


def test_poke_gym_env(pokemon_blue: PokemonBlueEnv):
    pokemon_blue.reset()
    for i in range(9):
        pokemon_blue.step(i)
    pokemon_blue.reset()


def test_add_reward(pokemon_blue: PokemonBlueEnv):
    env_dict = ObservationDict(pokemon_blue)
    env = ObservationAddPosition(env_dict)
    env.reset()
    env.step(0)
    env.step(0)
    env.reset()


def test_resize(pokemon_blue: PokemonBlueEnv):
    env_resized = ResizeObservation(pokemon_blue, shape=(64, 64))
    env_dict = ObservationDict(env_resized)
    env_position = ObservationAddPosition(env_dict)
    env_flat = FlattenObservation(env_position)
    env_flat.reset()
    env_flat.step(0)


def test_double_obs(pokemon_blue: PokemonBlueEnv):
    env_resized = ResizeObservation(pokemon_blue, shape=(64, 64))
    env_dict = ObservationDict(env_resized)
    env_position = ObservationAddPosition(env_dict)
    env_pokemon = ObservationAddPokemonLevel(env_position)
    env_pokemon.reset()
    env_pokemon.step(0)


def test_double_obs_flattened(pokemon_blue: PokemonBlueEnv):
    env_resized = ResizeObservation(pokemon_blue, shape=(64, 64))
    env_dict = ObservationDict(env_resized)
    env_position = ObservationAddPosition(env_dict)
    env_pokemon = ObservationAddPokemonLevel(env_position)
    env_flattened = ppFlattenInfo(env_pokemon)
    env_flattened.reset()
    env_flattened.step(0)
