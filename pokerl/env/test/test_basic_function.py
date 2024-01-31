import pytest
from pyboy import WindowEvent

from pokerl.env.pokemonblue import PokemonBlueEnv
from pokerl.env.pyboygym import PyBoyGym
from pokerl.env.settings import Pokesettings
from pokerl.env.wrapper.basic_rewards import PositionObservation
from gymnasium.wrappers import FlattenObservation, ResizeObservation



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
    env = PositionObservation(pokemon_blue)
    env.reset()
    env.step(0)
    env.step(0)
    env.reset()

def test_multiple_reward(pokemon_blue: PokemonBlueEnv):
    env = ResizeObservation(pokemon_blue, shape=(64, 64, 3))
    # env = PositionObservation(env)
    # env = FlattenObservation(env)
