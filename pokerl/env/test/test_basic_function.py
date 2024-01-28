import pytest
from pyboy import WindowEvent

from pokerl.env.pokemonblue import PokemonBlueEnv
from pokerl.env.pyboygym import PyBoyGym
from pokerl.env.rewards.basic_rewards import WrapperLevel
from pokerl.env.settings import Pokesettings


@pytest.fixture
def pyboy_gym():
    return PyBoyGym(rom_name=Pokesettings.rom_name, interactive=False)

def test_start_stop(pyboy_gym: PyBoyGym):
    """Test start and stop of PyBoyGym env"""
    pyboy_gym.start_game()
    pyboy_gym.tick()
    pyboy_gym.close()

def test_pressing_a_button(pyboy_gym: PyBoyGym):
    """Test pressing a button of PyBoyGym env"""
    pyboy_gym.start_game()
    pyboy_gym.send_input(WindowEvent.PRESS_BUTTON_A)
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
    env = WrapperLevel(pokemon_blue)
    env.reset()
    env.step(0)
    env.step(0)
    env.reset()

