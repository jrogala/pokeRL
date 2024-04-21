import pytest

from pokerl.env.pokemonblue import PokemonBlueEnv
from pokerl.env.pyboygym import PyBoyGym
from pokerl.env.settings import Pokesettings
from pokerl.env.wrappers import (
    RewardDecreasingNoChange,
    RewardDecreasingSteps,
    RewardIncreasingBadges,
    RewardIncreasingCapturePokemon,
    RewardIncreasingPokemonLevel,
    RewardIncreasingPositionExploration,
)


@pytest.fixture
def pyboy_gym():
    return PyBoyGym(rom_name=Pokesettings.rom_name, interactive=False)


@pytest.fixture
def pokemon_blue():
    """Return a PokemonBlueEnv instance"""
    return PokemonBlueEnv(rom_name=Pokesettings.rom_name, interactive=False)


def test_basic_reward_baptiste(pokemon_blue: PokemonBlueEnv):
    env = RewardDecreasingSteps(pokemon_blue, 1)  # 12
    env = RewardDecreasingNoChange(env, 10)  # 14 - 17
    env = RewardIncreasingPokemonLevel(env, 1)  # 19 - 22
    env = RewardIncreasingCapturePokemon(env, 10)  # 24 - 25
    env = RewardIncreasingBadges(env, 100)  # 27 - 28
    env.reset()
    _, reward, _, _, _ = env.step(0)
    assert int(reward) == -1
    _, reward, _, _, _ = env.step(0)
    assert int(reward) == -11


def test_reward_observation(pokemon_blue: PokemonBlueEnv):
    env = RewardIncreasingPositionExploration(pokemon_blue, 1)
    env.reset()
    _, reward, _, _, _ = env.step(0)
    assert int(reward) == 1
