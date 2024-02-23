import numpy as np
import pytest

from pokerl.env import PokemonBlueEnv
from pokerl.env.settings import Pokesettings
from pokerl.env.wrappers import (
    ObservationAddPokemonLevel,
    ObservationAddPosition,
    ObservationDict,
    RewardDecreasingNoChange,
    RewardIncreasingPositionExploration,
)


@pytest.fixture
def real_env():
    return PokemonBlueEnv(Pokesettings.rom_name)

def test_ObservationDict(real_env):
    env = ObservationDict(real_env)
    obs, _ = env.reset()
    assert "screen" in obs
    obs, _, _, _, _ = env.step(0)
    assert "screen" in obs

def test_ObservationAddPokemonLevel(real_env):
    env = ObservationDict(real_env)
    env = ObservationAddPokemonLevel(env)
    obs, _ = env.reset()
    assert "pokemon_level" in obs
    assert obs["pokemon_level"].shape == (6,)
    assert (obs["pokemon_level"] == np.array([0, 0, 0, 0, 0, 0])).all()
    obs, _, _, _, _ = env.step(0)
    assert "pokemon_level" in obs
    assert obs["pokemon_level"].shape == (6,)
    assert (obs["pokemon_level"] == np.array([0, 0, 0, 0, 0, 0])).all()

def test_ObservationAddPosition(real_env):
    env = ObservationDict(real_env)
    env = ObservationAddPosition(env)
    obs, _ = env.reset()
    assert "position" in obs
    assert obs["position"].shape == (2,)
    assert (obs["position"] == np.array([0, 0])).all()
    obs, _, _, _, _ = env.step(0)
    assert "position" in obs
    assert obs["position"].shape == (2,)
    assert (obs["position"] == np.array([0, 0])).all()

def test_RewardIncreasingPositionExploration(real_env):
    env = RewardIncreasingPositionExploration(real_env)
    env.reset()
    _, reward, _, _, _ = env.step(0)
    assert int(reward) == 1 # 1st step is always a new position

def test_RewardDecreasingNoChange(real_env):
    env = RewardDecreasingNoChange(real_env, 1)
    env.reset()
    _, reward, _, _, info1 = env.step(0)
    assert int(reward) == 0 # 1st step has no negative reward
    _, reward, _, _, info2 = env.step(0)
    assert int(reward) == -1 # 2nd step has a negative reward