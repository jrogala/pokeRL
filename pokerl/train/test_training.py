from gymnasium import Env
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, TimeLimit

from stable_baselines3 import ppo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from pokerl.agent.tools import get_device

from pokerl.env.pokemonblue import PokemonBlueEnv
from pokerl.env.wrappers import (
    ObservationAddPokemonLevel,
    ObservationAddPosition,
    ObservationDict,
    RewardDecreasingNoChange,
    RewardDecreasingSteps,
    RewardHistoryToInfo,
    RewardIncreasingBadges,
    RewardIncreasingCapturePokemon,
    RewardIncreasingPokemonLevel,
    RewardIncreasingPositionExploration,
    ppFlattenInfo,
)


def create_env(interactive=False) -> Env:
    env = PokemonBlueEnv(interactive=interactive)
    # Setting observation
    env = ResizeObservation(env, 64)
    env = GrayScaleObservation(env)
    env = ObservationDict(env)
    env = ObservationAddPosition(env)
    env = ObservationAddPokemonLevel(env)
    # Setting reward
    env = RewardDecreasingNoChange(env, 10)
    env = RewardDecreasingSteps(env, .01)
    env = RewardIncreasingBadges(env, 100)
    env = RewardIncreasingCapturePokemon(env, 10)
    env = RewardIncreasingPokemonLevel(env, 10)
    # env = RewardIncreasingPositionExploration(env, 1)
    env = RewardHistoryToInfo(env)
    # Post processing
    env = TimeLimit(env, 10000)
    # env = ppFlattenInfo(env)
    return env

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = create_env()
        env.reset(seed=(seed + rank))
        return env
    return _init


if __name__ == '__main__':
    subproc_envs = SubprocVecEnv([make_env(i) for i in range(16)])

    model = ppo.PPO(
        "MultiInputPolicy",
        subproc_envs,
        # device=get_device(),
        verbose=1
        )
    model.learn(total_timesteps=50000,
          progress_bar=True,
        #   callback=WandbCallback(),
          )
    # model.save('ppo_pokemon_blue_v1_1500000')
    subproc_envs.close()
