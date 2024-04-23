from gymnasium import Env
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, TimeLimit
from stable_baselines3 import ppo
from stable_baselines3.common.vec_env import SubprocVecEnv

import wandb
from pokerl.env.pokemonblue import PokemonBlueEnv
from pokerl.env.wrappers import (
    ObservationAddPokemonLevel,
    ObservationAddPosition,
    ObservationDict,
    RemoveSelectStartAction,
    RewardDecreasingLostBattle,
    RewardDecreasingNoChange,
    RewardDecreasingSteps,
    RewardHistoryToInfo,
    RewardIncreasingBadges,
    RewardIncreasingCapturePokemon,
    RewardIncreasingLandedAttack,
    RewardIncreasingPokemonLevel,
)
from wandb.integration.sb3 import WandbCallback


def create_env(interactive=False) -> Env:
    env = PokemonBlueEnv(interactive=interactive)
    # Setting observation
    env = ResizeObservation(env, 64)
    env = GrayScaleObservation(env)
    env = ObservationDict(env)
    env = ObservationAddPosition(env)
    env = ObservationAddPokemonLevel(env)
    env = RemoveSelectStartAction(env)
    # Setting reward
    env = RewardDecreasingNoChange(env, 0.01)
    env = RewardDecreasingSteps(env, .01)
    env = RewardIncreasingBadges(env, 100)
    env = RewardIncreasingCapturePokemon(env, 10)
    env = RewardIncreasingPokemonLevel(env, 10)
    env = RewardIncreasingLandedAttack(env, 0.05)
    env = RewardDecreasingLostBattle(env, 1)
    # env = RewardIncreasingPositionExploration(env, 1)
    env = RewardHistoryToInfo(env)
    # Post processing
    # env = TimeLimit(env, 300)
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


if __name__ == "__main__":
    NB_CPUS = 12
    EP_LENGTH = 2**14 # 16384
    NB_UPDATES = 2**8 # 256
    TOTAL_TIMESTEPS = EP_LENGTH*NB_CPUS*NB_UPDATES
    config = {
        "nb_cpus" : NB_CPUS,
        "ep_length" : EP_LENGTH,
        "policy_type": "MultiInputPolicy",
        "total_timesteps": TOTAL_TIMESTEPS,
        "env_name": "PokemonBlueEnv",
    }
    run = wandb.init(
        project="pokemonblue-rl",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    subproc = SubprocVecEnv([make_env(i) for i in range(NB_CPUS)])

    model = ppo.PPO(
        "MultiInputPolicy",
        subproc,
        learning_rate=0.001,
        n_steps=int(EP_LENGTH//NB_CPUS),
        batch_size=512,
        n_epochs=4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=2,
        tensorboard_log=f"runs/{run.id}",
    )
    model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    subproc.close()
    run.finish()
