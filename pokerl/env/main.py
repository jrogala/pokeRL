from pokerl.env.pokemonblue import PokemonBlueEnv
from pokerl.env.wrappers import ObservationDict, RewardCheckpoint, RewardHistoryToInfo


def play():
    """Play pokemon blue"""

    env = PokemonBlueEnv(interactive=True, save_state="game_start")
    env.reset()
    env.play()

def play_reward_env():
    env = PokemonBlueEnv(interactive=True, save_state="game_start")
    env = ObservationDict(env)
    env = RewardCheckpoint(env)
    env = RewardHistoryToInfo(env)
    while True:
        obs, reward, truncated, terminated, info = env.step(0)
        print(
            (
                f"Tick: {info['tick']}, ",
                f"Position: {info['position']}, ",
                f"Absolute Position: {info['absolute_position']}, ",
                f"Badges: {info['badges']}, ",
                f"Pokemon Level: {info['pokemon_level']}, ",
                f"Owned Pokemon: {info['owned_pokemon']}, ",
                f"Reward: {info['rewardHistory']}, ",
            )
        )
        print("\033[F" * 2, end="")
