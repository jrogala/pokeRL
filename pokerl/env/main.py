import argparse
from pathlib import Path

from pokerl.env import PokemonBlueEnv
from pokerl.env.wrappers import ObservationDict, RewardCheckpoint, RewardHistoryToInfo


def play(save_state="game_start", hasInfo=False, hasHistory=False):
    """Play pokemon blue"""

    env = PokemonBlueEnv(interactive=True, save_state=save_state)
    if hasInfo:
        env = ObservationDict(env)
        if hasHistory:
            env = RewardHistoryToInfo(env)
    env.reset()
    env.play()


def pprint_info(info):
    return "\n".join([f"{key}: {value}" for key, value in info.items()])


def main():
    parser = argparse.ArgumentParser(description="Play Pokemon Blue")
    parser.add_argument(
        "--save_state",
        type=str,
        default="game_start",
        help="Save state to load",
    )
    parser.add_argument(
        "--list_state",
        action="store_true",
        help="List all save states",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print info",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print history, must be used with --info",
    )
    args = parser.parse_args()
    if args.list_state:
        print("\n".join((x.name for x in Path("states").glob("*.state"))))
        return 0
    else:
        play(args.save_state, args.info, args.history)


if __name__ == "__main__":
    main()
