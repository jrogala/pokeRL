from gymnasium import Env, Wrapper, spaces

from pokerl.env.pokemonblue import GameboyAction


class RemoveSelectStartAction(Wrapper):
    """Wrapper to stop the game when a pokemon is encountered"""

    def __init__(self, env: Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(self.env.action_space.n - 2)
        self.action_space_convertissor = [
            button
            for button in self.env.unwrapped.action_space_convertissor
            if button not in [GameboyAction.SELECT, GameboyAction.START]
        ]


class RemoveABAction(Wrapper):
    """Wrapper to stop the game when a pokemon is encountered"""

    def __init__(self, env: Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(self.env.action_space.n - 2)
        self.action_space_convertissor = [
            button
            for button in self.env.unwrapped.action_space_convertissor
            if button not in [GameboyAction.A, GameboyAction.B]
        ]
