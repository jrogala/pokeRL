from pathlib import Path
from typing import Any

import pytesseract
from gymnasium import Env, Wrapper, spaces


class InfoReadText(Wrapper):
    """Wrapper for reading text from the game"""

    def __init__(self, env: Env, path: str):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise Exception("You should wrap your env in ObservationDict before using ObservationReadText")
        self.path = Path(path)
        pytesseract.pytesseract.tesseract_cmd = str(self.path / "tesseract.exe")

    def step(self, action) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, reward, truncated, terminated, info = self.env.step(action)
        text = pytesseract.image_to_string(observation["screen"])
        info["text"] = text
        return observation, reward, truncated, terminated, info

class InfoAddInBattleFlag(Wrapper):
    """Wrapper to add the in_battle flag to the info dict"""
    def __init__(self, env: Env):
        super().__init__(env)
        self.info_history = []
    
    def in_battle(self) -> bool:
        if self.env.helper.get_combat_turn() == 0:
            return True
        elif len(self.info_history) ==10 and all(self.info_history[-i]['ennemy_hp'] == 0 for i in range(10)):
            return False
        else:
            return self.info_history[-1]['in_battle'] if self.info_history else False
        
    def get_info(self) -> dict[str, Any]:
        info = self.env.get_info()
        info["in_battle"] = self.in_battle()
        return info
    
    def step(self, action) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, reward, truncated, terminated, info = self.env.step(action)
        self.info_history.append(info)
        if len(self.info_history) > 10:
            self.info_history.pop(0)
        info = self.get_info()
        info["in_battle"] = self.in_battle()
        return observation, reward, truncated, terminated, info
    
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        info["in_battle"] = False
        return observation, info
