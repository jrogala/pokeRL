from pathlib import Path
from typing import Any

import numpy as np
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
