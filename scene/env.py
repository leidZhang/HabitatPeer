from typing import Tuple, Any

import gym
from omegaconf import DictConfig
import habitat
from habitat.core.env import Env
from habitat.core.simulator import Observations


class CustomEnv(gym.Env):
    def __init__(self, config_paths: str) -> None:
        config_env: DictConfig = habitat.get_config(config_paths)
        self._env = Env(config_env)
        
    def reset(self) -> Observations:
        return self._env.reset()
    
    def step(self, action: dict) -> Tuple[Observations, Any, bool, dict]:
        observations: Observations = self._env.step(action)
        return observations, 0, False, {}
