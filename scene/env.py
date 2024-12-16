import os
import json
from typing import (
    Tuple,
    Any,
    List,
    Optional,
    Dict
)
from datetime import datetime
from collections import defaultdict

import gym
from tqdm import tqdm
from omegaconf import DictConfig
import habitat
from habitat.core.agent import Agent
from habitat.core.env import Env
from habitat.core.challenge import Challenge
from habitat.core.simulator import Observations

from .actions import TurnDoubleActionConfig

class CustomEnv(gym.Env):
    def __init__(self, config_paths: str) -> None:
        config_env: DictConfig = habitat.get_config(config_paths)
        self._env = Env(config_env)

    def reset(self) -> Observations:
        return self._env.reset()

    def step(self, action: dict) -> Tuple[Observations, Any, bool, dict]:
        observations: Observations = self._env.step(action)
        return observations, 0, False, {}

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        return self._env._sim.set_agent_state(
            position, rotation, agent_id, reset_sensors
        )


class CustomedChallenge(Challenge):
    def __init__(
        self, eval_remote: bool = False
    ) -> None: # Copied from the Benchmark class
        self.metrics: List[Dict[str, float]] = None
        config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
        config_env = habitat.get_config(config_paths)
        self.__config_custom_action(config_env) # Additional line compaired to the Benchmark class
        self._eval_remote = eval_remote

        if self._eval_remote is True:
            self._env = None
        else:
            self._env = Env(config=config_env)

    # NOTE: The only way I know to add new action in the habitat.Challenge class
    def __config_custom_action(self, config: DictConfig) -> None:
        with habitat.config.read_write(config):
            config.habitat['task'].actions["turn_right_2"] = TurnDoubleActionConfig(
                type="TurnRightDouble",
                turn_angle_deg=config.habitat.simulator.turn_angle,
                noise_amount=0.0
            )
        print("Custom action configured.")

    def local_evaluate(
        self, agent: Agent, num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            self.metrics = []
            observations = self._env.reset()
            agent.reset()

            while not self._env.episode_over:
                action = agent.act(observations)
                observations = self._env.step(action)
                metrics = self._env.get_metrics()
                self.metrics.append(metrics)
                print("Metrics: ", metrics)

            metrics = self._env.get_metrics()
            print("Metrics: ", metrics)
            self.metrics.append(metrics)

            with open(f"episode_{count_episodes}_{datetime.now()}.json", "w") as f:
                json.dump(self.metrics, f)
            print(f"Metrics for episode {count_episodes} saved.")

            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v
            count_episodes += 1
            pbar.update(1)

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics
