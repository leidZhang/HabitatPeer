# Implement the habitat simulation using the Habitat-lab API.
import json
import asyncio
from typing import List, Dict, Any
from multiprocessing import Process, Queue

import cv2
import gym
import numpy as np

from omegaconf import DictConfig
import habitat
from habitat.core.env import Env
from habitat.core.agent import Agent
from habitat.core.simulator import Observations

from scene.env import CustomEnv
from scene.agent import HabitatActuator
from remote import ProviderPeer


def start_habitat_env(queue_list: List[Queue]) -> None:
    agent: HabitatActuator = HabitatActuator()
    for queue in queue_list:
        agent.set_queue(queue)
    env: gym.Env = CustomEnv("configs/hm3d_test.yaml")

    observation: Observations = env.reset()
    for _ in range(500):
        action: Dict[str, Any] = agent.act(observation)
        observation, reward, done, info = env.step(action)
        if done:
            break
    print("Episode completed, closing environment...")


def start_aiortc_client(queue_list: List[Queue]) -> None:
    with open("ip_configs.json", "r") as f:
        config: dict = json.load(f)

    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    provider: ProviderPeer = ProviderPeer(
        config['signaling_ip'], config['port'], config['stun_url']
    )

    for queue in queue_list:
        provider.set_queue(queue)
    loop.run_until_complete(provider.run())


if __name__ == "__main__":
    queue_list: List[Queue] = [Queue(maxsize=1) for _ in range(5)]
    habitat_process: Process = Process(target=start_habitat_env, args=(queue_list,))
    aiortc_process: Process = Process(target=start_aiortc_client, args=(queue_list,))
    aiortc_process.start() # start the webRTC client
    habitat_process.start() # start the habitat environment
