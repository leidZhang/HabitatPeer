import faulthandler
faulthandler.enable()


import random
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import gym
import numpy as np

from omegaconf import DictConfig
import habitat
from habitat.core.env import Env
from habitat.core.agent import Agent
from habitat.core.simulator import Observations

from scene.env import CustomEnv


class MockAgent(Agent):    
    def reset(self) -> None:
        pass
        
    def act(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": random.randint(1, 5)}
    
    
def visualize_cam(observations: Observations, type: str = 'rgb') -> None:
    image: np.ndarray = observations[type]
    cv2.imshow(f"Sensor {type}", image)
    return 

    
def visualize_semantic(observations: Observations, fig, ax) -> None:
    semantic_data: np.ndarray = observations['semantic']
    ax.clear()
    ax.imshow(semantic_data)
    ax.set_title("Semantic Segmentation")
    plt.draw()
    plt.pause(0.001)
        
    
if __name__ == "__main__":
    test_data: list = []
    
    agent: Agent = MockAgent()
    env: gym.Env = CustomEnv("configs/hm3d_test.yaml")
    plt.ion()
    fig, ax = plt.subplots()
    
    observation: Observations = env.reset()
    visualize_semantic(observation, fig, ax)
    test_data.append(observation)
    keys = observation.keys()
    agent.reset()

    for i in range(10):
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        visualize_semantic(observation, fig, ax)
        print(observation['semantic'].dtype)
        observation_dict = {key: observation[key] for key in keys}
        print(observation_dict.keys())
        print(type(observation_dict))
        test_data.append(observation_dict)
        print(type(test_data[0]))
    print("Test ran successfully")
    
    npz_data: dict = {key: [] for key in keys}
    for data in test_data:
        for key in keys:
            npz_data[key].append(data[key])
    
    print(npz_data.keys())
    np.savez("test_data.npz", **npz_data)
    
    npz_data = np.load("test_data.npz", allow_pickle=True)
    for key in keys:
        print(type(npz_data[key]))
        print(type(npz_data[key][0]))
