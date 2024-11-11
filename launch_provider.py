import asyncio
from queue import Queue
from threading import Thread

import gym
from habitat.core.agent import Agent
from habitat.core.simulator import Observations

from remote import ProviderPeer
from scene import HabitatActuator, CustomEnv


def start_habitat(agent: Agent) -> None:
    env: gym.Env = CustomEnv("configs/hm3d_test.yaml")
    observation: Observations = env.reset()
    agent.reset()
    
    for _ in range(500):
        action = agent.act(observation)
        observation, _, _, _ = env.step(action)
    print("Test complete, waiting for finish...")


if __name__ == "__main__":
    ip, port, max_size = "localhost", 8765, 3
    provider: ProviderPeer = ProviderPeer(ip, port)
    agent: Agent = HabitatActuator()
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    queue_names: list = ["depth", "rgb", "semantic", "state", "action"]
    
    provider.set_loop(loop)
    for name in queue_names:
        named_queue: Queue = Queue(max_size)
        provider.set_queue(name, named_queue)
        agent.set_queue(name, named_queue)
    
    Thread(target=start_habitat, args=(agent, )).start()
    loop.run_until_complete(provider.run())
    