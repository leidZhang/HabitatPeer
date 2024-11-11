import warnings
import logging
import asyncio
from queue import Queue
from threading import Thread, Event

import gym
from habitat.core.agent import Agent
from habitat.core.simulator import Observations

from remote import ProviderPeer
from scene import HabitatActuator, CustomEnv


def start_habitat(agent: Agent, event: Event) -> None:
    print("Waiting for the connection...")
    event.wait() # wait until the connection established
    print("Connection established, initializing habitat env...")
    
    env: gym.Env = CustomEnv("configs/hm3d_test.yaml")
    observation: Observations = env.reset()
    agent.reset()
    
    for _ in range(500):
        action = agent.act(observation)
        observation, _, _, _ = env.step(action)
    print("Test complete, waiting for finish...")


if __name__ == "__main__":
    warnings.warn("Currently unable to send video frames to the remote peer!")
    logging.basicConfig(level=logging.INFO)
    
    connection_event: Event = Event()
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
    connection_event.set()
    
    Thread(target=start_habitat, args=(agent, connection_event)).start()
    loop.run_until_complete(provider.run())
    