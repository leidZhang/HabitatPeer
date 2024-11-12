import logging
import asyncio
from queue import Queue
from threading import Thread, Event

import numpy as np

from remote import ProviderPeer
from scene import HabitatActuator

def start_habitat(agent: HabitatActuator, event: Event) -> None:
    print("Waiting for the connection...")
    event.wait() # wait until the connection established
    print("Connection established, initializing habitat env...")

    observation = {
        "depth": np.random.rand(480, 640, 1).astype(np.float32),
        "rgb": np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
        "semantic": np.random.randint(0, 640, size=(480, 640, 1), dtype=np.int32),
        "gps": np.array([0, 0, 0]),
        "compass": np.array([0]),
    }
    agent.reset()

    for _ in range(500):
        action = agent.act(observation)
        print(f"Received agent action: {action}")
        observation = {
            "depth": np.random.rand(480, 640, 1).astype(np.float32),
            "rgb": np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
            "semantic": np.random.randint(0, 640, size=(480, 640, 1), dtype=np.int32),
            "gps": np.array([0, 0, 0]),
            "compass": np.array([0]),
        }

    print("Test complete, waiting for finish...")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)

    connection_event: Event = Event()
    ip, port, max_size = "localhost", 8765, 3
    provider: ProviderPeer = ProviderPeer(ip, port)
    agent: HabitatActuator = HabitatActuator()
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
