import logging
import asyncio
from queue import Queue
from threading import Thread, Event

import numpy as np

from remote import ProviderPeer
from remote.comm_utils import empty_queue
from scene import HabitatActuator


def start_habitat(agent: HabitatActuator, provider_event: asyncio.Event, loop: asyncio.AbstractEventLoop) -> None:
    width, height = 640, 480

    print("initializing habitat env...")

    observation = {
        "depth": np.random.rand(height, width, 1).astype(np.float32),
        # "rgb": np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8),
        "rgb": np.zeros((height, width, 3), dtype=np.uint8),
        "semantic": np.random.randint(0, 640, size=(height, width, 1), dtype=np.int32),
        "gps": np.array([0, 0, 0]),
        "compass": np.array([0]),
    }
    agent.reset()

    i = 0
    while i < 1000:
        # make sure not stuck here forever
        if provider_event.is_set():
            print("Detected provider stop event, exiting...")
            break

        action = agent.act(observation)
        # print(f"Received agent action: {action}")
        observation = {
            "depth": np.random.rand(height, width, 1).astype(np.float32),
            # "rgb": np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8),
            "rgb": np.ones((height, width, 3), dtype=np.uint8) * (i % 256),
            "semantic": np.random.randint(0, 640, size=(height, width, 1), dtype=np.int32),
            "gps": np.array([0, 0, 0]),
            "compass": np.array([0]),
        }

    if not provider_event.is_set():
        provider_event.set() # signal provider to stop
    # loop.stop()
    print("Test complete, waiting for finish...")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)

    ip, port, max_size = "localhost", 8765, 3
    provider: ProviderPeer = ProviderPeer(ip, port)
    agent: HabitatActuator = HabitatActuator()
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    queue_names: list = ["depth", "rgb", "semantic", "state", "action"]
    queue_list: list = []

    provider.set_loop(loop)
    for name in queue_names:
        named_queue: Queue = Queue(max_size)
        provider.set_queue(name, named_queue)
        agent.set_queue(name, named_queue)
        queue_list.append(named_queue)
    actuator_thread: Thread = Thread(
        target=start_habitat,
        args=(agent, provider.done, loop)
    )

    try:
        actuator_thread.start()
        loop.run_until_complete(provider.run())
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Clossing provider and loop...")
        provider.stop()

        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

        loop.stop()
        loop.close()
        actuator_thread.join()
        print("Program finished")
