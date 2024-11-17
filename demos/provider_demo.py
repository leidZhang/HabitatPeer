import time
import json
import logging
import asyncio
from queue import Queue
from datetime import datetime
from threading import Thread, Event

import cv2
import numpy as np

from remote import ProviderPeer
from remote.comm_utils import empty_queue
from scene import HabitatActuator


# TODO: Replace it with real habitat env
def start_habitat(
    agent: HabitatActuator,
    provider_event: asyncio.Event,
    width: int = 640,
    height: int = 480
) -> None:
    print("initializing habitat env...")

    observation = {
        "depth": np.random.rand(height, width, 1).astype(np.float32),
        "rgb": np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8),
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
        print(f"Got {action} and send {(i + 1) % 256} to the peer...")
        observation = {
            "depth": np.random.rand(height, width, 1).astype(np.float32),
            "rgb": np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8),
            "semantic": np.random.randint(0, 640, size=(height, width, 1), dtype=np.int32),
            "gps": np.array([0, 0, 0]),
            "compass": np.array([(i + 1) % 256])
        }

        i += 1
        i = 0 if i == 1000 else i # Temporary fix for testing

    if not provider_event.is_set():
        provider_event.set() # signal provider to stop
    # loop.stop()
    print("Test complete, waiting for finish...")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    with open("ip_configs.json", "r") as f:
        config: dict = json.load(f)

    provider: ProviderPeer = ProviderPeer(config['signaling_ip'], config['port'], config['stun_url'])
    agent: HabitatActuator = HabitatActuator()
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    queue_names: list = ["depth", "rgb", "semantic", "state", "action"]
    queue_list: list = []

    provider.set_loop(loop)
    for name in queue_names:
        named_queue: Queue = Queue(config['max_size'])
        provider.set_queue(name, named_queue)
        agent.set_queue(name, named_queue)
        queue_list.append(named_queue)
    actuator_thread: Thread = Thread(
        target=start_habitat,
        args=(agent, provider.done, config['width'], config['height'])
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
