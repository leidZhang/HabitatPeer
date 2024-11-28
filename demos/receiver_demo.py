import json
import time
import random
import logging
import asyncio
from queue import Queue
from threading import Thread

import cv2
from remote import ReceiverPeer
from remote.comm_utils import push_to_buffer


# Note: Only for the integration test on the local machine
def process_step_data(
    step_queue: Queue,
    action_queue: Queue,
    event: asyncio.Event,
    complete_event: asyncio.Event,
    loop: asyncio.AbstractEventLoop
) -> None:
    while not event.is_set():
        print("===============")
        step_data: dict = step_queue.get()
        if step_data["reset"]:
            print("Reset signal received, resetting...")
            continue

        print(f"Color: {step_data['depth'][0][0]}, PTS: {step_data['pts']}")
        cv2.imshow("RGB received", step_data["rgb"])
        cv2.imshow("Depth received", step_data["depth"])
        cv2.waitKey(30)
        # time.sleep(0.03)
        print("Putting action to the buffer...")
        action_queue.put({"action": random.randint(0, 5)}.copy())
        complete_event.set()

    loop.stop()
    print("Test complete, waiting for finish...")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    with open("ip_configs.json", "r") as f:
        config: dict = json.load(f)

    receiver: ReceiverPeer = ReceiverPeer(config['signaling_ip'], config['port'], config['stun_url'])
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    async_queue_names: list = ["rgb", "depth", "state", "semantic"]
    queue_names: list = ["action", "step"]

    receiver.set_loop(loop)
    for name in async_queue_names:
        receiver.set_queue(name, asyncio.Queue(config['max_size']))
    for name in queue_names:
        receiver.set_queue(name, Queue(config['max_size']))
    decision_thread: Thread = Thread(
        target=process_step_data,
        args=(
            receiver.step_queue,
            receiver.action_queue,
            receiver.done,
            receiver.action_event,
            loop
        )
    )

    try:
        loop.create_task(receiver.run())
        decision_thread.start()
        loop.run_forever()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Closing receiver and loop...")
        receiver.stop()

        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

        loop.stop()
        loop.close()
        decision_thread.join()
        print("Program finished")
