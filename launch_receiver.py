import logging
import asyncio
from queue import Queue
from threading import Thread
from typing import List, Callable

import cv2
from remote import ReceiverPeer


# Note: Only for the integration test on the local machine
def process_step_data(step_queue: Queue, action_queue: Queue) -> None:
    while True:
        print("Processing the data...")
        step_data: dict = step_queue.get()
        
        cv2.imshow("RGB received", step_data["rgb"])
        cv2.imshow("Depth received", step_data["depth"])
        cv2.waitKey(1)
        
        action_queue.put({"action": 0})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    ip, port, max_size = "localhost", 8765, 3
    receiver: ReceiverPeer = ReceiverPeer(ip, port)
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    async_queue_names: list = ["rgb", "depth", "semantic", "state"]
    queue_names: list = ["action", "step"]
    
    receiver.set_loop(loop)
    for name in async_queue_names:
        receiver.set_queue(name, asyncio.Queue(max_size))
    for name in queue_names:
        receiver.set_queue(name, Queue(max_size))
        

    # asyncio.run(receiver.run())
    # asyncio.create_task(receiver.syncronize_to_step())
    
    loop.create_task(receiver.run())
    # loop.create_task(receiver.syncronize_to_step())
    
    Thread(target=process_step_data, args=(receiver.step_queue, receiver.action_queue)).start()
    
    loop.run_forever()
    
