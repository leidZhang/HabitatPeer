import asyncio
from queue import Queue
from threading import Thread

import cv2
from remote import ReceiverPeer


# Note: Only for the integration test on the local machine
def process_step_data(queue: Queue):
    while True:
        step_data: dict = queue.get()
        
        cv2.imshow("RGB received", step_data["rgb"])
        cv2.imshow("Depth received", step_data["depth"])
        cv2.waitKey(1)


if __name__ == "__main__":
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
        

    asyncio.run(receiver.run())
    asyncio.create_task(receiver.syncronize_to_step())
    Thread(target=process_step_data, args=(receiver.step_queue,)).start()