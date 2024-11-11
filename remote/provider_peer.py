import time
import json
import asyncio
import fractions
from queue import Queue
from typing import Tuple, Dict, Any

import cv2
import numpy as np
from av.video import VideoFrame
from aiortc import VideoStreamTrack, RTCDataChannel

from .comm_utils import (
    BaseAsyncComponent, 
    encode_to_rgba,
    push_to_buffer
)
from .signaling_utils import WebRTCClient, initiate_signaling


# Copied from aiortc source code
VIDEO_PTIME = 1 / 30
VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)


class StateSender(BaseAsyncComponent):
    _timestamp: int
    _start: float
    
    def __init__(
        self, 
        data_channel: RTCDataChannel,
    ) -> None:
        super().__init__()
        self.data_channel: RTCDataChannel = data_channel

    # NOTE: This function is copied from aiortc source code
    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if hasattr(self, "_timestamp"):
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

    async def send_state(self) -> None:
        pts, _ = await self.next_timestamp()
        data: dict = await self.loop.run_in_executor(
            None, self.input_queue.get
        )
        data["pts"] = pts
        print(f"Sending state: {data}, type: {type(data)}")
        self.data_channel.send(json.dumps(data))
        
        
class RGBStreamTrack(VideoStreamTrack, BaseAsyncComponent):
    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame: np.ndarray = await self.loop.run_in_executor(
            None, self.input_queue.get
        )

        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.ascontiguousarray(frame) # Make sure frame is contiguous in memory

        # Create VideoFrame
        video_frame: VideoFrame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts, video_frame.time_base = pts, time_base
        print(f"Sending RGB frame, shape: {frame.shape} type: {type(video_frame)}")

        return video_frame
    
    
class RGBAStreamTrack(VideoStreamTrack, BaseAsyncComponent):
    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame: np.ndarray = await self.loop.run_in_executor(
            None, self.input_queue.get
        )

        # Convert frame to RGBA
        frame = encode_to_rgba(frame) # Use 4 channels to store int32 or float32
        frame = np.ascontiguousarray(frame) # Make sure frame is contiguous in memory

        # Create VideoFrame
        video_frame: VideoFrame = VideoFrame.from_ndarray(frame, format="rgba")
        video_frame.pts, video_frame.time_base = pts, time_base
        print(f"Sending RGBA frame, shape: {frame.shape} type: {type(video_frame)}")

        return video_frame
    
    
class ProviderPeer(WebRTCClient):
    def __init__(self, signaling_ip: str, signaling_port: int) -> None:
        super().__init__(signaling_ip, signaling_port)
        self.data_channel: RTCDataChannel = None
        self.data_sender: StateSender = None
        
        self.loop: asyncio.AbstractEventLoop = None
        # Queues for each stream/track
        self.depth_queue: Queue = None
        self.rgb_queue: Queue = None
        self.semantic_queue: Queue = None
        self.state_queue: Queue = None
        self.action_queue: Queue = None
        
    def __set_async_components(
        self, 
        component: BaseAsyncComponent, 
        queue: Queue,
    ) -> None:
        component.set_loop(self.loop)
        component.set_input_queue(queue)

    def __setup_track_callbacks(self) -> None:
        rgb_track: RGBStreamTrack = RGBStreamTrack()
        self.__set_async_components(rgb_track, self.rgb_queue)
        self.pc.addTrack(rgb_track)
        
        depth_track: RGBAStreamTrack = RGBAStreamTrack()
        self.__set_async_components(depth_track, self.depth_queue)
        self.pc.addTrack(depth_track)
        
        semantic_track: RGBAStreamTrack = RGBAStreamTrack()
        self.__set_async_components(semantic_track, self.semantic_queue)
        self.pc.addTrack(semantic_track)

    def __setup_datachannel_callbacks(self) -> None:
        self.data_channel = self.pc.createDataChannel("datachannel")
        self.data_sender: StateSender = StateSender(self.data_channel)
        self.__set_async_components(self.data_sender, self.state_queue)

        @self.data_channel.on("open")
        async def on_open() -> None:
            print("Data channel opened")
            while True:
                await self.data_sender.send_state()

        @self.data_channel.on("message")
        def on_message(message: bytes) -> None:
            action: Dict[str, Any] = json.loads(message)
            print(f"Received action: {action}")
            self.loop.run_in_executor(
                None, push_to_buffer, self.action_queue, action
            )

        @self.data_channel.on("close")
        def on_close() -> None:
            print("Data channel closed")

    async def run(self) -> None:
        await super().run()
        self.__setup_track_callbacks()
        self.__setup_datachannel_callbacks()
        await initiate_signaling(self.pc, self.signaling)

        await self.done.wait()
        await self.pc.close()
        await self.signaling.close()
        
    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        
    def set_queue(self, queue_name: str, queue: Queue) -> None:
        setattr(self, f"{queue_name}_queue", queue)


# if __name__ == "__main__":
#     ip, port = "localhost", 1234
#     max_queue_size: int = 5
#     logging.basicConfig(level=logging.ERROR)
    
#     peer: ProviderPeer = ProviderPeer(ip, port)
#     depth_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
#     rgb_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
#     semantic_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
#     state_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

#     loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
#     rgb_queue, depth_queue = Queue(max_queue_size), Queue(max_queue_size)
#     semantic_queue, state_queue = Queue(max_queue_size), Queue(max_queue_size)

#     try:
#         peer.set_loop(loop)
#         peer.set_queue("depth", depth_queue)
#         peer.set_queue("rgb", rgb_queue)
#         peer.set_queue("semantic", semantic_queue)
#         peer.set_queue("state", state_queue)
#     except KeyboardInterrupt:
#         print("User interrupted the program")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         print("Closing the program...")
#         peer.done.set()
#         empty_queue(depth_queue)
#         empty_queue(rgb_queue)
#         empty_queue(semantic_queue)
#         empty_queue(state_queue)
        
#         loop.close()
#         depth_executor.shutdown()
#         rgb_executor.shutdown()
#         semantic_executor.shutdown()
#         state_executor.shutdown()
        