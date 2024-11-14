import time
import json
import asyncio
import logging
import fractions
from queue import Queue
from datetime import datetime
from typing import Tuple, Dict, Any, List

import cv2
import numpy as np
from av import VideoFrame
from aiortc import RTCDataChannel, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole

from .comm_utils import (
    BaseAsyncComponent,
    encode_to_rgba,
    get_frame_from_buffer,
    push_to_buffer,
)
from .signaling_utils import WebRTCClient, initiate_signaling

# Copied from aiortc source code
VIDEO_PTIME = 1 / 30
VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)


def add_timestamp_to_image(image: np.ndarray) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    cv2.putText(image, timestamp, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


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
        self.data_channel.send(json.dumps(data))


class RGBStreamTrack(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self) -> None:
        super().__init__()
        self.last_frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame: np.ndarray = await self.loop.run_in_executor(
            None, get_frame_from_buffer, self.input_queue
        )

        # Convert frame to RGB
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.ascontiguousarray(frame) # Make sure frame is contiguous in memory
            self.last_frame = frame # Update last frame
        else:
            frame = self.last_frame # send the last frame if is not ready yet

        # Create VideoFrame
        add_timestamp_to_image(frame)
        video_frame: VideoFrame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts, video_frame.time_base = pts, time_base

        return video_frame


class RGBAStreamTrack(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self) -> None:
        super().__init__()
        self.last_frame: np.ndarray = np.zeros((480, 640, 4), dtype=np.uint8)

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame: np.ndarray = await self.loop.run_in_executor(
            None, get_frame_from_buffer, self.input_queue
        )

        # Convert frame to RGBA
        if frame is not None:
            frame = encode_to_rgba(frame) # Use 4 channels to store int32 or float32
            frame = np.ascontiguousarray(frame) # Make sure frame is contiguous in memory
            self.last_frame = frame # Update last frame
        else:
            frame = self.last_frame # send the last frame if is not ready yet

        # Create VideoFrame
        video_frame: VideoFrame = VideoFrame.from_ndarray(frame, format="rgba")
        video_frame.pts, video_frame.time_base = pts, time_base

        return video_frame


class ProviderPeer(WebRTCClient):
    def __init__(
        self,
        signaling_ip: str,
        signaling_port: int,
        stun_urls: List[str] = None
    ) -> None:
        super().__init__(signaling_ip, signaling_port, stun_urls=stun_urls)
        self.data_channel: RTCDataChannel = None
        self.data_sender: StateSender = None
        self.blackhole: MediaBlackhole = None

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
        self.blackhole = MediaBlackhole() # TODO: Temp measure, receiver should not send back any frames

        @self.pc.on("track")
        async def on_track(track: VideoStreamTrack) -> None:
            print("Received garbage track")
            self.blackhole.addTrack(track)
            await self.blackhole.start()

        rgb_track: VideoStreamTrack = RGBStreamTrack()
        self.__set_async_components(rgb_track, self.rgb_queue)
        self.pc.addTrack(rgb_track)

        depth_track: VideoStreamTrack = RGBAStreamTrack()
        self.__set_async_components(depth_track, self.depth_queue)
        self.pc.addTrack(depth_track)

        semantic_track: VideoStreamTrack = RGBAStreamTrack()
        self.__set_async_components(semantic_track, self.semantic_queue)
        self.pc.addTrack(semantic_track)

    def __setup_datachannel_callbacks(self) -> None:
        self.data_channel = self.pc.createDataChannel("datachannel")
        self.data_sender: StateSender = StateSender(self.data_channel)
        self.__set_async_components(self.data_sender, self.state_queue)

        @self.data_channel.on("open")
        async def on_open() -> None:
            logging.info("Data channel opened")
            while not self.done.is_set():
                await self.data_sender.send_state()

        @self.data_channel.on("message")
        def on_message(message: bytes) -> None:
            action: Dict[str, Any] = json.loads(message)
            # print(f"Received action: {action}")
            self.loop.run_in_executor(
                None, push_to_buffer, self.action_queue, action
            )

        @self.data_channel.on("close")
        def on_close() -> None:
            logging.info("Data channel closed")
            self.done.set()

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
