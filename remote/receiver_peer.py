import time
import json
import logging
import asyncio
import fractions
from queue import Queue
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from av import VideoFrame
from aiortc import RTCDataChannel, VideoStreamTrack, RTCRtpSender

from .signaling_utils import WebRTCClient, receive_signaling
from .comm_utils import (
    BaseAsyncComponent,
    decode_to_depth,
    decode_to_semantic,
    force_codec,
    push_to_buffer,
    push_to_async_buffer,
    empty_async_queue,
    empty_queue
)

GARBAGE_FRAME: VideoFrame = VideoFrame.from_ndarray(np.zeros((2, 2, 3), dtype=np.uint8), format="rgb24")
GARBAGE_FRAME.pts = 0
GARBAGE_FRAME.time_base = fractions.Fraction(1, 90000)
ASYNC_QUEUE_NAMES: List[str] = ['rgb', 'depth','semantic','state']
QUEUE_NAMES: List[str] = ['action', 'step']


class RGBProcessor(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self, track: VideoStreamTrack) -> None:
        super().__init__()
        self.track: VideoStreamTrack = track

    async def recv(self) -> VideoFrame:
        frame: VideoFrame = await asyncio.wait_for(self.track.recv(), timeout=2.0)  # 2 seconds timeout
        image: np.ndarray = frame.to_ndarray(format="rgb24")

        if np.all(image == 0):
            return GARBAGE_FRAME
        await push_to_async_buffer(self.input_queue, {'rgb': image, 'pts': frame.pts})

        return GARBAGE_FRAME


class DepthProcessor(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self, track: VideoStreamTrack) -> None:
        super().__init__()
        self.track: VideoStreamTrack = track

    async def recv(self) -> VideoFrame:
        frame: VideoFrame = await asyncio.wait_for(self.track.recv(), timeout=2.0)  # 2 seconds timeout
        image: np.ndarray = frame.to_ndarray(format="rgb24")
        image = decode_to_depth(image)

        if np.all(image == 0):
            return GARBAGE_FRAME
        await push_to_async_buffer(self.input_queue, {'depth': image, 'pts': frame.pts})

        return GARBAGE_FRAME


class SemanticProcessor(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self, track: VideoStreamTrack) -> None:
        super().__init__()
        self.track: VideoStreamTrack = track

    async def recv(self) -> VideoFrame:
        frame: VideoFrame = await asyncio.wait_for(self.track.recv(), timeout=2.0)  # 2 seconds timeout
        image: np.ndarray = frame.to_ndarray(format="rgb24")
        image = decode_to_semantic(image)

        if np.all(image == 0):
            return GARBAGE_FRAME
        await push_to_async_buffer(self.input_queue, {'semantic': image, 'pts': frame.pts})

        return GARBAGE_FRAME
    

class ReceiverPeer(WebRTCClient):
    def __init__(
        self,
        signaling_ip: str,
        signaling_port: int,
        stun_urls: List[str] = None
    ) -> None:
        super().__init__(signaling_ip, signaling_port, stun_urls=stun_urls)
        self.data_channel: RTCDataChannel = None
        self.track_counter: int = 0

        self.loop: asyncio.AbstractEventLoop = None
        # Queues for each stream/track
        self.depth_queue: asyncio.Queue = None
        self.rgb_queue: asyncio.Queue = None
        self.semantic_queue: asyncio.Queue = None
        self.state_queue: asyncio.Queue = None
        self.step_queue: Queue = None
        self.action_queue: Queue = None

    # TODO: Use the correct synchronization method rather simply waitting for the state to be updated
    async def syncronize_to_step(self, state: dict) -> None:
        logging.info("Synchronizing data...")
        rgb_data, depth_data, semantic_data = await self.__synchronize_images()

        print(rgb_data['pts'], depth_data['pts'], semantic_data['pts'], state['pts'])
        step: Dict[str, Any] = {
            'rgb': rgb_data['rgb'],
            'depth': depth_data['depth'],
            'semantic': semantic_data['semantic'],
        }
        step.update(state) # Merge the state with the step data
        await self.loop.run_in_executor(None, push_to_buffer, self.step_queue, step)

    async def send_action(self) -> None:
        action: Dict[str, Any] = await self.loop.run_in_executor(None, self.action_queue.get)
        print(f"Sending action {action} to provider...")
        self.data_channel.send(json.dumps(action))

    async def run(self) -> None: # asyncio.run(receiver.run())
        while not self.disconnected.set():
            await super().run()
            self.__setup_track_callbacks()
            self.__setup_datachannel_callbacks()
            await receive_signaling(self.pc, self.signaling)

            await self.done.wait()
            await self.pc.close()
            await empty_async_queue(self.rgb_queue)
            await self.signaling.close()

    async def clear_queues(self) -> None:
        for queue_name in ASYNC_QUEUE_NAMES:
            await empty_async_queue(getattr(self, f"{queue_name}_queue"))
        for queue_name in QUEUE_NAMES:
            empty_queue(getattr(self, f"{queue_name}_queue"))

    async def __synchronize_images(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        rgb_data: Dict[str, Any] = await self.rgb_queue.get()
        depth_data: Dict[str, Any] = await self.depth_queue.get()
        semantic_data: Dict[str, Any] = await self.semantic_queue.get()
        return rgb_data, depth_data, semantic_data

    # TODO: May have to find some way to avoid hard coding the track order
    def __handle_stream_tracks(self, track: VideoStreamTrack) -> None:
            if self.track_counter == 0:
                local_track: VideoStreamTrack = RGBProcessor(track)
                target_queue: asyncio.Queue = self.rgb_queue
            elif self.track_counter == 1:
                local_track: VideoStreamTrack = DepthProcessor(track)
                target_queue: asyncio.Queue = self.depth_queue
            elif self.track_counter == 2:
                local_track: VideoStreamTrack = SemanticProcessor(track)
                target_queue: asyncio.Queue = self.semantic_queue
            self.__set_async_components(local_track, target_queue)
            sender: RTCRtpSender =self.pc.addTrack(local_track)
            force_codec(self.pc, sender, "video/H264")

            self.track_counter = (self.track_counter + 1) % 3

    def __setup_track_callbacks(self) -> None:
        @self.pc.on("track")
        def on_track(track: VideoStreamTrack):
            if track.kind == "video":
                self.__handle_stream_tracks(track)

    def __setup_datachannel_callbacks(self) -> None:
        @self.pc.on("datachannel")
        async def on_datachannel(channel: RTCDataChannel) -> None:
            self.data_channel = channel

            @self.data_channel.on("open")
            async def on_open() -> None:
                print("Data channel opened")

            @self.data_channel.on("message")
            async def on_message(message: bytes) -> None:
                print(f"Received message: {message} for provider...")
                state: Dict[str, Any] = json.loads(message)
                await self.syncronize_to_step(state)
                await self.send_action()

            @self.data_channel.on("close")
            def on_close() -> None:
                logging.info("Data channel closed")
                self.done.set()

    def __set_async_components(
        self,
        component: BaseAsyncComponent,
        queue: Queue,
    ) -> None:
        component.set_loop(self.loop)
        component.set_input_queue(queue)

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def set_queue(self, queue_name: str, queue: Queue) -> None:
        setattr(self, f"{queue_name}_queue", queue)
