import time
import json
import logging
import asyncio
import fractions
from queue import Queue
from typing import Dict, Any, List

import numpy as np
from av import VideoFrame
from aiortc import RTCDataChannel, VideoStreamTrack

from .signaling_utils import WebRTCClient, receive_signaling
from .comm_utils import (
    BaseAsyncComponent,
    decode_from_rgba,
    push_to_buffer
)

GARBAGE_FRAME: VideoFrame = VideoFrame.from_ndarray(np.zeros((1, 1, 3), dtype=np.uint8), format="rgb24")
GARBAGE_FRAME.pts = 0
GARBAGE_FRAME.time_base = fractions.Fraction(1, 90000)


class RGBProcessor(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self, track: VideoStreamTrack) -> None:
        super().__init__()
        self.track: VideoStreamTrack = track

    async def recv(self) -> VideoFrame:
        # print("Receiving RGB frame...")
        frame: VideoFrame = await asyncio.wait_for(self.track.recv(), timeout=5.0)  # 5 seconds timeout
        # print("Decoding to rgb frame...")
        image: np.ndarray = frame.to_ndarray(format="rgb24")
        await self.input_queue.put({'rgb': image, 'pts': frame.pts})
        # print(f"PTS: {frame.pts} RGB image put into queue...")

        return GARBAGE_FRAME


class DepthProcessor(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self, track: VideoStreamTrack) -> None:
        super().__init__()
        self.track: VideoStreamTrack = track

    async def recv(self) -> VideoFrame:
        # print("Receiving Depth frame...")
        frame: VideoFrame = await asyncio.wait_for(self.track.recv(), timeout=5.0)  # 5 seconds timeout
        # print("Decoding to rgba frame...")
        image: np.ndarray = frame.to_ndarray(format="rgba")
        image = decode_from_rgba(image, np.float32)
        await self.input_queue.put({'depth': image, 'pts': frame.pts})
        # print(f"PTS: {frame.pts} Depth image put into queue...")

        return GARBAGE_FRAME


class SemanticProcessor(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self, track: VideoStreamTrack) -> None:
        super().__init__()
        self.track: VideoStreamTrack = track

    async def recv(self) -> VideoFrame:
        # print("Receiving Semantic frame...")
        frame: VideoFrame = await asyncio.wait_for(self.track.recv(), timeout=5.0)  # 5 seconds timeout
        # print("Decoding to rgba frame...")
        image: np.ndarray = frame.to_ndarray(format="rgba")
        image = decode_from_rgba(image, np.int32)
        await self.input_queue.put({'semantic': image, 'pts': frame.pts})
        # print(f"PTS: {frame.pts} Semantic image put into queue...")

        return GARBAGE_FRAME


class ReceiverPeer(WebRTCClient):
    def __init__(self, signaling_ip, signaling_port) -> None:
        super().__init__(signaling_ip, signaling_port)
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

    # TODO: May have to find some way to avoid hard coding the track order
    def __setup_track_callbacks(self) -> None:
        @self.pc.on("track")
        def on_track(track: VideoStreamTrack):
            if track.kind == "video":
                if self.track_counter == 0:
                    self.__handle_rgb_track(track)
                elif self.track_counter == 1:
                    self.__handle_depth_track(track)
                elif self.track_counter == 2:
                    self.__handle_semantic_track(track)

                self.track_counter += 1

    def __setup_datachannel_callbacks(self) -> None:
        @self.pc.on("datachannel")
        async def on_datachannel(channel: RTCDataChannel) -> None:
            self.data_channel = channel

            @self.data_channel.on("open")
            async def on_open() -> None:
                while not self.done.is_set():
                    action: Dict[str, Any] = await self.loop.run_in_executor(None, self.action_queue.get)
                    print(f"Sending action {action} to provider...")
                    self.data_channel.send(json.dumps(action))
                print("Waiting for provider to close data channel...")

            @self.data_channel.on("message")
            async def on_message(message: bytes) -> None:
                print(f"Received message: {message} for provider...")
                state: Dict[str, Any] = json.loads(message)
                await self.syncronize_to_step(state)

            @self.data_channel.on("close")
            def on_close() -> None:
                logging.info("Data channel closed")
                self.done.set()

            # NOTE: I dont know why this is needed, but without it, on_open() is not called
            if self.data_channel.readyState == "open":
                await on_open()

    def __set_async_components(
        self,
        component: BaseAsyncComponent,
        queue: Queue,
    ) -> None:
        component.set_loop(self.loop)
        component.set_input_queue(queue)

    def __handle_rgb_track(self, track: VideoStreamTrack) -> None:
        rgb_processor: VideoStreamTrack = RGBProcessor(track)
        self.__set_async_components(rgb_processor, self.rgb_queue)
        self.pc.addTrack(rgb_processor)

    def __handle_depth_track(self, track: VideoStreamTrack) -> None:
        depth_processor: VideoStreamTrack = DepthProcessor(track)
        self.__set_async_components(depth_processor, self.depth_queue)
        self.pc.addTrack(depth_processor)

    def __handle_semantic_track(self, track: VideoStreamTrack) -> None:
        semantic_processor: VideoStreamTrack = SemanticProcessor(track)
        self.__set_async_components(semantic_processor, self.semantic_queue)
        self.pc.addTrack(semantic_processor)

    # TODO: Use the correct synchronization method rather simply waitting for the state to be updated
    async def syncronize_to_step(self, state: dict) -> None: # asyncio.create_task(receiver.process_data())
        # while not self.done.is_set():
        logging.info("Synchronizing data...")
        rgb_data: Dict[str, Any] = await self.rgb_queue.get()
        depth_data: Dict[str, Any] = await self.depth_queue.get()
        semantic_data: Dict[str, Any] = await self.semantic_queue.get()
        # state: Dict[str, Any] = await self.state_queue.get()

        # if max_pts - min_pts <= 100:
        print(rgb_data['pts'], depth_data['pts'], semantic_data['pts'], state['pts'])
        step: Dict[str, Any] = {
            'rgb': rgb_data['rgb'],
            'depth': depth_data['depth'],
            'semantic': semantic_data['semantic'],
        }
        step.update(state)
        push_to_buffer(self.step_queue, step)

    async def run(self) -> None: # asyncio.run(receiver.run())
        await super().run()
        self.__setup_track_callbacks()
        self.__setup_datachannel_callbacks()
        await receive_signaling(self.pc, self.signaling)

        await self.done.wait()
        await self.pc.close()
        await self.signaling.close()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def set_queue(self, queue_name: str, queue: Queue) -> None:
        setattr(self, f"{queue_name}_queue", queue)
