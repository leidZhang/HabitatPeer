import time
import json
import logging
import asyncio
from queue import Queue
from typing import Dict, Any, List

import av
import numpy as np
from aiortc import RTCDataChannel, VideoStreamTrack

from .signaling_utils import WebRTCClient, receive_signaling
from .comm_utils import (
    BaseAsyncComponent, 
    decode_from_rgba,
    push_to_buffer
)


class RGBProcessor(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self, track: VideoStreamTrack) -> None:
        super().__init__()
        self.track: VideoStreamTrack = track
        
    async def recv(self) -> av.VideoFrame:
        frame: av.VideoFrame = await self.track.recv()
        image: np.ndarray = frame.to_ndarray(format="rgb24")
        await self.input_queue.put({'rgb': image, 'pts': frame.pts})
        
        return frame
    
    
class DepthProcessor(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self, track: VideoStreamTrack) -> None:
        super().__init__()
        self.track: VideoStreamTrack = track
        
    async def recv(self) -> av.VideoFrame:
        frame: av.VideoFrame = await self.track.recv()
        image: np.ndarray = frame.to_ndarray(format="rgba")
        image = decode_from_rgba(image, np.float32)
        await self.input_queue.put({'rgb': image, 'pts': frame.pts})
        
        return frame
    
    
class SemanticProcessor(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self, track: VideoStreamTrack) -> None:
        super().__init__()
        self.track: VideoStreamTrack = track
        
    async def recv(self) -> av.VideoFrame:
        frame: av.VideoFrame = await self.track.recv()
        image: np.ndarray = frame.to_ndarray(format="rgba")
        image = decode_from_rgba(image, np.int32)
        await self.input_queue.put({'rgb': image, 'pts': frame.pts})
        
        return frame
    
    
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
        
        # Buffers for each stream/track
        self.depth_buffer: List[Dict[str, Any]] = []
        self.rgb_buffer: List[Dict[str, Any]] = []
        self.semantic_buffer: List[Dict[str, Any]] = []
        self.state_buffer: List[Dict[str, Any]] = []
        
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
            def on_open() -> None:
                while not self.done.is_set():
                    action: Dict[str, Any] = self.loop.run_in_executor(None, self.action_queue.get)
                    self.data_channel.send(json.dumps(action))

            @self.data_channel.on("message")
            async def on_message(message: bytes) -> None:
                state: Dict[str, Any] = json.loads(message)
                await self.state_queue.put(state)

            @self.data_channel.on("close")
            def on_close() -> None:
                print("Data channel closed")

            # NOTE: I dont know why this is needed, but without it, on_open() is not called
            if self.data_channel.readyState == "open":
                await on_open()
                
    def __handle_rgb_track(self, track: VideoStreamTrack) -> None:
        rgb_processor: VideoStreamTrack = RGBProcessor(track)
        self.pc.addTrack(rgb_processor)
        
    def __handle_depth_track(self, track: VideoStreamTrack) -> None:
        depth_processor: VideoStreamTrack = DepthProcessor(track)
        self.pc.addTrack(depth_processor)
        
    def __handle_semantic_track(self, track: VideoStreamTrack) -> None:
        semantic_processor: VideoStreamTrack = SemanticProcessor(track)
        self.pc.addTrack(semantic_processor)
        
    async def syncronize_to_step(self) -> None: # asyncio.create_task(receiver.process_data())
        while not self.done.is_set():
            rgb_data: Dict[str, Any] = await self.rgb_queue.get()
            depth_data: Dict[str, Any] = await self.depth_queue.get()
            semantic_data: Dict[str, Any] = await self.semantic_queue.get()
            state: Dict[str, Any] = await self.state_queue.get()
            
            self.rgb_buffer.append(rgb_data)
            self.depth_buffer.append(depth_data)
            self.semantic_buffer.append(semantic_data)
            self.state_buffer.append(state)
            
            # Find the closest matching pts values
            min_pts: int = min(rgb_data['pts'], depth_data['pts'], semantic_data['pts'])
            max_pts: int = max(rgb_data['pts'], depth_data['pts'], semantic_data['pts'])
            
            if max_pts - min_pts > 100:
                return None
            
            step: Dict[str, Any] = {
                'rgb': rgb_data['rgb'], 
                'depth': depth_data['depth'], 
                'semantic': semantic_data['semantic'], 
            }
            step.update(state)
            push_to_buffer(self.step_queue, step)
            
            # Remove the used data from buffers
            self.rgb_buffer.remove(rgb_data)
            self.depth_buffer.remove(depth_data)
            self.semantic_buffer.remove(semantic_data)
            self.state_buffer.remove(state)
            
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
        