import time
import asyncio
from queue import Queue
from abc import ABC
from typing import Any, Union

import numpy as np
from av import VideoFrame
from aiortc import VideoStreamTrack

RGBA_CHANNELS: int = 4
LABEL_MAP_CHANNELS: int = 1


class InvalidImageShapeError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class InvalidDataTypeError(Exception):
    def __init__(self, message: str) -> None:
        self.message: str = message


def encode_to_rgba(image: np.ndarray) -> np.ndarray:
    height, width, channel = image.shape
    if image.dtype not in [np.float32, np.int32]:
        raise InvalidDataTypeError("Image must be of type float32 or int32")
    if channel != 1:
        raise InvalidImageShapeError("Image must be single channel")

    return image.view(np.uint8).reshape(height, width, RGBA_CHANNELS)

def decode_from_rgba(rgba: np.ndarray, data_type: np.dtype) -> np.ndarray:
    if rgba.shape[2] != 4:
        raise InvalidImageShapeError("RGBA image must have 4 channels")
    if data_type not in [np.float32, np.int32]:
        raise InvalidDataTypeError("Data type must be float32 or int32")

    return rgba.view(data_type).reshape(rgba.shape[0], rgba.shape[1], LABEL_MAP_CHANNELS)

def rgb_to_rgba(rgb: np.ndarray, alpha: int = 255) -> np.ndarray:
    if alpha < 0 or alpha > 255:
        raise ValueError("Alpha channel must be between 0 and 255")
    if rgb.shape[-1] != 3:
        raise InvalidImageShapeError("RGB image must have 3 channels")

    alpha_channel: np.ndarray = np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.uint8) * alpha
    return np.concatenate((rgb, alpha_channel), axis=-1)


def get_frame_from_buffer(buffer: Queue) -> Any:
    if buffer.empty(): # If buffer is empty, directly return None
        time.sleep(0.001)
        return None
    return buffer.get()

def push_to_buffer(buffer: Queue, data: Any) -> None:
    if buffer.full():
        buffer.get()
    buffer.put(data)
    
async def push_to_async_buffer(buffer: asyncio.Queue, data: Any) -> None:
    if buffer.full():
        await buffer.get()
    await buffer.put(data)

def empty_queue(queue: Queue) -> None:
    print(f"Emptying queue with {queue.qsize()} elements")
    while queue.qsize() > 0:
        queue.get()
    print(f"Current queue size is {queue.qsize()}")

async def empty_async_queue(queue: asyncio.Queue) -> None:
    print(f"Emptying async queue with {queue.qsize()} elements")
    while not queue.empty():
        await queue.get()
    print(f"Current async queue size is {queue.qsize()}")


class BaseAsyncComponent(ABC):
    def __init__(self):
        self.input_queue: Union[Queue, asyncio.Queue] = None
        self.loop: asyncio.AbstractEventLoop = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def set_input_queue(self, input_queue: Union[Queue, asyncio.Queue]) -> None:
        self.input_queue = input_queue
