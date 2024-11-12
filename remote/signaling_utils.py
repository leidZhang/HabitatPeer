import json
import logging
import asyncio
from typing import Any
from abc import ABC

import websockets
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription
)
from aiortc.contrib.signaling import TcpSocketSignaling


class InvalidAnswerException(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


class InvalidOfferException(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


class WebSocketSignaling:
    def __init__(self, uri: str) -> None:
        self.uri: int = uri
        self.websocket: websockets.WebSocketClientProtocol = None

    async def connect(self) -> None:
        print("Connecting to websocket server...")
        self.websocket = await websockets.connect(self.uri)
        print("Connected to websocket server")

    async def send(self, message: Any) -> None:
        print("Waiting for message sending to websocket server...")
        await self.websocket.send(json.dumps(message))
        print("Message sent to websocket server")

    async def receive(self) -> dict:
        print("Waiting for message receiving from websocket server...")
        data: dict = json.loads(await self.websocket.recv())
        print("Message received from websocket server")
        return data

    async def close(self) -> None:
        print("Waiting for websocket server closing...")
        await self.websocket.close()


class WebRTCClient(ABC):
    def __init__(self, signaling_ip: str, signaling_port: int, type: str = "websocket") -> None:
        if type == "websocket":
            self.signaling: WebSocketSignaling = WebSocketSignaling(
                f"ws://{signaling_ip}:{signaling_port}"
            )
        elif type == "tcp":
            self.signaling = TcpSocketSignaling(signaling_ip, signaling_port)
        else:
            raise ValueError("Invalid signaling type!")

        self.pc = RTCPeerConnection()
        self.ice_connection_state: str = "new"
        self.done: asyncio.Event = asyncio.Event()

    async def __setup(self) -> None:
        await self.signaling.connect()

        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate) -> None:
            await self.signaling.send({"ice": candidate})

        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange() -> None:
            print("ICE connection state is %s -> %s" % (
                self.ice_connection_state, self.pc.iceConnectionState
            ))
            self.ice_connection_state = self.pc.iceConnectionState
            if self.pc.iceConnectionState == "failed":
                await self.pc.close()
                await self.signaling.close()

    async def run(self) -> None:
        await self.__setup()


# TODO: May have to modify this function to handle TCP signaling
async def initiate_signaling(pc: RTCPeerConnection, signaling: WebSocketSignaling) -> None:
    if not isinstance(signaling, WebSocketSignaling):
        raise ValueError("signaling must be a WebSocketSignaling object!")

    # create sdp offer
    logging.info("Creating SDP offer...")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # send sdp offer to signaling server
    logging.info("Sending SDP offer to signaling server...")
    # await signaling.send(pc.localDescription)
    await signaling.send({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

    # wait for sdp answer from signaling server
    logging.info("Waiting for SDP answer from signaling server...")
    res = await signaling.receive()
    answer = RTCSessionDescription(res["sdp"], res['type'])
    await pc.setRemoteDescription(answer)


# TODO: May have to modify this function to handle TCP signaling
async def receive_signaling(pc: RTCPeerConnection, signaling: WebSocketSignaling) -> None:
    if not isinstance(signaling, WebSocketSignaling):
        raise ValueError("signaling must be a WebSocketSignaling object!")

    # receive sdp offer from signaling server
    logging.info("Waiting for SDP offer from signaling server...")
    res = await signaling.receive()
    offer = RTCSessionDescription(res["sdp"], res['type'])
    await pc.setRemoteDescription(offer)

    # create sdp answer
    logging.info("Creating SDP answer...")
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # send sdp answer to signaling server
    print("Sending SDP answer to signaling server...")
    await signaling.send({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })
