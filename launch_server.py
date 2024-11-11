import logging
import asyncio

import websockets

from remote.signaling_server import handler


async def launch_server(ip: str, port: int) -> None:
    async with websockets.serve(handler, ip, port):
        await asyncio.Future()  # run forever
        
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(launch_server("localhost", 8765))