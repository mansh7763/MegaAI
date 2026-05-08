import asyncio
import json
from typing import AsyncIterator


class SSEManager:
    """Manages an async queue of SSE events for streaming to clients."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._done = False

    def push(self, event: dict):
        self._queue.put_nowait(event)

    def close(self):
        self._done = True
        self._queue.put_nowait(None)

    async def stream(self) -> AsyncIterator[str]:
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"
