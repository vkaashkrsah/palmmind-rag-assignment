import json

import redis.asyncio as redis

from app.config import settings

Message = dict[str, str]


class ChatMemory:
    """Redis-backed chat history and in-progress booking state, keyed by session."""

    def __init__(self) -> None:
        self._r = redis.from_url(settings.redis_url, decode_responses=True)

    async def history(self, session_id: str) -> list[Message]:
        raw = await self._r.lrange(f"chat:{session_id}", 0, -1)
        return [json.loads(m) for m in raw]

    async def add(self, session_id: str, role: str, content: str) -> None:
        key = f"chat:{session_id}"
        await self._r.rpush(key, json.dumps({"role": role, "content": content}))
        await self._r.ltrim(key, -settings.chat_history_limit, -1)
        await self._r.expire(key, settings.chat_ttl_seconds)

    async def get_booking(self, session_id: str) -> dict[str, str]:
        return await self._r.hgetall(f"booking:{session_id}")

    async def set_booking(self, session_id: str, state: dict[str, str]) -> None:
        if not state:
            return
        key = f"booking:{session_id}"
        await self._r.hset(key, mapping=state)
        await self._r.expire(key, settings.chat_ttl_seconds)

    async def clear_booking(self, session_id: str) -> None:
        await self._r.delete(f"booking:{session_id}")
