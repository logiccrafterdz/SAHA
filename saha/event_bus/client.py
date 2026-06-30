"""
SAHA – Async Event Bus Client (Redis pub/sub backend).
Designed to be swapped for Kafka/RabbitMQ in Phase 2 without changing
any SAHA contract or topic name.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any

import redis.asyncio as aioredis
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class BusSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    REDIS_URL: str = "redis://localhost:6379/0"


class SAHABusClient:
    """
    Thin async wrapper around Redis pub/sub.
    Usage:
        bus = SAHABusClient()
        await bus.connect()
        await bus.publish("SAHA/agent_requests", payload_dict)
        await bus.subscribe("SAHA/provider_responses", handler)
    """

    def __init__(self) -> None:
        settings = BusSettings()
        self._redis_url = settings.REDIS_URL
        self._client: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._listener_task: asyncio.Task[None] | None = None
        self._handlers: dict[str, list[Callable[[dict[str, Any]], Coroutine[Any, Any, None]]]] = {}

    async def connect(self) -> None:
        self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        self._pubsub = self._client.pubsub()
        logger.info("SAHABusClient connected to %s", self._redis_url)

    async def disconnect(self) -> None:
        if self._listener_task:
            self._listener_task.cancel()
        if self._pubsub:
            await self._pubsub.close()
        if self._client:
            await self._client.aclose()
        logger.info("SAHABusClient disconnected")

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish a JSON-serialised payload to a topic."""
        if not self._client:
            raise RuntimeError("SAHABusClient not connected. Call connect() first.")
        message = json.dumps(payload)
        await self._client.publish(topic, message)
        logger.debug("Published to %s: %s", topic, message[:120])

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to a topic with an async handler function."""
        if not self._pubsub:
            raise RuntimeError("SAHABusClient not connected. Call connect() first.")

        if topic not in self._handlers:
            self._handlers[topic] = []
            await self._pubsub.subscribe(topic)
            logger.info("Subscribed to topic: %s", topic)

        self._handlers[topic].append(handler)

        # Start listener if not already running
        if not self._listener_task or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        """Background task that dispatches incoming messages to handlers."""
        assert self._pubsub is not None
        try:
            async for raw in self._pubsub.listen():
                if raw["type"] != "message":
                    continue
                topic: str = raw["channel"]
                try:
                    payload: dict[str, Any] = json.loads(raw["data"])
                except json.JSONDecodeError:
                    logger.warning("Non-JSON message on %s, skipping", topic)
                    continue

                for handler in self._handlers.get(topic, []):
                    try:
                        await handler(payload)
                    except Exception:
                        logger.exception("Handler error on topic %s", topic)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Event bus listener crashed")


# ─── Module-level singleton (one per service process) ───────────────────────
_bus: SAHABusClient | None = None


def get_bus() -> SAHABusClient:
    global _bus
    if _bus is None:
        _bus = SAHABusClient()
    return _bus
