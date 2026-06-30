"""
SAHA – Vendor API Service (port 8002)
Exposes Vendor Abstraction Layer via REST.
Spec ref: §2.1–2.5
"""
from __future__ import annotations

import logging
import time

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from saha.contracts.vendor import (
    BudgetInterruptSignal,
    ProviderProfile,
    UnifiedAgentRequest,
    UnifiedAgentResponse,
)
from saha.db.connection import close_pool, get_pool, run_migrations
from saha.event_bus.client import get_bus
from saha.event_bus import topics
from saha.vendor import VendorGateway, get_gateway

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
logger = structlog.get_logger()

app = FastAPI(
    title="SAHA Vendor API",
    description="Vendor Abstraction Layer – unified provider interface",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_gateway: VendorGateway | None = None


@app.on_event("startup")
async def startup() -> None:
    global _gateway
    await run_migrations()
    bus = get_bus()
    await bus.connect()

    # Subscribe: handle budget interrupts forwarded from Execution Harness
    async def on_budget_interrupt(payload: dict) -> None:
        signal = BudgetInterruptSignal(**payload)
        await _gateway.interrupt(signal.provider_id, signal)  # type: ignore[union-attr]

    await bus.subscribe(topics.BUDGET_INTERRUPTS, on_budget_interrupt)

    _gateway = get_gateway()
    logger.info("vendor_api started", providers=_gateway.available_providers())


@app.on_event("shutdown")
async def shutdown() -> None:
    bus = get_bus()
    await bus.disconnect()
    await close_pool()


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "vendor_api"}


@app.get("/providers", response_model=list[str])
async def list_providers() -> list[str]:
    """List all registered provider IDs."""
    return _gateway.available_providers()  # type: ignore[union-attr]


@app.get("/providers/{provider_id}/profile", response_model=ProviderProfile)
async def get_profile(provider_id: str) -> ProviderProfile:
    try:
        return _gateway.get_profile(provider_id)  # type: ignore[union-attr]
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/complete", response_model=UnifiedAgentResponse)
async def complete(
    provider_id: str,
    request: UnifiedAgentRequest,
) -> UnifiedAgentResponse:
    """
    Forward a unified agent request to the named provider.
    Returns a normalised UnifiedAgentResponse.
    """
    if not _gateway:
        raise HTTPException(status_code=503, detail="Gateway not initialised")
    try:
        response = await _gateway.complete(provider_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return response


@app.post("/interrupt")
async def interrupt(signal: BudgetInterruptSignal) -> dict:
    """Forward a budget interrupt signal to the provider."""
    if not _gateway:
        raise HTTPException(status_code=503, detail="Gateway not initialised")
    cancelled = await _gateway.interrupt(signal.provider_id, signal)
    return {"cancelled": cancelled, "run_id": signal.run_id}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False)
