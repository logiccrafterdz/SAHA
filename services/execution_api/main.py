"""
SAHA – Execution API Service (port 8001)
Exposes the Agent Execution Harness via REST.
Clients POST a task; the harness runs the full multi-turn loop
and returns the final agent state.
Spec ref: §3.1–3.4
"""
from __future__ import annotations

import logging
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from saha.contracts.common import TaskType, new_uuid
from saha.contracts.eval import SuccessContract
from saha.contracts.execution import AgentState
from saha.contracts.vendor import RequestOptions, ToolSchema
from saha.db.connection import close_pool, run_migrations
from saha.event_bus.client import get_bus
from saha.event_bus import topics
from saha.execution.agent_loop import AgentLoop, TaskSpec
from saha.execution.agent_state import AgentStateManager
from saha.execution.tool_runner import build_default_tool_runner
from saha.vendor import get_gateway

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
logger = structlog.get_logger()

app = FastAPI(
    title="SAHA Execution API",
    description="Agent Execution Harness – multi-turn agent loop",
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

_loop: AgentLoop | None = None


@app.on_event("startup")
async def startup() -> None:
    global _loop
    await run_migrations()
    bus = get_bus()
    await bus.connect()
    gateway = get_gateway()
    tool_runner = build_default_tool_runner()
    _loop = AgentLoop(
        gateway=gateway,
        bus=bus,
        tool_runner=tool_runner,
        state_manager=AgentStateManager(),
    )
    logger.info("execution_api started")


@app.on_event("shutdown")
async def shutdown() -> None:
    bus = get_bus()
    await bus.disconnect()
    await close_pool()


# ─── Request / Response models ────────────────────────────────────────────────

class RunTaskRequest(BaseModel):
    """Input for starting an agent task."""
    task_id:          str              = Field(default_factory=new_uuid)
    provider_id:      str              = "claude_3_5_sonnet"
    message:          str
    system_prompt:    str              = ""
    scenario_id:      str              = "GENERIC"
    domain_tags:      list[str]        = Field(default_factory=list)
    tools:            list[ToolSchema] = Field(default_factory=list)
    success_contract: SuccessContract  = Field(default_factory=SuccessContract)
    options:          RequestOptions   = Field(default_factory=RequestOptions)


class RunTaskResponse(BaseModel):
    agent_state_id: str
    task_id:        str
    status:         str
    budget_used:    float
    current_step:   int


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "execution_api"}


@app.post("/tasks/run", response_model=RunTaskResponse)
async def run_task(body: RunTaskRequest) -> RunTaskResponse:
    """
    Start and run the complete agent loop for a task.
    Returns the final agent state when the loop completes (COMPLETED or FAILED).
    """
    if not _loop:
        raise HTTPException(status_code=503, detail="Execution harness not initialised")

    # Idempotency check: if task_id already exists, return its latest state
    mgr = AgentStateManager()
    existing_state = await mgr.get_by_task_id(body.task_id)
    if existing_state:
        logger.info("Idempotency hit: returning existing state", task_id=body.task_id)
        return RunTaskResponse(
            agent_state_id=existing_state.agent_state_id,
            task_id=existing_state.task_id,
            status=existing_state.status,
            budget_used=existing_state.budget_used,
            current_step=existing_state.current_step,
        )

    spec = TaskSpec(
        task_id=body.task_id,
        provider_id=body.provider_id,
        message=body.message,
        system_prompt=body.system_prompt,
        scenario_id=body.scenario_id,
        domain_tags=body.domain_tags,
        tools=body.tools,
        success_contract=body.success_contract,
        options=body.options,
    )

    try:
        final_state: AgentState = await _loop.run(spec)
    except Exception as exc:
        logger.exception("Task run failed", task_id=body.task_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Execution error: {exc}") from exc

    return RunTaskResponse(
        agent_state_id=final_state.agent_state_id,
        task_id=final_state.task_id,
        status=final_state.status,
        budget_used=final_state.budget_used,
        current_step=final_state.current_step,
    )


@app.get("/tasks/{agent_state_id}", response_model=AgentState)
async def get_task_state(agent_state_id: str) -> AgentState:
    """Retrieve the current state of an agent by its state ID."""
    mgr = AgentStateManager()
    state = await mgr.get(agent_state_id)
    if not state:
        raise HTTPException(status_code=404, detail="Agent state not found")
    return state


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
