"""
SAHA – Agent State Manager.
Persists and retrieves AgentState from PostgreSQL.
Spec ref: §3.2
"""
from __future__ import annotations

import json
import logging
from typing import Any

from saha.contracts.execution import AgentMemory, AgentState, AgentStatus
from saha.contracts.common import new_uuid
from saha.db.connection import get_pool

logger = logging.getLogger(__name__)


class AgentStateManager:
    """
    CRUD for AgentState records in PostgreSQL (agent_states table).
    The full state is stored as JSONB in the `state` column.
    """

    async def create(
        self,
        task_id: str,
        provider_id: str,
        budget_cap: float = 5.00,
    ) -> AgentState:
        state = AgentState(
            agent_state_id=new_uuid(),
            task_id=task_id,
            provider_id=provider_id,
            budget_cap=budget_cap,
        )
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_states
                    (agent_state_id, task_id, provider_id, status,
                     state, budget_used, budget_cap, current_step)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)
                """,
                state.agent_state_id,
                state.task_id,
                state.provider_id,
                state.status,
                json.dumps(state.to_bus_payload()),
                state.budget_used,
                state.budget_cap,
                state.current_step,
            )
        logger.info("Created AgentState id=%s for task_id=%s", state.agent_state_id, task_id)
        return state

    async def get(self, agent_state_id: str) -> AgentState | None:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state FROM agent_states WHERE agent_state_id = $1",
                agent_state_id,
            )
        if not row:
            return None
        data: dict[str, Any] = json.loads(row["state"])
        return AgentState(**data)

    async def update(self, state: AgentState) -> None:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE agent_states
                SET status       = $2,
                    state        = $3::jsonb,
                    budget_used  = $4,
                    current_step = $5,
                    updated_at   = NOW()
                WHERE agent_state_id = $1
                """,
                state.agent_state_id,
                state.status,
                json.dumps(state.to_bus_payload()),
                state.budget_used,
                state.current_step,
            )

    async def mark_completed(self, agent_state_id: str) -> None:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE agent_states
                SET status = $2, updated_at = NOW()
                WHERE agent_state_id = $1
                """,
                agent_state_id,
                AgentStatus.COMPLETED,
            )

    async def mark_failed(self, agent_state_id: str) -> None:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE agent_states
                SET status = $2, updated_at = NOW()
                WHERE agent_state_id = $1
                """,
                agent_state_id,
                AgentStatus.FAILED,
            )
