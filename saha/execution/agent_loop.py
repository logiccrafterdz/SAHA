"""
SAHA – Agent Execution Loop (core of the Execution Harness)  [Phase 2]
Implements the multi-turn agent loop described in §3.3.
Handles: COMPLETED, NEEDS_TOOL, FAILED, budget tracking, bus events, Cost Router (§4).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from saha.contracts.common import CanonicalError, ErrorCode, ErrorSeverity, ErrorType, new_uuid
from saha.contracts.eval import EvalContext, EvalInput, EvalProviderInfo, SuccessContract
from saha.contracts.execution import AgentMemory, AgentState, AgentStatus
from saha.contracts.vendor import (
    BudgetInterruptSignal,
    RequestOptions,
    ToolSchema,
    UnifiedAgentRequest,
    UnifiedAgentResponse,
)
from saha.db.connection import get_pool
from saha.event_bus.client import SAHABusClient
from saha.event_bus import topics
from saha.execution.agent_state import AgentStateManager
from saha.execution.tool_runner import ToolRunner, build_default_tool_runner
from saha.vendor import VendorGateway

logger = logging.getLogger(__name__)

# Max steps before forcing termination (guard against infinite loops)
MAX_STEPS = 50


@dataclass
class TaskSpec:
    """Minimal description of a task passed to the Agent Loop."""
    task_id:          str
    provider_id:      str
    message:          str
    system_prompt:    str              = ""
    scenario_id:      str             = "GENERIC"
    importance:       str             = "NORMAL"   # CRITICAL | NORMAL | LOW
    domain_tags:      list[str]       = None          # type: ignore[assignment]
    tools:            list[ToolSchema] = None         # type: ignore[assignment]
    success_contract: SuccessContract  = None         # type: ignore[assignment]
    options:          RequestOptions   = None         # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.domain_tags is None:
            self.domain_tags = []
        if self.tools is None:
            self.tools = []
        if self.success_contract is None:
            self.success_contract = SuccessContract()
        if self.options is None:
            self.options = RequestOptions()


class AgentLoop:
    """
    Orchestrates the multi-turn execution loop for a single task.
    Spec ref: §3.3 (Agent Execution Loop Behavior Contract).

    Phase 2: accepts optional CostRouter (§4). When present, routing decisions
    are made dynamically; when absent, provider_id from TaskSpec is used directly
    (Phase 1 behaviour, backward-compatible).

    Flow:
    1. Resolve provider via CostRouter (or use TaskSpec.provider_id directly).
    2. Create AgentState in DB.
    3. Loop: build request → call VendorGateway → handle response.
       - COMPLETED → send EvalInput to bus → done.
       - NEEDS_TOOL → execute tool → append to memory → repeat.
       - FAILED → mark state, check EscalationPolicy, publish error trace → done.
    4. Budget is checked every step; interrupt signal sent if exceeded.
    """

    def __init__(
        self,
        gateway:       VendorGateway,
        bus:           SAHABusClient,
        tool_runner:   ToolRunner | None       = None,
        state_manager: AgentStateManager | None = None,
        router:        Any | None              = None,   # CostRouter | None
    ) -> None:
        self._gateway       = gateway
        self._bus           = bus
        self._tool_runner   = tool_runner or build_default_tool_runner()
        self._state_manager = state_manager or AgentStateManager()
        self._router        = router   # injected CostRouter (None → Phase 1 mode)

    async def run(self, spec: TaskSpec) -> AgentState:
        """
        Execute the full agent loop for a task.
        Returns the final AgentState (COMPLETED or FAILED).
        """
        # 1 ─ Resolve provider via Cost Router (Phase 2) or use spec directly (Phase 1)
        provider_id = await self._resolve_provider(spec)

        # 2 ─ Initialise state
        state = await self._state_manager.create(
            task_id=spec.task_id,
            provider_id=provider_id,
            budget_cap=spec.options.budget_cap,
        )
        logger.info(
            "AgentLoop started | task_id=%s agent_state_id=%s provider=%s",
            spec.task_id, state.agent_state_id, provider_id,
        )

        total_tool_calls = 0
        run_id = new_uuid()
        loop_start_ms = int(time.monotonic() * 1000)

        # 2 ─ Main loop
        for step in range(MAX_STEPS):
            state.current_step = step

            # Budget guard
            if state.budget_used >= state.budget_cap:
                await self._handle_budget_exceeded(state, run_id, spec)
                return state

            # Build and send request
            request = self._build_request(spec, state)
            response = await self._gateway.complete(spec.provider_id, request)

            # Update budget
            state.budget_used  += response.cost_estimate
            state.context_tokens_used = response.context_tokens_used
            await self._state_manager.update(state)

            # ── Publish execution trace ──────────────────────────────────────
            await self._publish_execution_trace(
                run_id=run_id,
                state=state,
                response=response,
                request_id=request.request_id,
            )

            # ── Branch on response status ────────────────────────────────────
            if response.status == "COMPLETED":
                state.status = AgentStatus.COMPLETED
                await self._state_manager.mark_completed(state.agent_state_id)

                # Compose and publish EvalInput
                latency_ms = int(time.monotonic() * 1000) - loop_start_ms
                eval_input = self._compose_eval_input(
                    spec=spec,
                    state=state,
                    response=response,
                    run_id=run_id,
                    total_tool_calls=total_tool_calls,
                    latency_ms=latency_ms,
                )
                await self._bus.publish(
                    topics.EVAL_INPUTS,
                    eval_input.to_bus_payload(),
                )
                logger.info(
                    "AgentLoop COMPLETED | task_id=%s steps=%d cost=$%.4f",
                    spec.task_id, step + 1, state.budget_used,
                )
                return state

            elif response.status == "NEEDS_TOOL":
                state.status = AgentStatus.WAITING_FOR_TOOL
                if not response.pending_tool_call:
                    logger.warning("NEEDS_TOOL with no pending_tool_call; treating as FAILED")
                    await self._handle_failure(state, response.error, run_id)
                    return state

                tool_call = response.pending_tool_call
                state.pending_tool_call = tool_call
                await self._state_manager.update(state)

                # Execute tool
                tool_result = await self._tool_runner.run(
                    tool_name=tool_call["name"],
                    arguments=tool_call.get("arguments", {}),
                )
                total_tool_calls += 1

                # Append assistant turn + tool result to short-term memory
                state.memory.short_term.append({
                    "role": "assistant",
                    "content": response.normalized_output.get("text", ""),
                })
                state.memory.short_term.append({
                    "role": "user",
                    "content": tool_result.as_message_content(),
                })
                state.pending_tool_call = None
                state.status = AgentStatus.RUNNING
                await self._state_manager.update(state)

                logger.debug(
                    "Tool '%s' executed at step %d | success=%s",
                    tool_call["name"], step, tool_result.success,
                )

            elif response.status == "FAILED":
                await self._handle_failure(state, response.error, run_id)
                return state

            else:
                logger.error("Unknown response status: %s", response.status)
                await self._handle_failure(
                    state,
                    CanonicalError(
                        type=ErrorType.INFRA_ERROR,
                        code=ErrorCode.UNKNOWN,
                        severity=ErrorSeverity.CRITICAL,
                        details=f"Unknown status: {response.status}",
                    ),
                    run_id,
                )
                return state

        # Exceeded MAX_STEPS
        logger.error("AgentLoop exceeded MAX_STEPS=%d for task_id=%s", MAX_STEPS, spec.task_id)
        await self._handle_failure(
            state,
            CanonicalError(
                type=ErrorType.INFRA_ERROR,
                code=ErrorCode.UNKNOWN,
                severity=ErrorSeverity.CRITICAL,
                details=f"Exceeded maximum steps ({MAX_STEPS})",
            ),
            run_id,
        )
        return state

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_request(self, spec: TaskSpec, state: AgentState) -> UnifiedAgentRequest:
        """Construct UnifiedAgentRequest from current task spec + agent state."""
        # Inject short_term memory as conversation history.
        # We smuggle history via options (Phase 1 convention; Phase 2 will use
        # a dedicated conversation context field once multi-turn API is stable).
        message = spec.message
        if state.memory.short_term:
            # Reconstruct the user message as the last user turn
            user_turns = [
                m["content"]
                for m in state.memory.short_term
                if m.get("role") == "user"
            ]
            if user_turns:
                message = user_turns[-1]

        return UnifiedAgentRequest(
            task_id=spec.task_id,
            agent_state_id=state.agent_state_id,
            message=message,
            system_prompt=spec.system_prompt,
            tools=spec.tools,
            options=spec.options,
        )

    def _compose_eval_input(
        self,
        spec: TaskSpec,
        state: AgentState,
        response: UnifiedAgentResponse,
        run_id: str,
        total_tool_calls: int,
        latency_ms: int,
    ) -> EvalInput:
        from saha.contracts.eval import EvalInput, TaskType
        return EvalInput(
            task_type=getattr(TaskType, spec.domain_tags[0].upper(), TaskType.GENERIC)
                if spec.domain_tags else TaskType.GENERIC,
            scenario_id=spec.scenario_id,
            domain_tags=spec.domain_tags,
            input_normalized={"message": spec.message},
            normalized_output=response.normalized_output,
            success_contract=spec.success_contract,
            provider_info=EvalProviderInfo(
                provider_id=spec.provider_id,
                run_id=run_id,
                raw_output_ref=response.raw_output_ref,
            ),
            context=EvalContext(
                tool_calls_count=total_tool_calls,
                context_tokens_used=state.context_tokens_used,
            ),
        )

    async def _handle_budget_exceeded(
        self, state: AgentState, run_id: str, spec: TaskSpec
    ) -> None:
        signal = BudgetInterruptSignal(
            task_id=state.task_id,
            run_id=run_id,
            provider_id=state.provider_id,
            budget_cap=state.budget_cap,
            budget_used=state.budget_used,
        )
        await self._bus.publish(topics.BUDGET_INTERRUPTS, signal.to_bus_payload())
        await self._gateway.interrupt(state.provider_id, signal)

        state.status = AgentStatus.FAILED
        await self._state_manager.mark_failed(state.agent_state_id)
        logger.warning(
            "Budget exceeded | task_id=%s used=$%.4f cap=$%.2f",
            state.task_id, state.budget_used, state.budget_cap,
        )

    async def _handle_failure(
        self, state: AgentState, error: CanonicalError, run_id: str
    ) -> None:
        state.status = AgentStatus.FAILED
        await self._state_manager.mark_failed(state.agent_state_id)
        logger.error(
            "AgentLoop FAILED | task_id=%s error=%s.%s: %s",
            state.task_id, error.type, error.code, error.details,
        )

    async def _publish_execution_trace(
        self,
        run_id: str,
        state: AgentState,
        response: UnifiedAgentResponse,
        request_id: str,
    ) -> None:
        """Publish lightweight execution trace to Observability (Phase 2 listener)."""
        trace = {
            "run_id": run_id,
            "task_id": state.task_id,
            "agent_state_id": state.agent_state_id,
            "provider_id": state.provider_id,
            "request_id": request_id,
            "latency_ms": response.latency_ms,
            "tool_calls_count": response.tool_calls_count,
            "context_tokens_used": response.context_tokens_used,
            "budget_used": state.budget_used,
            "budget_cap": state.budget_cap,
            "error": response.error.to_bus_payload(),
        }
        # Persist to DB (optional – silently skipped when no DB is available,
        # e.g. during unit / integration tests with mocked state managers)
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO execution_traces
                        (run_id, task_id, agent_state_id, provider_id, request_id,
                         latency_ms, tool_calls_count, context_tokens_used,
                         budget_used, budget_cap, error)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb)
                    ON CONFLICT (run_id) DO NOTHING
                    """,
                    run_id,
                    state.task_id,
                    state.agent_state_id,
                    state.provider_id,
                    request_id,
                    response.latency_ms,
                    response.tool_calls_count,
                    response.context_tokens_used,
                    state.budget_used,
                    state.budget_cap,
                    json.dumps(response.error.to_bus_payload()),
                )
        except Exception as db_exc:
            logger.debug("Execution trace DB persist skipped: %s", db_exc)

    async def _log_routing_decision(self, state: AgentState, spec: TaskSpec) -> None:
        """
        Phase 1 routing stub.
        Logs the hard-coded provider selection to routing_decisions so the table
        is populated and queryable immediately. Phase 2 Cost Router will replace
        this call with a real decision – AgentLoop needs no changes.
        Spec ref: §4.2 (Routing Decision Contract – Phase 1 minimal form)
        """
        decision_id = new_uuid()
        reason = (
            f"Phase 1: provider hard-coded to '{state.provider_id}'. "
            f"Cost Routing deferred to Phase 2."
        )
        payload = json.dumps({
            "scenario_id":  spec.scenario_id,
            "domain_tags":  spec.domain_tags,
            "budget_cap":   spec.options.budget_cap,
            "routing_mode": spec.options.routing_mode,
        })
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO routing_decisions
                        (decision_id, task_id, chosen_provider_id,
                         fallback_provider_id, mode, reason, payload)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                    """,
                    decision_id,
                    state.task_id,
                    state.provider_id,
                    None,                        # no fallback in Phase 1
                    spec.options.routing_mode,
                    reason,
                    payload,
                )
            logger.debug(
                "Routing decision logged | decision_id=%s provider=%s",
                decision_id, state.provider_id,
            )
        except Exception as db_exc:
            logger.debug("Routing decision DB persist skipped: %s", db_exc)

    async def _resolve_provider(self, spec: TaskSpec) -> str:
        """
        Phase 2: delegate provider selection to CostRouter when injected.
        Phase 1 fallback: use spec.provider_id directly (backward-compatible).
        Spec ref: §3.3 step 1, §4.2 (routing decision)
        """
        if self._router is None:
            # Phase 1 mode — use stub for backward compatibility
            await self._log_routing_decision_stub(spec)
            return spec.provider_id

        # Phase 2 mode — ask Cost Router for optimal provider
        try:
            from saha.contracts.routing import TaskProfile
            task_profile = TaskProfile(
                task_id      = spec.task_id,
                task_type    = "generic",
                scenario_id  = spec.scenario_id,
                domain_tags  = spec.domain_tags,
                importance   = spec.importance,
                budget_cap   = spec.options.budget_cap,
                routing_mode = spec.options.routing_mode,
            )
            # Registered providers come from the gateway's adapter registry
            candidate_ids = list(self._gateway._adapters.keys())
            decision = await self._router.decide(task_profile, candidate_ids)
            logger.info(
                "CostRouter decision | chosen=%s fallback=%s cold_start=%s",
                decision.chosen_provider_id,
                decision.fallback_provider_id,
                decision.cold_start,
            )
            return decision.chosen_provider_id
        except Exception as exc:
            logger.warning(
                "CostRouter failed (%s) — falling back to spec.provider_id=%s",
                exc, spec.provider_id,
            )
            return spec.provider_id

    async def _log_routing_decision_stub(self, spec: TaskSpec) -> None:
        """
        Phase 1 compatibility stub — kept so existing tests pass unchanged.
        Called only when no CostRouter is injected.
        """
        decision_id = new_uuid()
        reason = (
            f"Phase 1 stub: provider hard-coded to '{spec.provider_id}'. "
            f"Inject CostRouter for dynamic routing (Phase 2)."
        )
        payload = json.dumps({
            "scenario_id":  spec.scenario_id,
            "domain_tags":  spec.domain_tags,
            "budget_cap":   spec.options.budget_cap,
            "routing_mode": spec.options.routing_mode,
        })
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO routing_decisions
                        (decision_id, task_id, chosen_provider_id,
                         fallback_provider_id, mode, reason, payload)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                    """,
                    decision_id, spec.task_id, spec.provider_id,
                    None, spec.options.routing_mode, reason, payload,
                )
        except Exception as db_exc:
            logger.debug("Routing stub DB persist skipped: %s", db_exc)



