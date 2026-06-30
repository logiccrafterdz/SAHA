"""
SAHA – Unit tests for contracts (Pydantic models).
Verifies serialisation, defaults, and canonical error construction.
"""
import pytest
from saha.contracts.common import (
    CanonicalError,
    ErrorCode,
    ErrorSeverity,
    ErrorType,
    new_uuid,
)
from saha.contracts.eval import EvalInput, EvalResult, SuccessContract, Verdict
from saha.contracts.execution import AgentMemory, AgentState, AgentStatus
from saha.contracts.vendor import (
    BudgetInterruptSignal,
    ProviderCapabilities,
    ProviderProfile,
    RequestOptions,
    UnifiedAgentRequest,
    UnifiedAgentResponse,
)


class TestCanonicalError:
    def test_none_factory(self) -> None:
        err = CanonicalError.none()
        assert err.type == ErrorType.NONE
        assert err.code == ErrorCode.NONE
        assert err.severity == ErrorSeverity.INFO

    def test_critical_factory(self) -> None:
        err = CanonicalError.critical(
            type=ErrorType.MODEL_ERROR,
            code=ErrorCode.HALLUCINATION,
            details="test",
        )
        assert err.severity == ErrorSeverity.CRITICAL
        assert err.type == ErrorType.MODEL_ERROR

    def test_bus_payload_is_dict(self) -> None:
        err = CanonicalError.none()
        payload = err.to_bus_payload()
        assert isinstance(payload, dict)
        assert "type" in payload


class TestUnifiedAgentRequest:
    def test_defaults(self) -> None:
        req = UnifiedAgentRequest(
            task_id="t1",
            agent_state_id="a1",
            message="hello",
        )
        assert req.request_id  # auto-generated UUID
        assert req.options.budget_cap == 5.00
        assert req.tools == []

    def test_serialise_round_trip(self) -> None:
        req = UnifiedAgentRequest(
            task_id="t1",
            agent_state_id="a1",
            message="test",
        )
        data = req.to_bus_payload()
        req2 = UnifiedAgentRequest(**data)
        assert req2.task_id == req.task_id
        assert req2.message == req.message


class TestUnifiedAgentResponse:
    def test_status_completed(self) -> None:
        resp = UnifiedAgentResponse(
            request_id=new_uuid(),
            provider_id="claude_3_5_sonnet",
            status="COMPLETED",
            normalized_output={"text": "done"},
        )
        assert resp.status == "COMPLETED"
        assert resp.error.type == ErrorType.NONE


class TestAgentState:
    def test_initial_state(self) -> None:
        state = AgentState(
            task_id="task-1",
            provider_id="claude_3_5_sonnet",
        )
        assert state.current_step == 0
        assert state.status == AgentStatus.RUNNING
        assert state.budget_used == 0.0
        assert state.memory.short_term == []

    def test_bus_payload(self) -> None:
        state = AgentState(task_id="t", provider_id="p")
        payload = state.to_bus_payload()
        assert payload["task_id"] == "t"
        assert "memory" in payload


class TestSuccessContract:
    def test_defaults(self) -> None:
        sc = SuccessContract()
        assert sc.max_tool_calls == 20
        assert "NONE" in sc.allowed_error_types

    def test_custom(self) -> None:
        sc = SuccessContract(
            must_pass_tests=True,
            must_pass_linter=True,
            max_tool_calls=5,
            allowed_error_types=["NONE"],
        )
        assert sc.must_pass_tests is True
        assert sc.max_tool_calls == 5


class TestEvalResult:
    def test_defaults(self) -> None:
        result = EvalResult(
            scenario_id="SCENARIO_1",
            final_verdict=Verdict.SUCCESS,
        )
        assert result.quality_score == 0  # must be set by grader
        assert result.safety_score == 100

    def test_serialise(self) -> None:
        result = EvalResult(scenario_id="S1", final_verdict=Verdict.PARTIAL)
        data = result.to_bus_payload()
        assert data["final_verdict"] == "PARTIAL"


class TestBudgetInterruptSignal:
    def test_fields(self) -> None:
        sig = BudgetInterruptSignal(
            task_id="t1",
            run_id="r1",
            provider_id="claude_3_5_sonnet",
            budget_cap=5.00,
            budget_used=5.02,
        )
        assert sig.reason == "BUDGET_CAP_REACHED"
        assert sig.budget_used > sig.budget_cap
