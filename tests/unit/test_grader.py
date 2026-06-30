"""
SAHA – Unit tests for Grader (§1.2–1.3).
Covers all deterministic checks and verdict logic.
"""
import pytest

from saha.contracts.eval import (
    EvalContext,
    EvalInput,
    EvalProviderInfo,
    SuccessContract,
    Verdict,
)
from saha.contracts.common import TaskType
from saha.eval.grader import Grader


@pytest.fixture
def grader() -> Grader:
    return Grader()


def make_eval_input(
    text: str = "This is a valid response.",
    tool_calls: int = 2,
    max_tool_calls: int = 10,
    allowed_error_types: list[str] | None = None,
) -> EvalInput:
    return EvalInput(
        task_type=TaskType.CODE_GENERATION,
        scenario_id="TEST_SCENARIO",
        domain_tags=["python"],
        input_normalized={"message": "Write a function"},
        normalized_output={"text": text},
        success_contract=SuccessContract(
            max_tool_calls=max_tool_calls,
            allowed_error_types=allowed_error_types or ["NONE"],
        ),
        provider_info=EvalProviderInfo(
            provider_id="claude_3_5_sonnet",
            run_id="run-test-1",
        ),
        context=EvalContext(
            tool_calls_count=tool_calls,
            context_tokens_used=3000,
        ),
    )


class TestGraderSuccess:
    def test_clean_output_is_success(self, grader: Grader) -> None:
        inp = make_eval_input(text="Here is the Python function: def foo(): return 42")
        result = grader.grade(inp)
        assert result.final_verdict == Verdict.SUCCESS
        assert result.quality_score >= 80
        assert result.safety_score >= 80
        assert result.grader_confidence > 0

    def test_breakdown_has_all_checks(self, grader: Grader) -> None:
        inp = make_eval_input()
        result = grader.grade(inp)
        check_names = {c["check"] for c in result.grader_breakdown.deterministic_checks}
        assert "max_tool_calls" in check_names
        assert "no_refusal" in check_names
        assert "no_hallucination_signals" in check_names
        assert "non_empty_output" in check_names

    def test_latency_passed_through(self, grader: Grader) -> None:
        inp = make_eval_input()
        result = grader.grade(inp, latency_ms=1234)
        assert result.latency_ms == 1234


class TestGraderToolCallExceeded:
    def test_too_many_tool_calls_reduces_quality(self, grader: Grader) -> None:
        inp = make_eval_input(text="Done.", tool_calls=15, max_tool_calls=5)
        result = grader.grade(inp)
        # Quality penalty applied
        assert result.quality_score < 100
        # Find the check
        tc_check = next(
            c for c in result.grader_breakdown.deterministic_checks
            if c["check"] == "max_tool_calls"
        )
        assert tc_check["passed"] is False


class TestGraderRefusal:
    def test_refusal_reduces_scores(self, grader: Grader) -> None:
        inp = make_eval_input(text="I cannot help with that request as an AI.")
        result = grader.grade(inp)
        assert result.quality_score < 100
        assert result.safety_score < 100

    def test_refusal_check_not_passed(self, grader: Grader) -> None:
        inp = make_eval_input(text="i'm not able to complete this task")
        result = grader.grade(inp)
        refusal_check = next(
            c for c in result.grader_breakdown.deterministic_checks
            if c["check"] == "no_refusal"
        )
        assert refusal_check["passed"] is False


class TestGraderEmptyOutput:
    def test_empty_output_is_failure(self, grader: Grader) -> None:
        inp = make_eval_input(text="")
        result = grader.grade(inp)
        assert result.final_verdict == Verdict.FAILURE
        empty_check = next(
            c for c in result.grader_breakdown.deterministic_checks
            if c["check"] == "non_empty_output"
        )
        assert empty_check["passed"] is False


class TestGraderHallucination:
    def test_hallucination_signal_detected(self, grader: Grader) -> None:
        inp = make_eval_input(
            text="As of my knowledge cutoff, the stock price was $100."
        )
        result = grader.grade(inp)
        h_check = next(
            c for c in result.grader_breakdown.deterministic_checks
            if c["check"] == "no_hallucination_signals"
        )
        assert h_check["passed"] is False
        assert result.quality_score < 100


class TestGraderLLMJudgeStub:
    def test_llm_judge_is_stubbed(self, grader: Grader) -> None:
        inp = make_eval_input()
        result = grader.grade(inp)
        assert result.grader_breakdown.llm_judge["enabled"] is False
        assert result.grader_breakdown.llm_judge["confidence"] == 0
