"""
SAHA – Unit tests for LLM-as-Judge (Phase 2, M1).
All tests mock the Anthropic API — no real API key required.

Coverage:
  - LLMJudge.is_applicable() logic
  - LLMJudge.judge() happy path (JSON parsing, field clamping)
  - LLMJudge.judge() graceful fallback on API error
  - LLMJudge._build_prompt() provider neutrality
  - LLMJudge._parse_response() edge cases (malformed JSON, ranges)
  - Grader.grade() with judge_result (weighted blend)
  - Grader.grade() without judge_result (deterministic fallback)
  - JudgeCalibration with small dataset (below MIN_CASES_REQUIRED)
  - GoldenDatasetLoader loading and validation
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saha.contracts.common import TaskType, new_uuid
from saha.contracts.eval import (
    EvalContext,
    EvalInput,
    EvalProviderInfo,
    JudgeResult,
    SuccessContract,
    Verdict,
)
from saha.eval.golden_dataset.loader import GoldenDatasetLoader
from saha.eval.grader import Grader
from saha.eval.judge_calibration import (
    JudgeCalibration,
    MIN_CASES_REQUIRED,
)
from saha.eval.llm_judge import LLMJudge, JUDGE_ENABLED


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_eval_input(
    text: str = "def foo(): return 42",
    custom_rubric: str = "",
    tool_calls: int = 0,
) -> EvalInput:
    return EvalInput(
        task_type=TaskType.CODE_GENERATION,
        scenario_id="SCENARIO_PY_FIX",
        domain_tags=["python"],
        normalized_output={"text": text},
        success_contract=SuccessContract(
            custom_rubric=custom_rubric,
            max_tool_calls=5,
        ),
        provider_info=EvalProviderInfo(provider_id="claude_3_5_sonnet", run_id="r1"),
        context=EvalContext(tool_calls_count=tool_calls),
    )


def _mock_claude_judge(
    quality: int = 90,
    safety: int = 100,
    confidence: int = 85,
    reasoning: str = "Output is correct and clean.",
) -> MagicMock:
    """Build a fake anthropic.types.Message with judge JSON."""
    msg = MagicMock()
    text_block       = MagicMock()
    text_block.text  = json.dumps({
        "quality_score": quality,
        "safety_score":  safety,
        "confidence":    confidence,
        "reasoning":     reasoning,
    })
    msg.content = [text_block]
    return msg


def _patch_judge_client(response_msg: MagicMock) -> Any:
    mock_client   = MagicMock()
    mock_messages = AsyncMock()
    mock_messages.create = AsyncMock(return_value=response_msg)
    mock_client.messages = mock_messages
    return patch(
        "saha.eval.llm_judge.anthropic.AsyncAnthropic",
        return_value=mock_client,
    )


# Fix missing import in helper
from typing import Any


# ─── LLMJudge.is_applicable() ────────────────────────────────────────────────

class TestLLMJudgeApplicability:
    def test_applicable_with_custom_rubric(self) -> None:
        with patch("saha.eval.llm_judge.anthropic.AsyncAnthropic"):
            judge = LLMJudge()
        inp = _make_eval_input(custom_rubric="Rate clarity from 1 to 5")
        assert judge.is_applicable(inp) is True

    def test_not_applicable_without_rubric_and_not_always_on(self) -> None:
        with patch("saha.eval.llm_judge.anthropic.AsyncAnthropic"):
            judge = LLMJudge()
        inp = _make_eval_input(custom_rubric="")
        # JUDGE_ALWAYS_ON defaults to False in test env
        with patch("saha.eval.llm_judge.JUDGE_ALWAYS_ON", False):
            result = judge.is_applicable(inp)
        assert result is False

    def test_always_on_activates_without_rubric(self) -> None:
        with patch("saha.eval.llm_judge.anthropic.AsyncAnthropic"):
            judge = LLMJudge()
        inp = _make_eval_input(custom_rubric="")
        with patch("saha.eval.llm_judge.JUDGE_ALWAYS_ON", True):
            result = judge.is_applicable(inp)
        assert result is True

    def test_disabled_env_overrides_rubric(self) -> None:
        with patch("saha.eval.llm_judge.anthropic.AsyncAnthropic"):
            judge = LLMJudge()
        inp = _make_eval_input(custom_rubric="Rate clarity")
        with patch("saha.eval.llm_judge.JUDGE_ENABLED", False):
            result = judge.is_applicable(inp)
        assert result is False


# ─── LLMJudge.judge() ────────────────────────────────────────────────────────

class TestLLMJudgeHappyPath:
    async def test_returns_judge_result(self) -> None:
        fake_msg = _mock_claude_judge(quality=88, safety=95, confidence=80)
        with _patch_judge_client(fake_msg):
            judge = LLMJudge()
        inp = _make_eval_input(custom_rubric="Is the code correct?")
        with _patch_judge_client(fake_msg):
            judge = LLMJudge()
            with patch("saha.eval.llm_judge.JUDGE_ALWAYS_ON", True):
                result = await judge.judge(inp)
        assert result is not None
        assert result.quality_score == 88
        assert result.safety_score  == 95
        assert result.confidence    == 80
        assert "correct" in result.reasoning.lower() or result.reasoning != ""

    async def test_scores_clamped_to_0_100(self) -> None:
        """Scores > 100 or < 0 must be clamped."""
        msg = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "quality_score": 150,  # too high
            "safety_score":  -10,  # too low
            "confidence":    999,
            "reasoning":     "test",
        })
        msg.content = [text_block]
        with _patch_judge_client(msg):
            judge = LLMJudge()
            with patch("saha.eval.llm_judge.JUDGE_ALWAYS_ON", True):
                result = await judge.judge(_make_eval_input(custom_rubric="x"))
        assert result is not None
        assert result.quality_score <= 100
        assert result.safety_score  >= 0
        assert result.confidence    <= 100

    async def test_judge_result_contains_model_name(self) -> None:
        fake_msg = _mock_claude_judge()
        with _patch_judge_client(fake_msg):
            judge = LLMJudge()
            with patch("saha.eval.llm_judge.JUDGE_ALWAYS_ON", True):
                result = await judge.judge(_make_eval_input(custom_rubric="check"))
        assert result is not None
        assert "claude" in result.judge_model

    async def test_rubric_stored_in_result(self) -> None:
        rubric = "Rate explanation clarity from 1 to 10"
        fake_msg = _mock_claude_judge()
        with _patch_judge_client(fake_msg):
            judge = LLMJudge()
            with patch("saha.eval.llm_judge.JUDGE_ALWAYS_ON", True):
                result = await judge.judge(_make_eval_input(custom_rubric=rubric))
        assert result is not None
        assert result.rubric_used == rubric


class TestLLMJudgeGracefulFallback:
    async def test_api_error_returns_none(self) -> None:
        mock_client   = MagicMock()
        mock_messages = AsyncMock()
        import anthropic
        mock_messages.create = AsyncMock(
            side_effect=anthropic.APIStatusError(
                "Rate limit", response=MagicMock(status_code=429), body={}
            )
        )
        mock_client.messages = mock_messages
        with patch("saha.eval.llm_judge.anthropic.AsyncAnthropic", return_value=mock_client):
            judge = LLMJudge()
            with patch("saha.eval.llm_judge.JUDGE_ALWAYS_ON", True):
                result = await judge.judge(_make_eval_input(custom_rubric="x"))
        assert result is None

    async def test_unexpected_exception_returns_none(self) -> None:
        mock_client   = MagicMock()
        mock_messages = AsyncMock()
        mock_messages.create = AsyncMock(side_effect=RuntimeError("network down"))
        mock_client.messages = mock_messages
        with patch("saha.eval.llm_judge.anthropic.AsyncAnthropic", return_value=mock_client):
            judge = LLMJudge()
            with patch("saha.eval.llm_judge.JUDGE_ALWAYS_ON", True):
                result = await judge.judge(_make_eval_input(custom_rubric="x"))
        assert result is None  # never raises


class TestLLMJudgePromptNeutrality:
    def test_prompt_does_not_contain_provider_id(self) -> None:
        """Judge prompt must be provider-blind (§1.3 neutrality)."""
        with patch("saha.eval.llm_judge.anthropic.AsyncAnthropic"):
            judge = LLMJudge()
        inp = _make_eval_input(custom_rubric="check quality")
        inp.provider_info = EvalProviderInfo(
            provider_id="claude_3_5_sonnet", run_id="r"
        )
        prompt = judge._build_prompt(inp)
        assert "claude_3_5_sonnet" not in prompt
        assert "anthropic" not in prompt.lower()

    def test_prompt_contains_scenario_and_rubric(self) -> None:
        with patch("saha.eval.llm_judge.anthropic.AsyncAnthropic"):
            judge = LLMJudge()
        rubric = "Is the output correct and idiomatic?"
        inp    = _make_eval_input(custom_rubric=rubric)
        prompt = judge._build_prompt(inp)
        assert "SCENARIO_PY_FIX" in prompt
        assert rubric in prompt

    def test_prompt_truncates_long_output(self) -> None:
        with patch("saha.eval.llm_judge.anthropic.AsyncAnthropic"):
            judge = LLMJudge()
        long_text = "x" * 10_000
        inp = _make_eval_input(text=long_text, custom_rubric="check")
        prompt = judge._build_prompt(inp)
        # Output section must be at most 4000 chars
        assert prompt.count("x") <= 4001


class TestLLMJudgeParseResponse:
    def test_parse_valid_json(self) -> None:
        with patch("saha.eval.llm_judge.anthropic.AsyncAnthropic"):
            judge = LLMJudge()
        inp  = _make_eval_input()
        raw  = '{"quality_score":80,"safety_score":90,"reasoning":"ok","confidence":75}'
        res  = judge._parse_response(raw, inp)
        assert res.quality_score == 80
        assert res.safety_score  == 90
        assert res.confidence    == 75

    def test_parse_json_with_markdown_fences(self) -> None:
        with patch("saha.eval.llm_judge.anthropic.AsyncAnthropic"):
            judge = LLMJudge()
        inp = _make_eval_input()
        raw = "```json\n{\"quality_score\":70,\"safety_score\":85,\"reasoning\":\"good\",\"confidence\":60}\n```"
        res = judge._parse_response(raw, inp)
        assert res.quality_score == 70


# ─── Grader + JudgeResult Integration ────────────────────────────────────────

class TestGraderWithJudge:
    def test_blend_raises_quality_with_good_judge(self) -> None:
        """Good judge score should raise the blended quality above deterministic-only."""
        grader = Grader()
        inp    = _make_eval_input(text="def foo(): return 42")
        # Deterministic alone:
        det_result = grader.grade(inp)
        # With judge giving 100:
        judge_result = JudgeResult(quality_score=100, safety_score=100, confidence=90)
        blended_result = grader.grade(inp, judge_result=judge_result)
        assert blended_result.quality_score >= det_result.quality_score
        assert blended_result.grader_breakdown.llm_judge["enabled"] is True

    def test_judge_in_breakdown(self) -> None:
        grader       = Grader()
        inp          = _make_eval_input(text="good output")
        judge_result = JudgeResult(quality_score=85, safety_score=95, confidence=80, reasoning="Nice.")
        result       = grader.grade(inp, judge_result=judge_result)
        jb = result.grader_breakdown.llm_judge
        assert jb["enabled"]       is True
        assert jb["quality_score"] == 85
        assert jb["reasoning"]     == "Nice."
        assert jb["weight"]        == pytest.approx(0.70, rel=1e-3)

    def test_no_judge_shows_not_enabled(self) -> None:
        grader = Grader()
        inp    = _make_eval_input()
        result = grader.grade(inp)
        assert result.grader_breakdown.llm_judge["enabled"] is False

    def test_bad_judge_score_does_not_override_empty_failure(self) -> None:
        """Even a perfect judge score cannot rescue an empty output (deterministic FAILURE)."""
        grader       = Grader()
        inp          = _make_eval_input(text="")   # empty → FAILURE
        judge_result = JudgeResult(quality_score=100, safety_score=100, confidence=99)
        result       = grader.grade(inp, judge_result=judge_result)
        # Blended quality = 0.3*60 + 0.7*100 = 88... but FAILURE is set deterministically
        # The verdict from empty-output check forces FAILURE before blending
        assert result.final_verdict == Verdict.FAILURE


# ─── GoldenDatasetLoader ──────────────────────────────────────────────────────

class TestGoldenDatasetLoader:
    def test_loads_scenario_py_fix(self) -> None:
        """The SCENARIO_PY_FIX.json shipped with the project must load cleanly."""
        loader = GoldenDatasetLoader()
        cases  = loader.load_scenario("SCENARIO_PY_FIX")
        assert len(cases) >= 10
        for case in cases:
            assert "output_text" in case
            assert "ground_truth" in case
            assert "quality_score" in case["ground_truth"]
            assert "safety_score" in case["ground_truth"]

    def test_load_all_returns_list(self) -> None:
        loader = GoldenDatasetLoader()
        cases  = loader.load_all()
        assert isinstance(cases, list)
        assert len(cases) >= 10

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        loader = GoldenDatasetLoader(scenarios_dir=tmp_path / "nonexistent")
        assert loader.load_all() == []

    def test_invalid_json_skipped(self, tmp_path: Path) -> None:
        bad = tmp_path / "BAD_SCENARIO.json"
        bad.write_text("not valid json", encoding="utf-8")
        loader = GoldenDatasetLoader(scenarios_dir=tmp_path)
        cases  = loader.load_all()
        assert cases == []

    def test_missing_ground_truth_skipped(self, tmp_path: Path) -> None:
        data = [{"output_text": "def foo(): pass"}]  # missing ground_truth
        (tmp_path / "MISSING_GT.json").write_text(json.dumps(data), encoding="utf-8")
        loader = GoldenDatasetLoader(scenarios_dir=tmp_path)
        assert loader.load_all() == []


# ─── JudgeCalibration ────────────────────────────────────────────────────────

class TestJudgeCalibrationInsufficientData:
    async def test_below_min_cases_returns_early(self, tmp_path: Path) -> None:
        """With < MIN_CASES_REQUIRED cases, calibration returns immediately without calling judge."""
        # Create a tiny dataset (3 cases)
        tiny = [
            {
                "case_id": f"c{i}",
                "output_text": "def foo(): pass",
                "ground_truth": {"quality_score": 80, "safety_score": 100},
                "custom_rubric": "check",
            }
            for i in range(3)
        ]
        (tmp_path / "TINY.json").write_text(json.dumps(tiny), encoding="utf-8")

        mock_judge = MagicMock(spec=LLMJudge)
        mock_judge.judge = AsyncMock()   # should NOT be called

        calibrator = JudgeCalibration(
            judge=mock_judge,
            loader=GoldenDatasetLoader(scenarios_dir=tmp_path),
        )
        report = await calibrator.run()

        assert report.judge_enabled is True
        assert "Insufficient" in report.recommendation
        mock_judge.judge.assert_not_called()

    async def test_calibration_with_accurate_judge_passes(self, tmp_path: Path) -> None:
        """Judge within threshold → judge_enabled=True, PASSED recommendation."""
        # Build MIN_CASES_REQUIRED cases where judge agrees with ground truth
        cases = [
            {
                "case_id": f"c{i}",
                "output_text": "def foo(): return 42",
                "ground_truth": {"quality_score": 80, "safety_score": 100},
                "custom_rubric": "Is it correct?",
            }
            for i in range(MIN_CASES_REQUIRED)
        ]
        (tmp_path / "ACCURATE.json").write_text(json.dumps(cases), encoding="utf-8")

        # Mock judge always returns scores close to ground truth (deviation ≈ 2pp)
        mock_judge = MagicMock(spec=LLMJudge)
        mock_judge.judge = AsyncMock(return_value=JudgeResult(
            quality_score=82, safety_score=100, confidence=80, reasoning="Good."
        ))

        calibrator = JudgeCalibration(
            judge=mock_judge,
            loader=GoldenDatasetLoader(scenarios_dir=tmp_path),
            deviation_threshold=5.0,
        )
        report = await calibrator.run()
        assert report.judge_enabled is True
        assert "PASSED" in report.recommendation or "\u2705" in report.recommendation

    async def test_calibration_with_inaccurate_judge_disables(self, tmp_path: Path) -> None:
        """Judge deviation > threshold → judge_enabled=False."""
        cases = [
            {
                "case_id": f"c{i}",
                "output_text": "def foo(): pass",
                "ground_truth": {"quality_score": 80, "safety_score": 100},
                "custom_rubric": "check quality",
            }
            for i in range(MIN_CASES_REQUIRED)
        ]
        (tmp_path / "INACCURATE.json").write_text(json.dumps(cases), encoding="utf-8")

        # Mock judge always deviates by 20pp (way above threshold)
        mock_judge = MagicMock(spec=LLMJudge)
        mock_judge.judge = AsyncMock(return_value=JudgeResult(
            quality_score=60, safety_score=80, confidence=50, reasoning="Nope."
        ))

        calibrator = JudgeCalibration(
            judge=mock_judge,
            loader=GoldenDatasetLoader(scenarios_dir=tmp_path),
            deviation_threshold=5.0,
        )
        report = await calibrator.run()
        assert report.judge_enabled is False
        assert "\u26a0" in report.recommendation or "DISABLED" in report.recommendation
