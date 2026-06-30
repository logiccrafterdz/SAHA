"""
SAHA – Unit tests for Phase 2 M4 components:
  - OpenAIAdapter (GPT-4o): tool format, response parsing, error handling
  - GeminiAdapter: profile load, response parsing, error handling
  - PrivacyGate: PII detection, redaction, residency, output scan, blocking mode
  - HITLService: all four §6.2 workflows, override propagation, triage lifecycle

No real API calls — all SDK clients mocked.
"""
from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saha.contracts.common import new_uuid
from saha.privacy.gate import PIIDetection, PrivacyGate
from saha.hitl.service import HITLService
from saha.routing.constraints import ConstraintManager


# ═══════════════════════════════════════════════════════════════════════════════
# PrivacyGate
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrivacyGateDetection:
    def test_clean_text_no_detection(self) -> None:
        gate = PrivacyGate(redact=True)
        texts, detection = gate.check_request(
            "Fix the bug in calculate_average()", "", "claude_3_5_sonnet"
        )
        assert detection is None
        assert texts["message"] == "Fix the bug in calculate_average()"

    def test_email_detected_and_redacted(self) -> None:
        gate = PrivacyGate(redact=True)
        texts, detection = gate.check_request(
            "Send report to john.doe@example.com please", "", "claude_3_5_sonnet"
        )
        assert detection is not None
        assert detection.found is True
        assert "EMAIL" in detection.types
        assert "john.doe@example.com" not in texts["message"]
        assert "[REDACTED_EMAIL]" in texts["message"]

    def test_ssn_detected_critical_severity(self) -> None:
        gate = PrivacyGate(redact=True)
        _, detection = gate.check_request(
            "User SSN is 123-45-6789", "", "gpt_4o"
        )
        assert detection is not None
        assert "SSN" in detection.types
        assert "CRITICAL" in detection.severities

    def test_api_key_detected(self) -> None:
        gate = PrivacyGate(redact=True)
        _, detection = gate.check_request(
            "My key is sk-abcdefghij1234567890XYZ", "", "claude_3_5_sonnet"
        )
        assert detection is not None
        assert "API_KEY" in detection.types

    def test_credit_card_detected(self) -> None:
        gate = PrivacyGate(redact=True)
        _, detection = gate.check_request(
            "Card: 4111 1111 1111 1111", "", "claude_3_5_sonnet"
        )
        assert detection is not None
        assert "CREDIT_CARD" in detection.types

    def test_multiple_pii_types_in_one_request(self) -> None:
        gate = PrivacyGate(redact=True)
        msg  = "Email: bob@test.com, Phone: 555-123-4567, SSN: 987-65-4321"
        _, detection = gate.check_request(msg, "", "gpt_4o")
        assert detection is not None
        assert len(detection.types) >= 2

    def test_no_pii_returns_none_detection(self) -> None:
        gate = PrivacyGate(redact=True)
        _, detection = gate.check_request("Hello, world!", "", "claude_3_5_sonnet")
        assert detection is None

    def test_system_prompt_also_scanned(self) -> None:
        gate = PrivacyGate(redact=True)
        texts, detection = gate.check_request(
            "Clean message",
            "Secret key: sk-abcdefghij1234567890XYZ",  # PII in system prompt
            "claude_3_5_sonnet",
        )
        assert detection is not None
        assert "API_KEY" in detection.types
        assert "[REDACTED_API_KEY]" in texts["system"]


class TestPrivacyGateBlockingMode:
    def test_blocking_mode_preserves_original_text(self) -> None:
        """In blocking mode, original text is returned (caller must reject)."""
        gate = PrivacyGate(redact=False)
        original = "Email: bad@test.com"
        texts, detection = gate.check_request(original, "", "claude_3_5_sonnet")
        assert detection is not None
        assert detection.found is True
        # Original text preserved (not redacted)
        assert texts["message"] == original

    def test_policy_error_built_correctly(self) -> None:
        from saha.contracts.common import ErrorType, ErrorCode, ErrorSeverity
        gate      = PrivacyGate(redact=False)
        detection = PIIDetection(found=True, types=["SSN"], severities=["CRITICAL"])
        err       = gate.build_policy_error(detection, "gpt_4o")
        assert err.type     == ErrorType.POLICY_ERROR
        assert err.code     == ErrorCode.SAFETY_POLICY_VIOLATION
        assert err.severity == ErrorSeverity.CRITICAL
        assert "SSN" in err.details
        assert "gpt_4o" in err.details



class TestPrivacyGateResidency:
    def test_eu_residency_blocks_gemini(self) -> None:
        gate = PrivacyGate(redact=True)
        _, detection = gate.check_request(
            "Clean message", "", "gemini_1_5_pro", residency=["eu"]
        )
        assert detection is not None
        assert "RESIDENCY_VIOLATION" in detection.types

    def test_global_residency_allows_all(self) -> None:
        gate = PrivacyGate(redact=True)
        for provider in ["claude_3_5_sonnet", "gpt_4o", "gemini_1_5_pro"]:
            _, detection = gate.check_request(
                "Clean message", "", provider, residency=["global"]
            )
            assert detection is None, f"Expected None for {provider}"

    def test_us_residency_allows_all_three(self) -> None:
        gate = PrivacyGate(redact=True)
        for provider in ["claude_3_5_sonnet", "gpt_4o", "gemini_1_5_pro"]:
            _, detection = gate.check_request(
                "Clean", "", provider, residency=["us"]
            )
            assert detection is None


class TestPrivacyGateOutputScan:
    def test_output_scan_detects_leaked_email(self) -> None:
        gate = PrivacyGate(redact=True)
        detection = gate.check_output("The user's email is leaked@example.com in the response")
        assert detection is not None
        assert "EMAIL" in detection.types

    def test_output_scan_clean_returns_none(self) -> None:
        gate = PrivacyGate(redact=True)
        detection = gate.check_output("The average is 42.5 and the function works correctly.")
        assert detection is None

    def test_audit_log_shape(self) -> None:
        gate      = PrivacyGate()
        detection = PIIDetection(found=True, types=["EMAIL"], severities=["HIGH"])
        log       = gate.audit_log("task-1", "claude_3_5_sonnet", detection, "request")
        assert log["task_id"]   == "task-1"
        assert log["pii_found"] is True
        assert "EMAIL" in log["pii_types"]


# ═══════════════════════════════════════════════════════════════════════════════
# HITLService
# ═══════════════════════════════════════════════════════════════════════════════

class TestHITLServiceOverrides:
    async def test_apply_override_propagates_to_constraints(self) -> None:
        cm      = ConstraintManager()
        service = HITLService(constraint_manager=cm)
        override = await service.apply_router_override(
            scope       = "global",
            change      = {"quality_min": 50},
            reason      = "Testing",
            approved_by = "test_user",
        )
        assert override.scope == "global"
        assert override.change["quality_min"] == 50
        # Verify applied to ConstraintManager
        c = cm.get_constraints("conservative")
        assert c.quality_min == 50

    async def test_multiple_overrides_stack(self) -> None:
        cm      = ConstraintManager()
        service = HITLService(constraint_manager=cm)
        await service.apply_router_override("global", {"quality_min": 60}, "r1", "u1")
        await service.apply_router_override("global", {"safety_min":  85}, "r2", "u1")
        c = cm.get_constraints("conservative")
        assert c.quality_min == 60
        assert c.safety_min  == 85

    async def test_clear_override_restores_defaults(self) -> None:
        cm      = ConstraintManager()
        service = HITLService(constraint_manager=cm)
        await service.apply_router_override("global", {"quality_min": 50}, "r", "u")
        service.clear_override("global", "u")
        c = cm.get_constraints("conservative")
        assert c.quality_min == 80   # back to default

    async def test_get_active_overrides(self) -> None:
        service = HITLService()
        await service.apply_router_override("global",    {"quality_min": 70}, "r1", "u")
        await service.apply_router_override("scenario_X", {"safety_min": 80}, "r2", "u")
        assert len(service.get_active_overrides("global"))     == 1
        assert len(service.get_active_overrides("scenario_X")) == 1


class TestHITLServiceContracts:
    async def test_update_contract_recorded(self) -> None:
        service = HITLService()
        update  = await service.update_success_contract(
            scenario_id  = "SCENARIO_PY_FIX",
            old_contract = {"min_quality": 80},
            new_contract = {"min_quality": 85, "custom_rubric": "Checks edge cases"},
            reason       = "False negatives on refusal cases",
            approved_by  = "lead_engineer",
        )
        assert update.scenario_id  == "SCENARIO_PY_FIX"
        assert update.approved_by  == "lead_engineer"

    async def test_get_latest_contract(self) -> None:
        service = HITLService()
        await service.update_success_contract("S1", {"v": 1}, {"v": 2}, "r", "u")
        await service.update_success_contract("S1", {"v": 2}, {"v": 3}, "r", "u")
        latest = service.get_latest_contract("S1")
        assert latest is not None
        assert latest.new_contract["v"] == 3

    async def test_get_contract_missing_scenario_returns_none(self) -> None:
        service = HITLService()
        assert service.get_latest_contract("NONEXISTENT") is None


class TestHITLServiceTriage:
    async def test_record_triage_creates_incident(self) -> None:
        service = HITLService()
        triage  = await service.record_failure_triage(
            run_id          = "run-001",
            classified_root = "MODEL_HALLUCINATION",
            notes           = "Fabricated citation",
            action_items    = ["Tighten grader", "Add golden test case"],
        )
        assert triage.classified_root == "MODEL_HALLUCINATION"
        assert len(triage.action_items) == 2
        assert triage.resolved is False

    async def test_open_triages_excludes_resolved(self) -> None:
        service = HITLService()
        t1 = await service.record_failure_triage("run-1", "BAD_PROMPT", "", [])
        await service.record_failure_triage("run-2", "INFRA", "", [])
        service.resolve_triage(t1.incident_id)
        open_t = service.get_open_triages()
        assert len(open_t) == 1
        assert open_t[0].classified_root == "INFRA"

    async def test_resolve_unknown_incident_returns_false(self) -> None:
        service = HITLService()
        result  = service.resolve_triage("nonexistent-id")
        assert result is False

    async def test_bus_event_published_on_triage(self) -> None:
        mock_bus        = MagicMock()
        mock_bus.publish = AsyncMock()
        service = HITLService(bus=mock_bus)
        await service.record_failure_triage("run-99", "INFRA", "", [])
        mock_bus.publish.assert_called_once()
        call_args = mock_bus.publish.call_args
        assert call_args[0][0] == "SAHA/anomaly_alerts"


class TestHITLServiceCalibration:
    async def test_calibration_trigger_publishes_event(self) -> None:
        mock_bus        = MagicMock()
        mock_bus.publish = AsyncMock()
        service = HITLService(bus=mock_bus)
        result  = await service.trigger_judge_calibration(
            scenario_filter = ["SCENARIO_PY_FIX"],
            approved_by     = "test_user",
        )
        assert result["status"] == "requested"
        assert result["approved_by"] == "test_user"
        mock_bus.publish.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAIAdapter (unit — SDK mocked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpenAIAdapterToolFormat:
    def test_build_tools_empty(self) -> None:
        from saha.vendor.adapters.openai import OpenAIAdapter
        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        result  = adapter._build_tools_from_saha([])
        assert result == []

    def test_build_tools_converts_to_function_format(self) -> None:
        from saha.vendor.adapters.openai import OpenAIAdapter
        from saha.contracts.vendor import ToolSchema
        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        tool = ToolSchema(
            name="run_tests",
            input_schema={
                "description": "Runs the test suite",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        )
        result = adapter._build_tools_from_saha([tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "run_tests"
        assert "parameters" in result[0]["function"]


class TestOpenAIAdapterParsing:
    def _make_completion_mock(self, content: str = "Hello", tool_calls=None) -> MagicMock:
        choice  = MagicMock()
        message = MagicMock()
        message.content    = content
        message.tool_calls = tool_calls or []
        choice.message      = message
        choice.finish_reason = "stop"
        usage = MagicMock()
        usage.prompt_tokens     = 100
        usage.completion_tokens = 50
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage   = usage
        return resp

    def test_parse_text_response(self) -> None:
        from saha.vendor.adapters.openai import OpenAIAdapter
        adapter  = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter._profile = MagicMock()
        mock_resp = self._make_completion_mock(content="Fixed the bug!")
        result = adapter._parse_response(mock_resp, "req-1", "run-1", 500)
        assert result.status == "COMPLETED"
        assert result.normalized_output["text"] == "Fixed the bug!"
        assert result.tool_calls_count == 0

    def test_parse_tool_call_response(self) -> None:
        from saha.vendor.adapters.openai import OpenAIAdapter
        adapter  = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter._profile = MagicMock()

        tc = MagicMock()
        tc.id = "call_123"
        tc.function.name      = "run_tests"
        tc.function.arguments = '{"path": "./tests"}'

        mock_resp = self._make_completion_mock(tool_calls=[tc])
        result = adapter._parse_response(mock_resp, "req-1", "run-1", 300)
        assert result.status == "NEEDS_TOOL"
        assert result.pending_tool_call["name"] == "run_tests"
        assert result.pending_tool_call["arguments"]["path"] == "./tests"

    def test_cost_calculation(self) -> None:
        from saha.vendor.adapters.openai import OpenAIAdapter, _INPUT_COST_PER_TOKEN, _OUTPUT_COST_PER_TOKEN
        adapter  = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter._profile = MagicMock()
        mock_resp = self._make_completion_mock()
        result = adapter._parse_response(mock_resp, "r", "r", 100)
        expected_cost = 100 * _INPUT_COST_PER_TOKEN + 50 * _OUTPUT_COST_PER_TOKEN
        assert result.cost_estimate == pytest.approx(expected_cost, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# GeminiAdapter (unit — SDK mocked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeminiAdapterParsing:
    def _make_gemini_response(self, text: str = "Result", has_fc: bool = False) -> MagicMock:
        part = MagicMock()
        if has_fc:
            fc = MagicMock()
            fc.name = "execute_code"
            fc.args = {"code": "print('hi')"}
            part.function_call = fc
            part.text          = None
        else:
            part.function_call = None
            part.text          = text

        content = MagicMock()
        content.parts = [part]

        candidate = MagicMock()
        candidate.content       = content
        candidate.finish_reason = "STOP"

        usage = MagicMock()
        usage.prompt_token_count     = 80
        usage.candidates_token_count = 30

        resp = MagicMock()
        resp.candidates    = [candidate]
        resp.usage_metadata = usage
        return resp

    def test_parse_text_response(self) -> None:
        from saha.vendor.adapters.gemini import GeminiAdapter
        adapter = GeminiAdapter.__new__(GeminiAdapter)
        adapter._profile = MagicMock()
        mock_resp = self._make_gemini_response(text="Answer here")
        result = adapter._parse_response(mock_resp, "req-1", "run-1", 700)
        assert result.status == "COMPLETED"
        assert result.normalized_output["text"] == "Answer here"

    def test_parse_function_call_response(self) -> None:
        from saha.vendor.adapters.gemini import GeminiAdapter
        adapter = GeminiAdapter.__new__(GeminiAdapter)
        adapter._profile = MagicMock()
        mock_resp = self._make_gemini_response(has_fc=True)
        result = adapter._parse_response(mock_resp, "req-1", "run-1", 400)
        assert result.status == "NEEDS_TOOL"
        assert result.pending_tool_call["name"] == "execute_code"

    def test_empty_candidates_returns_failed(self) -> None:
        from saha.vendor.adapters.gemini import GeminiAdapter
        adapter = GeminiAdapter.__new__(GeminiAdapter)
        adapter._profile = MagicMock()
        resp = MagicMock()
        resp.candidates    = []
        resp.usage_metadata = None
        result = adapter._parse_response(resp, "r", "r", 100)
        assert result.status == "FAILED"

    def test_cost_calculation_cheaper_than_claude(self) -> None:
        from saha.vendor.adapters.gemini import GeminiAdapter, _INPUT_COST_PER_TOKEN, _OUTPUT_COST_PER_TOKEN
        from saha.vendor.adapters.claude import _PROFILE_PATH  # noqa: F401 (just checking import)
        adapter = GeminiAdapter.__new__(GeminiAdapter)
        adapter._profile = MagicMock()
        mock_resp = self._make_gemini_response()
        result = adapter._parse_response(mock_resp, "r", "r", 100)
        expected = 80 * _INPUT_COST_PER_TOKEN + 30 * _OUTPUT_COST_PER_TOKEN
        assert result.cost_estimate == pytest.approx(expected, rel=1e-6)
        # Gemini must be cheaper per token than Claude ($3/$15)
        assert _INPUT_COST_PER_TOKEN < 3.0 / 1_000_000
