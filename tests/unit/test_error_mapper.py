"""
SAHA – Unit tests for ErrorMapper (§2.5).
"""
import httpx
import pytest

from saha.contracts.common import ErrorCode, ErrorSeverity, ErrorType
from saha.vendor.error_mapper import ErrorMapper


class TestErrorMapperFromException:
    def test_timeout(self) -> None:
        exc = httpx.ReadTimeout("timed out", request=None)
        err = ErrorMapper.from_exception(exc, provider_id="claude")
        assert err.type == ErrorType.INFRA_ERROR
        assert err.code == ErrorCode.PROVIDER_UNAVAILABLE
        assert err.severity == ErrorSeverity.WARNING
        assert "claude" in err.details

    def test_rate_limit_by_name(self) -> None:
        class FakeRateLimitError(Exception):
            pass
        FakeRateLimitError.__name__ = "RateLimitError"
        err = ErrorMapper.from_exception(FakeRateLimitError("429"), provider_id="claude")
        assert err.type == ErrorType.INFRA_ERROR
        assert err.code == ErrorCode.PROVIDER_RATE_LIMIT

    def test_unknown_exception(self) -> None:
        err = ErrorMapper.from_exception(RuntimeError("boom"), provider_id="test")
        assert err.type == ErrorType.INFRA_ERROR
        assert err.code == ErrorCode.UNKNOWN
        assert err.severity == ErrorSeverity.WARNING

    def test_no_provider_id(self) -> None:
        err = ErrorMapper.from_exception(ValueError("err"))
        assert err.type == ErrorType.INFRA_ERROR
        assert "[" not in err.details or "[]" not in err.details  # no empty prefix


class TestErrorMapperFromHttp:
    def test_429(self) -> None:
        err = ErrorMapper.from_http_status(429)
        assert err.code == ErrorCode.PROVIDER_RATE_LIMIT

    def test_401(self) -> None:
        err = ErrorMapper.from_http_status(401)
        assert err.type == ErrorType.POLICY_ERROR
        assert err.severity == ErrorSeverity.CRITICAL

    def test_403(self) -> None:
        err = ErrorMapper.from_http_status(403)
        assert err.type == ErrorType.POLICY_ERROR

    def test_500(self) -> None:
        err = ErrorMapper.from_http_status(503)
        assert err.type == ErrorType.INFRA_ERROR
        assert err.code == ErrorCode.PROVIDER_UNAVAILABLE

    def test_unknown_status(self) -> None:
        err = ErrorMapper.from_http_status(418)
        assert err.code == ErrorCode.UNKNOWN


class TestBudgetExceeded:
    def test_budget_exceeded(self) -> None:
        err = ErrorMapper.budget_exceeded(budget_cap=5.0, budget_used=5.05)
        assert err.type == ErrorType.POLICY_ERROR
        assert err.code == ErrorCode.BUDGET_EXCEEDED
        assert err.severity == ErrorSeverity.CRITICAL
        assert "5.00" in err.details
