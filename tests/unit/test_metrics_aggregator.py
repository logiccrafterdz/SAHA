"""
SAHA – Unit tests for MetricsAggregator (Phase 2, M2).
Uses in-memory data (no DB required) to test aggregation logic.
"""
from __future__ import annotations

import pytest
from saha.observability.metrics import _percentile, ProviderWindowStats


class TestPercentileHelper:
    def test_p50_single_value(self) -> None:
        assert _percentile([42], 50) == pytest.approx(42.0)

    def test_p50_even_list(self) -> None:
        result = _percentile([10, 20, 30, 40], 50)
        assert result == pytest.approx(25.0)

    def test_p90_of_10_values(self) -> None:
        data   = list(range(1, 11))  # 1..10
        result = _percentile(data, 90)
        assert result == pytest.approx(9.1)

    def test_empty_returns_zero(self) -> None:
        assert _percentile([], 50) == 0.0

    def test_p100_is_max(self) -> None:
        data = [5, 15, 25, 35, 100]
        assert _percentile(data, 100) == pytest.approx(100.0)

    def test_p0_is_min(self) -> None:
        data = [5, 15, 25]
        assert _percentile(data, 0) == pytest.approx(5.0)


class TestProviderWindowStats:
    def test_to_dict_has_all_keys(self) -> None:
        stats = ProviderWindowStats(
            provider_id    = "claude_3_5_sonnet",
            scenario_id    = "SCENARIO_PY_FIX",
            window         = "7d",
            quality_p50    = 87.5,
            quality_p90    = 95.0,
            safety_avg     = 98.0,
            success_rate   = 0.92,
            error_rate     = 0.08,
            cost_per_task  = 0.004200,
            latency_p50_ms = 1200,
            latency_p90_ms = 2500,
            sample_count   = 50,
        )
        d = stats.to_dict()
        assert d["provider_id"]    == "claude_3_5_sonnet"
        assert d["quality_p50"]    == pytest.approx(87.5, rel=1e-3)
        assert d["success_rate"]   == pytest.approx(0.92, rel=1e-3)
        assert d["sample_count"]   == 50
        assert "computed_at" in d

    def test_rounded_cost(self) -> None:
        stats = ProviderWindowStats(
            provider_id="p", scenario_id="s", window="24h",
            cost_per_task=0.00420012345,
        )
        d = stats.to_dict()
        # Cost should be rounded to 6 decimal places
        assert str(d["cost_per_task"]).count(".") == 1

    def test_default_window_stats_are_zero(self) -> None:
        stats = ProviderWindowStats(provider_id="p", scenario_id="s", window="30d")
        d = stats.to_dict()
        assert d["quality_p50"]  == 0.0
        assert d["sample_count"] == 0


class TestMetricsAggregatorWithoutDB:
    """
    Tests that MetricsAggregator degrades gracefully when no DB is available.
    All methods should return empty lists/dicts, not raise exceptions.
    """

    async def test_get_provider_stats_no_db(self) -> None:
        from saha.observability.metrics import MetricsAggregator
        agg    = MetricsAggregator()
        result = await agg.get_provider_stats("claude_3_5_sonnet")
        assert result == []

    async def test_get_all_providers_summary_no_db(self) -> None:
        from saha.observability.metrics import MetricsAggregator
        agg    = MetricsAggregator()
        result = await agg.get_all_providers_summary()
        assert result == []

    async def test_get_cross_provider_report_no_db(self) -> None:
        from saha.observability.metrics import MetricsAggregator
        agg    = MetricsAggregator()
        result = await agg.get_cross_provider_report("SCENARIO_PY_FIX")
        assert result["scenario_id"] == "SCENARIO_PY_FIX"
        assert result["providers"]   == {}

    async def test_compute_all_no_db(self) -> None:
        from saha.observability.metrics import MetricsAggregator
        agg    = MetricsAggregator()
        result = await agg.compute_all()
        assert result == []


class TestAnomalyDetectorWithoutDB:
    """AnomalyDetector must degrade gracefully when no DB is available."""

    async def test_get_recent_anomalies_no_db(self) -> None:
        from saha.observability.anomaly_detector import AnomalyDetector
        from unittest.mock import AsyncMock, MagicMock
        mock_bus = MagicMock()
        mock_bus.publish = AsyncMock()
        detector = AnomalyDetector(bus=mock_bus)
        result   = await detector.get_recent_anomalies()
        assert result == []

    async def test_run_all_checks_no_db(self) -> None:
        from saha.observability.anomaly_detector import AnomalyDetector
        from unittest.mock import AsyncMock, MagicMock
        mock_bus = MagicMock()
        mock_bus.publish = AsyncMock()
        detector  = AnomalyDetector(bus=mock_bus)
        anomalies = await detector.run_all_checks()
        assert anomalies == []  # graceful, no raise
