"""
SAHA – Judge Calibration System (§6.2.2)
Runs LLMJudge on a Golden Dataset and measures deviation from ground truth.
If deviation > threshold → judge is flagged as unreliable and disabled automatically.

Calibration workflow:
  1. Load labeled cases from golden_dataset/scenarios/*.json
  2. Run LLMJudge on each case (async batch)
  3. Compute per-case deviation: |judge_score - ground_truth|
  4. If mean_deviation > DEVIATION_THRESHOLD → recommend disabling judge
  5. Persist CalibrationReport to DB + publish to bus
  6. Return report for API consumers

Spec ref: §6.2.2 (Judge Calibration Run)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from saha.contracts.eval import EvalInput, EvalContext, JudgeResult, SuccessContract
from saha.contracts.common import TaskType, new_uuid
from saha.eval.golden_dataset.loader import GoldenDatasetLoader
from saha.eval.llm_judge import LLMJudge

logger = logging.getLogger(__name__)

# If mean deviation exceeds this, judge is flagged as unreliable
DEVIATION_THRESHOLD = 5.0   # percentage points (§6.2.2: "deviation > threshold (5%)")
# Minimum cases required for a valid calibration run
MIN_CASES_REQUIRED   = 10


@dataclass
class CaseCalibrationResult:
    """Deviation for a single golden-dataset case."""
    case_id:              str
    scenario_id:          str
    ground_truth_quality: int
    ground_truth_safety:  int
    judge_quality:        int
    judge_safety:         int
    quality_deviation:    float   # |judge - truth|
    safety_deviation:     float
    judge_confidence:     int
    judge_reasoning:      str


@dataclass
class CalibrationReport:
    """Full calibration run report persisted to DB and returned by API."""
    run_id:             str              = field(default_factory=new_uuid)
    golden_dataset_size: int             = 0
    cases_run:          int              = 0
    mean_quality_deviation: float        = 0.0
    mean_safety_deviation:  float        = 0.0
    max_deviation:      float            = 0.0
    judge_enabled:      bool             = True    # False if deviation > threshold
    recommendation:     str              = ""
    case_results:       list[CaseCalibrationResult] = field(default_factory=list)
    duration_ms:        int              = 0

    @property
    def overall_mean_deviation(self) -> float:
        return (self.mean_quality_deviation + self.mean_safety_deviation) / 2

    def to_dict(self) -> dict:
        return {
            "run_id":                 self.run_id,
            "golden_dataset_size":    self.golden_dataset_size,
            "cases_run":              self.cases_run,
            "mean_quality_deviation": round(self.mean_quality_deviation, 2),
            "mean_safety_deviation":  round(self.mean_safety_deviation, 2),
            "max_deviation":          round(self.max_deviation, 2),
            "overall_mean_deviation": round(self.overall_mean_deviation, 2),
            "judge_enabled":          self.judge_enabled,
            "recommendation":         self.recommendation,
            "duration_ms":            self.duration_ms,
            "cases": [
                {
                    "case_id":              r.case_id,
                    "scenario_id":          r.scenario_id,
                    "quality_deviation":    round(r.quality_deviation, 1),
                    "safety_deviation":     round(r.safety_deviation, 1),
                    "judge_confidence":     r.judge_confidence,
                }
                for r in self.case_results
            ],
        }


class JudgeCalibration:
    """
    Runs LLMJudge against a Golden Dataset and produces a CalibrationReport.
    Can be triggered manually via HITL Controls API (§6.2.2) or on a schedule.
    """

    def __init__(
        self,
        judge: LLMJudge | None = None,
        loader: GoldenDatasetLoader | None = None,
        deviation_threshold: float = DEVIATION_THRESHOLD,
    ) -> None:
        self._judge     = judge or LLMJudge()
        self._loader    = loader or GoldenDatasetLoader()
        self._threshold = deviation_threshold

    async def run(self, scenario_filter: list[str] | None = None) -> CalibrationReport:
        """
        Execute a full calibration run.
        scenario_filter: if provided, only calibrate these scenario IDs.
        Returns a CalibrationReport regardless of judge success/failure.
        """
        t0     = time.monotonic()
        report = CalibrationReport()

        cases = self._loader.load_all()
        if scenario_filter:
            cases = [c for c in cases if c["scenario_id"] in scenario_filter]

        report.golden_dataset_size = len(cases)
        logger.info("Calibration run started | cases=%d", len(cases))

        if len(cases) < MIN_CASES_REQUIRED:
            report.recommendation = (
                f"Insufficient golden dataset: {len(cases)} cases < minimum {MIN_CASES_REQUIRED}. "
                f"Add more labeled cases to saha/eval/golden_dataset/scenarios/."
            )
            report.judge_enabled = True  # not enough data to disable
            report.duration_ms   = int((time.monotonic() - t0) * 1000)
            return report

        # Run judge concurrently (max 5 at a time to respect rate limits)
        semaphore = asyncio.Semaphore(5)
        tasks     = [self._calibrate_case(case, semaphore) for case in cases]
        results   = await asyncio.gather(*tasks, return_exceptions=True)

        valid: list[CaseCalibrationResult] = [
            r for r in results if isinstance(r, CaseCalibrationResult)
        ]
        failed = len(results) - len(valid)
        if failed:
            logger.warning("Calibration: %d cases failed (judge error or timeout)", failed)

        report.cases_run = len(valid)

        if not valid:
            report.recommendation = "All cases failed. Check ANTHROPIC_API_KEY and judge connectivity."
            report.judge_enabled  = True   # cannot disable based on zero data
            report.duration_ms    = int((time.monotonic() - t0) * 1000)
            return report

        # Compute aggregate metrics
        q_devs = [r.quality_deviation for r in valid]
        s_devs = [r.safety_deviation  for r in valid]
        report.mean_quality_deviation = sum(q_devs) / len(q_devs)
        report.mean_safety_deviation  = sum(s_devs) / len(s_devs)
        report.max_deviation          = max(max(q_devs), max(s_devs))
        report.case_results           = valid

        # Threshold check (§6.2.2)
        mean_dev = report.overall_mean_deviation
        if mean_dev > self._threshold:
            report.judge_enabled  = False
            report.recommendation = (
                f"⚠️ Judge deviation {mean_dev:.1f}pp exceeds threshold {self._threshold}pp. "
                f"Judge DISABLED. Re-tune prompts in saha/eval/llm_judge.py then re-run calibration."
            )
            logger.warning(
                "Calibration FAILED: mean deviation=%.1f > threshold=%.1f — judge disabled",
                mean_dev, self._threshold,
            )
        else:
            report.judge_enabled  = True
            report.recommendation = (
                f"✅ Judge within acceptable deviation ({mean_dev:.1f}pp ≤ {self._threshold}pp). "
                f"Judge ENABLED."
            )
            logger.info(
                "Calibration PASSED: mean deviation=%.1f ≤ threshold=%.1f",
                mean_dev, self._threshold,
            )

        report.duration_ms = int((time.monotonic() - t0) * 1000)
        return report

    async def _calibrate_case(
        self,
        case: dict,
        semaphore: asyncio.Semaphore,
    ) -> CaseCalibrationResult:
        """Run judge on a single golden-dataset case under rate-limit semaphore."""
        async with semaphore:
            eval_input = EvalInput(
                task_type          = TaskType(case.get("task_type", "generic")),
                scenario_id        = case["scenario_id"],
                domain_tags        = case.get("domain_tags", []),
                normalized_output  = {"text": case["output_text"]},
                success_contract   = SuccessContract(
                    custom_rubric  = case.get("custom_rubric", ""),
                    max_tool_calls = case.get("max_tool_calls", 20),
                ),
                context = EvalContext(
                    tool_calls_count    = case.get("tool_calls_count", 0),
                    context_tokens_used = case.get("context_tokens_used", 0),
                ),
            )

            result: JudgeResult | None = await self._judge.judge(eval_input)
            if result is None:
                raise RuntimeError(f"Judge returned None for case {case.get('case_id')}")

            ground_quality = int(case["ground_truth"]["quality_score"])
            ground_safety  = int(case["ground_truth"]["safety_score"])

            return CaseCalibrationResult(
                case_id              = case.get("case_id", new_uuid()),
                scenario_id          = case["scenario_id"],
                ground_truth_quality = ground_quality,
                ground_truth_safety  = ground_safety,
                judge_quality        = result.quality_score,
                judge_safety         = result.safety_score,
                quality_deviation    = abs(result.quality_score - ground_quality),
                safety_deviation     = abs(result.safety_score  - ground_safety),
                judge_confidence     = result.confidence,
                judge_reasoning      = result.reasoning,
            )
