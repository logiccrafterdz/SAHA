"""
SAHA – Golden Dataset Loader for Judge Calibration (§6.2.2)
Loads labeled evaluation cases from JSON scenario files.
Each case must specify:
  - scenario_id, task_type, output_text (agent output to judge)
  - ground_truth: {quality_score, safety_score}  (human-labeled)
  - Optional: custom_rubric, domain_tags, max_tool_calls

File format: one JSON file per scenario in:
  saha/eval/golden_dataset/scenarios/<SCENARIO_ID>.json
Each file is a list of case objects.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


class GoldenDatasetLoader:
    """Loads and validates all golden-dataset case files."""

    def __init__(self, scenarios_dir: Path | None = None) -> None:
        self._dir = scenarios_dir or SCENARIOS_DIR

    def load_all(self) -> list[dict]:
        """Load all *.json files in the scenarios directory."""
        cases: list[dict] = []
        if not self._dir.exists():
            logger.warning("Golden dataset directory not found: %s", self._dir)
            return cases

        for path in sorted(self._dir.glob("*.json")):
            file_cases = self._load_file(path)
            cases.extend(file_cases)
            logger.debug("Loaded %d cases from %s", len(file_cases), path.name)

        logger.info("Golden dataset loaded: %d total cases from %s", len(cases), self._dir)
        return cases

    def load_scenario(self, scenario_id: str) -> list[dict]:
        """Load cases for a specific scenario ID."""
        path = self._dir / f"{scenario_id}.json"
        if not path.exists():
            logger.warning("No golden dataset for scenario: %s", scenario_id)
            return []
        return self._load_file(path)

    def _load_file(self, path: Path) -> list[dict]:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning("%s: expected list, got %s", path.name, type(data).__name__)
                return []
            # Validate required fields
            valid = []
            for case in data:
                if self._validate(case, path.stem):
                    valid.append(case)
            return valid
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load golden dataset %s: %s", path.name, exc)
            return []

    @staticmethod
    def _validate(case: dict, scenario_id: str) -> bool:
        required = ("output_text", "ground_truth")
        for field in required:
            if field not in case:
                logger.warning("Case missing '%s' field in %s — skipping", field, scenario_id)
                return False
        gt = case.get("ground_truth", {})
        if "quality_score" not in gt or "safety_score" not in gt:
            logger.warning("Case missing ground_truth scores in %s — skipping", scenario_id)
            return False
        # Default fields
        case.setdefault("scenario_id",  scenario_id)
        case.setdefault("task_type",    "generic")
        case.setdefault("domain_tags",  [])
        case.setdefault("custom_rubric", "")
        case.setdefault("max_tool_calls", 20)
        case.setdefault("tool_calls_count", 0)
        case.setdefault("context_tokens_used", 0)
        return True
