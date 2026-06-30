from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataevol.benchmarks import FrozenBenchmarkBuilder, FrozenBenchmarkError, build_frozen_benchmark
from dataevol.evolve import generate_idea_prd, validate_idea_prd
from dataevol.experiments import run_router_policy_experiment
from dataevol.promotion import PromotionGate, PromotionRejected


class EvolutionLoopTests(unittest.TestCase):
    def test_idea_prd_validation_accepts_complete_prd_and_reports_missing_sections(self) -> None:
        opportunity = {
            "id": "router_policy_cost",
            "observation": "Low-risk accepted tasks used expensive models.",
            "hypothesis": "Cost-aware routing can preserve quality at lower cost.",
            "proposed_change": "Prefer cheap verified workers for low-risk tasks.",
            "expected_metric": "cost_per_verified_task",
        }
        prd = generate_idea_prd(opportunity)

        valid, missing = validate_idea_prd(prd)
        self.assertTrue(valid)
        self.assertEqual(missing, [])

        invalid, missing_invalid = validate_idea_prd("# Idea PRD: Incomplete\n\n## Observation\nOnly one section.\n")
        self.assertFalse(invalid)
        self.assertIn("Hypothesis", missing_invalid)
        self.assertIn("Promotion Rule", missing_invalid)


    def test_frozen_benchmark_rejects_overwrite_and_detects_mutation(self) -> None:
        with _tmp_path() as tmp_path:
            result = build_frozen_benchmark([{"id": "case_001", "task": "route cheaply"}], tmp_path)
            self.assertEqual(result.item_count, 1)

            with self.assertRaises(FrozenBenchmarkError):
                build_frozen_benchmark([{"id": "case_002", "task": "mutate"}], tmp_path)

            result.benchmark_path.write_text('{"id":"case_001","task":"changed"}\n', encoding="utf-8")
            with self.assertRaises(FrozenBenchmarkError):
                FrozenBenchmarkBuilder().assert_immutable(result.manifest_path)


    def test_failed_experiment_cannot_promote(self) -> None:
        with _tmp_path() as tmp_path:
            rollback = tmp_path / "rollback.json"
            rollback.write_text('{"version":"control"}\n', encoding="utf-8")
            fixture_metrics = {
                "control": [
                    {"cost_per_verified_task": 1.0, "correctness": 0.92, "verification_pass_rate": 0.9, "safety_score": 1.0},
                    {"cost_per_verified_task": 1.0, "correctness": 0.92, "verification_pass_rate": 0.9, "safety_score": 1.0},
                ],
                "variant": [
                    {"cost_per_verified_task": 1.2, "correctness": 0.92, "verification_pass_rate": 0.9, "safety_score": 1.0},
                    {"cost_per_verified_task": 1.1, "correctness": 0.92, "verification_pass_rate": 0.9, "safety_score": 1.0},
                ],
            }

            report = run_router_policy_experiment(fixture_metrics, tmp_path, rollback_snapshot=str(rollback))
            self.assertFalse(report["primary_metric_improved"])

            with self.assertRaises(PromotionRejected):
                PromotionGate().promote(report, tmp_path / "promotions")


    def test_successful_promotion_path_writes_promotion_record(self) -> None:
        with _tmp_path() as tmp_path:
            rollback = tmp_path / "router_policy_v0.json"
            rollback.write_text('{"version":"control"}\n', encoding="utf-8")
            fixture_metrics = {
                "control": [
                    {"cost_per_verified_task": 1.0, "correctness": 0.92, "verification_pass_rate": 0.9, "safety_score": 1.0},
                    {"cost_per_verified_task": 1.1, "correctness": 0.91, "verification_pass_rate": 0.9, "safety_score": 1.0},
                ],
                "variant": [
                    {"cost_per_verified_task": 0.8, "correctness": 0.92, "verification_pass_rate": 0.9, "safety_score": 1.0},
                    {"cost_per_verified_task": 0.9, "correctness": 0.91, "verification_pass_rate": 0.9, "safety_score": 1.0},
                ],
            }

            report = run_router_policy_experiment(fixture_metrics, tmp_path, rollback_snapshot=str(rollback))
            decision = PromotionGate().promote(report, tmp_path / "promotions")

            self.assertTrue(decision.promoted)
            self.assertIsNotNone(decision.promotion_path)
            self.assertTrue(decision.promotion_path.exists())


class _tmp_path:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
