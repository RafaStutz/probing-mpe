from __future__ import annotations

import json
import tempfile
import unittest
from enum import Enum
from pathlib import Path

from probing_mpe.plotting import (
    FINAL_DIAGNOSTIC_PLOT_NAME,
    PNG_SIGNATURE,
    REQUIRED_PLOT_NAMES,
    SUMMARY_FILE_NAME,
    SummaryKey,
    build_analysis_outputs,
)


class TestValue(float, Enum):
    return_mean_base = 10.0
    return_increment = 1.0
    diagnostic_base = 0.1
    diagnostic_null_offset = 0.05
    behavioral_base = 0.2


class MakePlotsTest(unittest.TestCase):
    def test_build_analysis_outputs_writes_summary_and_required_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_dir = root / "runs"
            plots_dir = root / "plots"
            _write_complete_synthetic_matrix(runs_dir)

            summary = build_analysis_outputs(runs_dir=runs_dir, plots_dir=plots_dir)

            self.assertEqual(summary[SummaryKey.run_count.value], 24)
            self.assertEqual(len(summary[SummaryKey.per_run_metrics.value]), 24)
            self.assertEqual(
                summary[SummaryKey.final_diagnostics_above_null.value][
                    "simple_spread_v3/ippo_ff"
                ]["HARnorm"],
                "3/3",
            )
            self.assertIn(
                "simple_speaker_listener_v4/MAPPO",
                summary[SummaryKey.memory_gap.value],
            )

            summary_path = plots_dir / SUMMARY_FILE_NAME
            self.assertTrue(summary_path.exists())
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded_summary[SummaryKey.run_count.value], 24)

            for plot_name in REQUIRED_PLOT_NAMES:
                plot_path = plots_dir / plot_name
                self.assertTrue(plot_path.exists(), plot_name)
                self.assertEqual(
                    plot_path.read_bytes()[: len(PNG_SIGNATURE)],
                    PNG_SIGNATURE,
                )

    def test_build_analysis_outputs_handles_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_dir = root / "runs"
            plots_dir = root / "plots"
            run_dir = runs_dir / "simple_spread_v3" / "ippo_ff" / "seed_0"
            run_dir.mkdir(parents=True)
            _write_final_artifacts(run_dir, "simple_spread_v3", "ippo_ff", 0)

            summary = build_analysis_outputs(runs_dir=runs_dir, plots_dir=plots_dir)

            self.assertEqual(summary[SummaryKey.run_count.value], 1)
            self.assertEqual(len(summary[SummaryKey.missing_artifacts.value]), 0)
            self.assertTrue((plots_dir / FINAL_DIAGNOSTIC_PLOT_NAME).exists())


def _write_complete_synthetic_matrix(runs_dir: Path) -> None:
    env_names = ("simple_spread_v3", "simple_speaker_listener_v4")
    config_ids = ("ippo_ff", "ippo_rnn", "mappo_ff", "mappo_rnn")
    seeds = (0, 1, 2)
    for env_name in env_names:
        for config_id in config_ids:
            for seed in seeds:
                run_dir = runs_dir / env_name / config_id / f"seed_{seed}"
                run_dir.mkdir(parents=True)
                _write_final_artifacts(run_dir, env_name, config_id, seed)
                if config_id == "mappo_rnn":
                    _write_checkpointed_diagnostics(run_dir, env_name, config_id, seed)


def _write_final_artifacts(
    run_dir: Path,
    env_name: str,
    config_id: str,
    seed: int,
) -> None:
    value = _base_value(config_id, seed)
    diagnostics = {
        "metadata": {"env_name": env_name, "config_id": config_id, "seed": seed},
        "diagnostics": {
            "OAR": {"normalized": value},
            "HAR": {"normalized": value + TestValue.diagnostic_base.value},
            "PIF": {"normalized": value + TestValue.diagnostic_base.value},
            "AA": {"normalized": value + TestValue.diagnostic_base.value},
            "DAI": {"normalized": value + TestValue.diagnostic_base.value},
        },
    }
    null_diagnostics = {
        "metadata": {"env_name": env_name, "config_id": config_id, "seed": seed},
        "null_diagnostics": {
            "OAR": {"normalized_mean": value - TestValue.diagnostic_null_offset.value},
            "HAR": {"normalized_mean": value - TestValue.diagnostic_null_offset.value},
            "PIF": {"normalized_mean": value - TestValue.diagnostic_null_offset.value},
            "AA": {"normalized_mean": value - TestValue.diagnostic_null_offset.value},
            "DAI": {"normalized_mean": value - TestValue.diagnostic_null_offset.value},
        },
    }
    behavioral_metrics = {
        "metadata": {"env_name": env_name, "config_id": config_id, "seed": seed},
        "behavioral_metrics": {
            "eval/return_mean": TestValue.return_mean_base.value + value,
            "eval/coverage_success_rate": TestValue.behavioral_base.value + value,
            "eval/final_landmark_distance_mean": value,
            "eval/collision_rate": value,
            "eval/target_success_rate": TestValue.behavioral_base.value + value,
            "eval/final_target_distance_mean": value,
            "eval/wrong_landmark_rate": value,
        },
    }
    (run_dir / "diagnostics_final.json").write_text(
        json.dumps(diagnostics),
        encoding="utf-8",
    )
    (run_dir / "diagnostics_null_final.json").write_text(
        json.dumps(null_diagnostics),
        encoding="utf-8",
    )
    (run_dir / "behavioral_metrics_final.json").write_text(
        json.dumps(behavioral_metrics),
        encoding="utf-8",
    )


def _write_checkpointed_diagnostics(
    run_dir: Path,
    env_name: str,
    config_id: str,
    seed: int,
) -> None:
    diagnostics_dir = run_dir / "diagnostics_by_progress"
    diagnostics_dir.mkdir()
    for progress in (25, 50, 75, 100):
        value = _base_value(config_id, seed) + progress / 1000.0
        diagnostics = {
            "metadata": {
                "env_name": env_name,
                "config_id": config_id,
                "seed": seed,
                "training_progress_percent": progress,
            },
            "diagnostics": {
                "OAR": {"normalized": value},
                "HAR": {"normalized": value},
                "PIF": {"normalized": value},
                "DAI": {"normalized": value},
            },
        }
        (diagnostics_dir / f"diagnostics_{progress}.json").write_text(
            json.dumps(diagnostics),
            encoding="utf-8",
        )


def _base_value(config_id: str, seed: int) -> float:
    config_offsets = {
        "ippo_ff": 0.0,
        "ippo_rnn": 1.0,
        "mappo_ff": 2.0,
        "mappo_rnn": 3.0,
    }
    return (
        TestValue.return_increment.value * float(seed)
        + config_offsets[config_id]
        + TestValue.diagnostic_base.value
    )


if __name__ == "__main__":
    unittest.main()
