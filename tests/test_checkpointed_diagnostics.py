from __future__ import annotations

import tempfile
import unittest
from enum import Enum
from pathlib import Path

from probing_mpe.experiments.run_checkpointed_diagnostics import (
    CheckpointedDiagnosticPaths,
    ProgressPercent,
    build_progress_artifact_paths,
    discover_progress_checkpoint,
    run_checkpointed_diagnostics,
)


CHECKPOINT_CONTENT = "checkpoint"
SINGLE_EPISODE = 1
SMALL_TARGET_TRANSITIONS = 10
SMALL_HISTORY_K = 3
SMALL_CMI_K = 5
SINGLE_NULL_REP = 1
SMALL_MAX_SAMPLES = 50
TEST_POSTERIOR_ALPHA = 0.01
TEST_MIN_EFFECT = 0.01
SINGLE_PARALLEL_WORKER = 1


class FakePayloadKey(str, Enum):
    progress = "progress"
    kind = "kind"


class FakeDiagnosticKind(str, Enum):
    diagnostics = "diagnostics"
    null = "null"


class FakeDiagnosticsModule:
    pass


class CheckpointedDiagnosticsTest(unittest.TestCase):
    def test_build_progress_artifact_paths_matches_plan_layout(self) -> None:
        run_dir = Path("/runs/simple_spread_v3/mappo_rnn/seed_0")

        paths = build_progress_artifact_paths(run_dir, ProgressPercent.twenty_five)

        self.assertEqual(
            paths.trajectory_path,
            run_dir / "trajectories_by_progress" / "trajectory_eval_25.pkl",
        )
        self.assertEqual(
            paths.diagnostics_path,
            run_dir / "diagnostics_by_progress" / "diagnostics_25.json",
        )
        self.assertEqual(
            paths.null_diagnostics_path,
            run_dir / "diagnostics_by_progress" / "diagnostics_null_25.json",
        )

    def test_discover_progress_checkpoint_accepts_plan_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            checkpoint_dir = run_dir / "checkpoints" / "checkpoint_50"
            checkpoint_dir.mkdir(parents=True)
            checkpoint_path = checkpoint_dir / "checkpoint_50.pt"
            checkpoint_path.write_text(CHECKPOINT_CONTENT, encoding="utf-8")

            discovered = discover_progress_checkpoint(run_dir, ProgressPercent.fifty)

            self.assertEqual(discovered, checkpoint_path)

    def test_run_checkpointed_diagnostics_exports_and_computes_all_progress_points(self) -> None:
        calls: list[tuple[str, int, Path]] = []

        def export_checkpoint(
            checkpoint_path: Path,
            output_path: Path,
            progress_percent: int,
            episodes: int | None,
            target_transitions: int,
            env_name_override: str | None,
            config_id_override: str | None,
        ) -> dict[str, object]:
            calls.append((checkpoint_path.name, progress_percent, output_path))
            return {FakePayloadKey.progress.value: progress_percent}

        def compute_diagnostics(
            trajectory: dict[str, object],
            diagnostics_module: object,
            history_k: int,
            cmi_k: int,
            null_reps: int,
            max_samples: int | None,
            posterior_alpha: float,
            metrics: tuple[str, ...],
            min_effect: float,
            parallel_workers: int,
            force_continuous_actions: bool | None,
        ) -> tuple[dict[str, object], dict[str, object]]:
            return (
                {
                    FakePayloadKey.progress.value: trajectory[FakePayloadKey.progress.value],
                    FakePayloadKey.kind.value: FakeDiagnosticKind.diagnostics.value,
                },
                {
                    FakePayloadKey.progress.value: trajectory[FakePayloadKey.progress.value],
                    FakePayloadKey.kind.value: FakeDiagnosticKind.null.value,
                },
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            checkpoint_paths: dict[ProgressPercent, Path] = {}
            for progress in ProgressPercent:
                checkpoint_path = run_dir / f"checkpoint_{progress.value}.pt"
                checkpoint_path.write_text(CHECKPOINT_CONTENT, encoding="utf-8")
                checkpoint_paths[progress] = checkpoint_path

            outputs = run_checkpointed_diagnostics(
                run_dir=run_dir,
                checkpoint_paths=checkpoint_paths,
                diagnostics_module=FakeDiagnosticsModule(),
                env_name="simple_spread_v3",
                config_id="mappo_rnn",
                episodes=SINGLE_EPISODE,
                target_transitions=SMALL_TARGET_TRANSITIONS,
                history_k=SMALL_HISTORY_K,
                cmi_k=SMALL_CMI_K,
                null_reps=SINGLE_NULL_REP,
                max_samples=SMALL_MAX_SAMPLES,
                posterior_alpha=TEST_POSTERIOR_ALPHA,
                metrics=("oar", "har"),
                min_effect=TEST_MIN_EFFECT,
                parallel_workers=SINGLE_PARALLEL_WORKER,
                force_continuous_actions=None,
                export_function=export_checkpoint,
                compute_function=compute_diagnostics,
            )

            self.assertEqual(
                [call[1] for call in calls],
                [25, 50, 75, 100],
            )
            self.assertEqual(len(outputs), 4)
            for paths in outputs:
                self.assertIsInstance(paths, CheckpointedDiagnosticPaths)
                self.assertTrue(paths.diagnostics_path.exists())
                self.assertTrue(paths.null_diagnostics_path.exists())


if __name__ == "__main__":
    unittest.main()
