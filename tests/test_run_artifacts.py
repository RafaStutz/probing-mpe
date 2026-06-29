from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from probing_mpe.experiments.artifacts import (
    ArtifactFileName,
    ArtifactKey,
    CheckpointProgress,
    DirectoryName,
    MetadataKey,
    MetadataStatus,
    ProgressCheckpointKey,
    RunArtifactPaths,
    final_artifacts_are_valid,
    normalize_final_checkpoint,
    progress_checkpoints,
    reloadable_checkpoint_path,
    run_is_complete,
    training_checkpoint_exists,
    write_run_metadata,
)
from probing_mpe.evaluation import DiagnosticJsonKey
from probing_mpe.metrics import BehavioralMetricKey
from probing_mpe.trajectories import GroupName, TrajectoryKey


CHECKPOINT_CONTENT_LOW = "low"
CHECKPOINT_CONTENT_HIGH = "high"
FINAL_FRAME = 10_000_000
LOW_FRAME = 2_498_560
MID_FRAME = 4_999_168
HIGH_FRAME = 7_499_776
SEED = 0


class RunArtifactsTest(unittest.TestCase):
    def test_normalize_final_checkpoint_copies_highest_numbered_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            checkpoints_dir = run_dir / DirectoryName.checkpoints.value
            checkpoints_dir.mkdir()
            (checkpoints_dir / "checkpoint_2048.pt").write_text(
                CHECKPOINT_CONTENT_LOW,
                encoding="utf-8",
            )
            source_checkpoint = checkpoints_dir / f"checkpoint_{FINAL_FRAME}.pt"
            source_checkpoint.write_text(CHECKPOINT_CONTENT_HIGH, encoding="utf-8")

            normalized = normalize_final_checkpoint(run_dir)

            self.assertEqual(normalized.source_path, source_checkpoint)
            self.assertEqual(normalized.frame, FINAL_FRAME)
            self.assertEqual(
                normalized.normalized_path,
                run_dir
                / DirectoryName.checkpoint_final.value
                / ArtifactFileName.checkpoint.value,
            )
            self.assertEqual(
                normalized.normalized_path.read_text(encoding="utf-8"),
                CHECKPOINT_CONTENT_HIGH,
            )

    def test_progress_checkpoints_use_nearest_lower_frame_and_final_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            checkpoints_dir = run_dir / DirectoryName.checkpoints.value
            checkpoints_dir.mkdir()
            for frame in (LOW_FRAME, MID_FRAME, HIGH_FRAME, FINAL_FRAME):
                (checkpoints_dir / f"checkpoint_{frame}.pt").write_text(
                    str(frame),
                    encoding="utf-8",
                )

            selected = progress_checkpoints(run_dir)

            self.assertEqual(
                [checkpoint.progress_percent for checkpoint in selected],
                [25, 50, 75, 100],
            )
            self.assertEqual(
                [checkpoint.selected_frame for checkpoint in selected],
                [LOW_FRAME, MID_FRAME, HIGH_FRAME, FINAL_FRAME],
            )
            self.assertEqual(
                selected[-1].path,
                run_dir
                / DirectoryName.checkpoint_final.value
                / ArtifactFileName.checkpoint.value,
            )

    def test_reloadable_checkpoint_path_uses_original_benchmarl_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            experiment_dir = run_dir / "benchmarl_experiment"
            checkpoints_dir = experiment_dir / DirectoryName.checkpoints.value
            checkpoints_dir.mkdir(parents=True)
            source_checkpoint = checkpoints_dir / f"checkpoint_{FINAL_FRAME}.pt"
            source_checkpoint.write_text(CHECKPOINT_CONTENT_HIGH, encoding="utf-8")
            normalized = normalize_final_checkpoint(run_dir)

            reloadable = reloadable_checkpoint_path(
                run_dir,
                normalized.normalized_path,
            )

            self.assertEqual(reloadable, source_checkpoint)
            self.assertTrue(training_checkpoint_exists(run_dir))

    def test_final_artifacts_are_valid_and_run_metadata_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _artifact_paths(Path(temp_dir))
            paths.checkpoint_path.parent.mkdir(parents=True)
            paths.checkpoint_path.write_text(CHECKPOINT_CONTENT_HIGH, encoding="utf-8")
            _write_valid_final_artifacts(paths)

            self.assertTrue(final_artifacts_are_valid(paths))
            self.assertTrue(run_is_complete(paths, checkpointed_required=False))

            write_run_metadata(
                paths.metadata_path,
                {
                    MetadataKey.schema_version.value: 1,
                    MetadataKey.status.value: MetadataStatus.complete.value,
                    MetadataKey.env_name.value: paths.env_name,
                    MetadataKey.config_id.value: paths.config_id,
                    MetadataKey.seed.value: paths.seed,
                    MetadataKey.artifacts.value: {
                        ArtifactKey.trajectory_final.value: str(paths.trajectory_path)
                    },
                    MetadataKey.progress_checkpoints.value: [
                        CheckpointProgress(
                            progress_percent=25,
                            target_frame=2_500_000,
                            selected_frame=LOW_FRAME,
                            actual_progress_percent=24.9856,
                            path=Path("/checkpoint.pt"),
                        ).to_json()
                    ],
                },
            )

            loaded = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(
                loaded[MetadataKey.status.value],
                MetadataStatus.complete.value,
            )
            self.assertEqual(
                loaded[MetadataKey.progress_checkpoints.value][0][
                    ProgressCheckpointKey.selected_frame.value
                ],
                LOW_FRAME,
            )

    def test_run_is_complete_requires_checkpointed_artifacts_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _artifact_paths(Path(temp_dir))
            paths.checkpoint_path.parent.mkdir(parents=True)
            paths.checkpoint_path.write_text(CHECKPOINT_CONTENT_HIGH, encoding="utf-8")
            _write_valid_final_artifacts(paths)

            self.assertFalse(run_is_complete(paths, checkpointed_required=True))

            for progress in (25, 50, 75, 100):
                trajectory_path = (
                    paths.run_dir
                    / DirectoryName.trajectories_by_progress.value
                    / f"trajectory_eval_{progress}.pkl"
                )
                trajectory_path.parent.mkdir(parents=True, exist_ok=True)
                with trajectory_path.open("wb") as trajectory_file:
                    pickle.dump(_valid_trajectory(), trajectory_file)
                diagnostics_dir = (
                    paths.run_dir / DirectoryName.diagnostics_by_progress.value
                )
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                (diagnostics_dir / f"diagnostics_{progress}.json").write_text(
                    json.dumps(
                        {
                            DiagnosticJsonKey.diagnostics.value: {},
                        }
                    ),
                    encoding="utf-8",
                )
                (diagnostics_dir / f"diagnostics_null_{progress}.json").write_text(
                    json.dumps(
                        {
                            DiagnosticJsonKey.null_diagnostics.value: {},
                        }
                    ),
                    encoding="utf-8",
                )

            self.assertTrue(run_is_complete(paths, checkpointed_required=True))


def _artifact_paths(run_dir: Path) -> RunArtifactPaths:
    return RunArtifactPaths(
        env_name="simple_spread_v3",
        config_id="ippo_ff",
        seed=SEED,
        run_dir=run_dir,
        checkpoint_path=run_dir
        / DirectoryName.checkpoint_final.value
        / ArtifactFileName.checkpoint.value,
        trajectory_path=run_dir / ArtifactFileName.trajectory_final.value,
        diagnostics_path=run_dir / ArtifactFileName.diagnostics_final.value,
        null_diagnostics_path=run_dir / ArtifactFileName.diagnostics_null_final.value,
        behavioral_metrics_path=run_dir
        / ArtifactFileName.behavioral_metrics_final.value,
        metadata_path=run_dir / ArtifactFileName.run_metadata.value,
    )


def _write_valid_final_artifacts(paths: RunArtifactPaths) -> None:
    with paths.trajectory_path.open("wb") as trajectory_file:
        pickle.dump(_valid_trajectory(), trajectory_file)
    paths.diagnostics_path.write_text(
        json.dumps({DiagnosticJsonKey.diagnostics.value: {}}),
        encoding="utf-8",
    )
    paths.null_diagnostics_path.write_text(
        json.dumps({DiagnosticJsonKey.null_diagnostics.value: {}}),
        encoding="utf-8",
    )
    paths.behavioral_metrics_path.write_text(
        json.dumps({BehavioralMetricKey.behavioral_metrics.value: {}}),
        encoding="utf-8",
    )


def _valid_trajectory() -> dict[str, object]:
    agent_values = {
        "agent_0": np.array([0, 1], dtype=np.int64),
        "agent_1": np.array([1, 0], dtype=np.int64),
    }
    return {
        TrajectoryKey.env_name.value: "simple_spread_v3",
        TrajectoryKey.algorithm.value: "ippo",
        TrajectoryKey.policy_architecture.value: "ff",
        TrajectoryKey.config_id.value: "ippo_ff",
        TrajectoryKey.seed.value: SEED,
        TrajectoryKey.parameter_sharing.value: False,
        TrajectoryKey.training_progress_percent.value: 100,
        TrajectoryKey.observations.value: {
            "agent_0": np.zeros((2, 3), dtype=np.float32),
            "agent_1": np.zeros((2, 3), dtype=np.float32),
        },
        TrajectoryKey.actions_raw.value: agent_values,
        TrajectoryKey.actions_diagnostic.value: agent_values,
        TrajectoryKey.rewards.value: {
            "agent_0": np.zeros(2, dtype=np.float32),
            "agent_1": np.zeros(2, dtype=np.float32),
        },
        TrajectoryKey.timesteps.value: {
            "agent_0": np.array([0, 1], dtype=np.int64),
            "agent_1": np.array([0, 1], dtype=np.int64),
        },
        TrajectoryKey.episode_ids.value: {
            "agent_0": np.array([0, 0], dtype=np.int64),
            "agent_1": np.array([0, 0], dtype=np.int64),
        },
        TrajectoryKey.dones.value: {
            "agent_0": np.array([False, True]),
            "agent_1": np.array([False, True]),
        },
        TrajectoryKey.infos.value: {"agent_0": [{}, {}], "agent_1": [{}, {}]},
        TrajectoryKey.global_state.value: np.zeros((2, 4), dtype=np.float32),
        TrajectoryKey.hidden_states.value: None,
        TrajectoryKey.agent_role_map.value: {
            "agent_0": GroupName.agent.value,
            "agent_1": GroupName.agent.value,
        },
        TrajectoryKey.action_space_description.value: {
            "agent_0": {"type": "categorical_discrete"},
            "agent_1": {"type": "categorical_discrete"},
        },
    }


if __name__ == "__main__":
    unittest.main()
