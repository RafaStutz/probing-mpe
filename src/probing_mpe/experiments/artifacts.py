from __future__ import annotations

import json
import math
import pickle
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from probing_mpe.evaluation import DiagnosticJsonKey
from probing_mpe.metrics import BehavioralMetricKey
from probing_mpe.trajectories import validate_trajectory_schema


TOTAL_TRAINING_FRAMES = 10_000_000
CHECKPOINT_NAME_PREFIX = "checkpoint_"
CHECKPOINT_NAME_SUFFIX = ".pt"
JSON_INDENT = 2
PROGRESS_PERCENTS = (25, 50, 75, 100)
PERCENT_DENOMINATOR = 100


class DirectoryName(str, Enum):
    checkpoint_final = "checkpoint_final"
    checkpoints = "checkpoints"
    diagnostics_by_progress = "diagnostics_by_progress"
    trajectories_by_progress = "trajectories_by_progress"


class ArtifactFileName(str, Enum):
    checkpoint = "checkpoint.pt"
    checkpoint_glob = "*.pt"
    trajectory_final = "trajectory_eval_final.pkl"
    diagnostics_final = "diagnostics_final.json"
    diagnostics_null_final = "diagnostics_null_final.json"
    behavioral_metrics_final = "behavioral_metrics_final.json"
    run_metadata = "run_metadata.json"


class ArtifactKey(str, Enum):
    trajectory_final = "trajectory_final"
    diagnostics_final = "diagnostics_final"
    diagnostics_null_final = "diagnostics_null_final"
    behavioral_metrics_final = "behavioral_metrics_final"


class MetadataKey(str, Enum):
    schema_version = "schema_version"
    status = "status"
    env_name = "env_name"
    config_id = "config_id"
    seed = "seed"
    run_dir = "run_dir"
    benchmarl_root = "benchmarl_root"
    python_executable = "python_executable"
    wandb_enabled = "wandb_enabled"
    commands = "commands"
    final_checkpoint = "final_checkpoint"
    progress_checkpoints = "progress_checkpoints"
    artifacts = "artifacts"


class MetadataStatus(str, Enum):
    started = "started"
    training_complete = "training_complete"
    postprocessing_complete = "postprocessing_complete"
    complete = "complete"
    failed = "failed"
    skipped = "skipped"


class CheckpointMetadataKey(str, Enum):
    source_path = "source_path"
    normalized_path = "normalized_path"
    frame = "frame"


class ProgressCheckpointKey(str, Enum):
    progress_percent = "progress_percent"
    target_frame = "target_frame"
    selected_frame = "selected_frame"
    actual_progress_percent = "actual_progress_percent"
    path = "path"


class ErrorMessage(str, Enum):
    missing_checkpoint = "No checkpoint found"
    missing_progress_checkpoint = "No checkpoint found for progress"
    invalid_checkpoint_name = "Invalid checkpoint name"


@dataclass(frozen=True)
class CheckpointCandidate:
    frame: int
    path: Path


@dataclass(frozen=True)
class NormalizedCheckpoint:
    source_path: Path
    normalized_path: Path
    frame: int | None

    def to_json(self) -> dict[str, object]:
        return {
            CheckpointMetadataKey.source_path.value: str(self.source_path),
            CheckpointMetadataKey.normalized_path.value: str(self.normalized_path),
            CheckpointMetadataKey.frame.value: self.frame,
        }


@dataclass(frozen=True)
class CheckpointProgress:
    progress_percent: int
    target_frame: int
    selected_frame: int
    actual_progress_percent: float
    path: Path

    def to_json(self) -> dict[str, object]:
        return {
            ProgressCheckpointKey.progress_percent.value: self.progress_percent,
            ProgressCheckpointKey.target_frame.value: self.target_frame,
            ProgressCheckpointKey.selected_frame.value: self.selected_frame,
            ProgressCheckpointKey.actual_progress_percent.value: (
                self.actual_progress_percent
            ),
            ProgressCheckpointKey.path.value: str(self.path),
        }


@dataclass(frozen=True)
class RunArtifactPaths:
    env_name: str
    config_id: str
    seed: int
    run_dir: Path
    checkpoint_path: Path
    trajectory_path: Path
    diagnostics_path: Path
    null_diagnostics_path: Path
    behavioral_metrics_path: Path
    metadata_path: Path

    def artifact_json(self) -> dict[str, str]:
        return {
            ArtifactKey.trajectory_final.value: str(self.trajectory_path),
            ArtifactKey.diagnostics_final.value: str(self.diagnostics_path),
            ArtifactKey.diagnostics_null_final.value: str(
                self.null_diagnostics_path
            ),
            ArtifactKey.behavioral_metrics_final.value: str(
                self.behavioral_metrics_path
            ),
        }


def run_artifact_paths(
    env_name: str,
    config_id: str,
    seed: int,
    run_dir: Path,
) -> RunArtifactPaths:
    return RunArtifactPaths(
        env_name=env_name,
        config_id=config_id,
        seed=seed,
        run_dir=run_dir,
        checkpoint_path=default_final_checkpoint(run_dir),
        trajectory_path=run_dir / ArtifactFileName.trajectory_final.value,
        diagnostics_path=run_dir / ArtifactFileName.diagnostics_final.value,
        null_diagnostics_path=run_dir / ArtifactFileName.diagnostics_null_final.value,
        behavioral_metrics_path=run_dir
        / ArtifactFileName.behavioral_metrics_final.value,
        metadata_path=run_dir / ArtifactFileName.run_metadata.value,
    )


def default_final_checkpoint(run_dir: Path) -> Path:
    return (
        run_dir
        / DirectoryName.checkpoint_final.value
        / ArtifactFileName.checkpoint.value
    )


def checkpoint_frame(checkpoint_path: Path) -> int | None:
    name = checkpoint_path.name
    if not (
        name.startswith(CHECKPOINT_NAME_PREFIX)
        and name.endswith(CHECKPOINT_NAME_SUFFIX)
    ):
        return None
    frame_text = name[
        len(CHECKPOINT_NAME_PREFIX) : -len(CHECKPOINT_NAME_SUFFIX)
    ]
    if not frame_text.isdigit():
        return None
    return int(frame_text)


def numbered_checkpoints(run_dir: Path) -> list[CheckpointCandidate]:
    checkpoint_root = run_dir / DirectoryName.checkpoints.value
    if not checkpoint_root.exists():
        return []
    candidates: list[CheckpointCandidate] = []
    for checkpoint_path in sorted(
        checkpoint_root.rglob(ArtifactFileName.checkpoint_glob.value)
    ):
        frame = checkpoint_frame(checkpoint_path)
        if frame is not None:
            candidates.append(CheckpointCandidate(frame=frame, path=checkpoint_path))
    return sorted(candidates, key=lambda candidate: (candidate.frame, str(candidate.path)))


def normalize_final_checkpoint(run_dir: Path) -> NormalizedCheckpoint:
    normalized_path = default_final_checkpoint(run_dir)
    if normalized_path.exists():
        return NormalizedCheckpoint(
            source_path=normalized_path,
            normalized_path=normalized_path,
            frame=_largest_numbered_frame(run_dir),
        )

    candidates = numbered_checkpoints(run_dir)
    if not candidates:
        raise FileNotFoundError(f"{ErrorMessage.missing_checkpoint.value}: {run_dir}")
    selected = candidates[-1]
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected.path, normalized_path)
    return NormalizedCheckpoint(
        source_path=selected.path,
        normalized_path=normalized_path,
        frame=selected.frame,
    )


def progress_checkpoints(run_dir: Path) -> list[CheckpointProgress]:
    output: list[CheckpointProgress] = []
    for progress_percent in PROGRESS_PERCENTS:
        output.append(progress_checkpoint(run_dir, progress_percent))
    return output


def progress_checkpoint(run_dir: Path, progress_percent: int) -> CheckpointProgress:
    target_frame = _target_frame(progress_percent)
    if progress_percent == PERCENT_DENOMINATOR:
        normalized = normalize_final_checkpoint(run_dir)
        selected_frame = normalized.frame or target_frame
        return CheckpointProgress(
            progress_percent=progress_percent,
            target_frame=target_frame,
            selected_frame=selected_frame,
            actual_progress_percent=_actual_progress_percent(selected_frame),
            path=normalized.normalized_path,
        )

    candidates = [
        candidate
        for candidate in numbered_checkpoints(run_dir)
        if candidate.frame <= target_frame
    ]
    if not candidates:
        raise FileNotFoundError(
            f"{ErrorMessage.missing_progress_checkpoint.value} {progress_percent}: {run_dir}"
        )
    selected = candidates[-1]
    return CheckpointProgress(
        progress_percent=progress_percent,
        target_frame=target_frame,
        selected_frame=selected.frame,
        actual_progress_percent=_actual_progress_percent(selected.frame),
        path=selected.path,
    )


def final_artifacts_are_valid(paths: RunArtifactPaths) -> bool:
    return (
        paths.checkpoint_path.exists()
        and trajectory_is_valid(paths.trajectory_path)
        and diagnostics_are_valid(paths.diagnostics_path)
        and null_diagnostics_are_valid(paths.null_diagnostics_path)
        and behavioral_metrics_are_valid(paths.behavioral_metrics_path)
    )


def run_is_complete(paths: RunArtifactPaths, checkpointed_required: bool) -> bool:
    if not final_artifacts_are_valid(paths):
        return False
    if checkpointed_required:
        return checkpointed_artifacts_are_valid(paths.run_dir)
    return True


def checkpointed_artifacts_are_valid(run_dir: Path) -> bool:
    for progress_percent in PROGRESS_PERCENTS:
        if not trajectory_is_valid(
            run_dir
            / DirectoryName.trajectories_by_progress.value
            / f"trajectory_eval_{progress_percent}.pkl"
        ):
            return False
        diagnostics_dir = run_dir / DirectoryName.diagnostics_by_progress.value
        if not diagnostics_are_valid(
            diagnostics_dir / f"diagnostics_{progress_percent}.json"
        ):
            return False
        if not null_diagnostics_are_valid(
            diagnostics_dir / f"diagnostics_null_{progress_percent}.json"
        ):
            return False
    return True


def trajectory_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("rb") as trajectory_file:
            trajectory = pickle.load(trajectory_file)
        if not isinstance(trajectory, Mapping):
            return False
        validate_trajectory_schema(trajectory)
    except (OSError, pickle.PickleError, ValueError, TypeError, EOFError):
        return False
    return True


def diagnostics_are_valid(path: Path) -> bool:
    return _json_has_key(path, DiagnosticJsonKey.diagnostics.value)


def null_diagnostics_are_valid(path: Path) -> bool:
    return _json_has_key(path, DiagnosticJsonKey.null_diagnostics.value)


def behavioral_metrics_are_valid(path: Path) -> bool:
    return _json_has_key(path, BehavioralMetricKey.behavioral_metrics.value)


def write_run_metadata(metadata_path: Path, metadata: Mapping[str, object]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(_to_jsonable(metadata), indent=JSON_INDENT, allow_nan=True),
        encoding="utf-8",
    )


def base_run_metadata(
    paths: RunArtifactPaths,
    status: MetadataStatus,
    benchmarl_root: Path,
    python_executable: str,
    wandb_enabled: bool,
    commands: Mapping[str, Sequence[str]],
    final_checkpoint: NormalizedCheckpoint | None,
    progress_records: Sequence[CheckpointProgress],
) -> dict[str, object]:
    return {
        MetadataKey.schema_version.value: 1,
        MetadataKey.status.value: status.value,
        MetadataKey.env_name.value: paths.env_name,
        MetadataKey.config_id.value: paths.config_id,
        MetadataKey.seed.value: paths.seed,
        MetadataKey.run_dir.value: str(paths.run_dir),
        MetadataKey.benchmarl_root.value: str(benchmarl_root),
        MetadataKey.python_executable.value: python_executable,
        MetadataKey.wandb_enabled.value: wandb_enabled,
        MetadataKey.commands.value: {
            command_name: list(command)
            for command_name, command in commands.items()
        },
        MetadataKey.final_checkpoint.value: (
            final_checkpoint.to_json() if final_checkpoint is not None else None
        ),
        MetadataKey.progress_checkpoints.value: [
            progress_record.to_json() for progress_record in progress_records
        ],
        MetadataKey.artifacts.value: paths.artifact_json(),
    }


def _json_has_key(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(loaded, Mapping) and key in loaded


def _target_frame(progress_percent: int) -> int:
    return int(TOTAL_TRAINING_FRAMES * progress_percent / PERCENT_DENOMINATOR)


def _actual_progress_percent(selected_frame: int) -> float:
    return selected_frame / TOTAL_TRAINING_FRAMES * PERCENT_DENOMINATOR


def _largest_numbered_frame(run_dir: Path) -> int | None:
    checkpoints = numbered_checkpoints(run_dir)
    if not checkpoints:
        return None
    return checkpoints[-1].frame


def _to_jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return value
    return value
