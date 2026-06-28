from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

REPOSITORY_ROOT_PARENT_INDEX = 1
sys.path.insert(0, str(Path(__file__).resolve().parents[REPOSITORY_ROOT_PARENT_INDEX]))

from probing_mpe.evaluation import (
    DEFAULT_CMI_K,
    DEFAULT_HISTORY_K,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_EFFECT,
    DEFAULT_NULL_REPS,
    DEFAULT_PARALLEL_WORKERS,
    DEFAULT_POSTERIOR_ALPHA,
    DiagnosticName,
    compute_diagnostics_for_trajectory,
    write_diagnostic_outputs,
)
from probing_mpe.experiments.artifacts import progress_checkpoint
from probing_mpe.trajectories import (
    TARGET_DIAGNOSTIC_TRANSITIONS,
    export_trajectory_from_checkpoint,
)


DEFAULT_CONFIG_ID = "mappo_rnn"
CHECKPOINT_FILE_GLOB = "*.pt"
DEFAULT_OUTPUT_EPISODES: int | None = None


class ProgressPercent(int, Enum):
    twenty_five = 25
    fifty = 50
    seventy_five = 75
    one_hundred = 100


class DirectoryName(str, Enum):
    checkpoints = "checkpoints"
    trajectories_by_progress = "trajectories_by_progress"
    diagnostics_by_progress = "diagnostics_by_progress"


class FileNameTemplate(str, Enum):
    checkpoint_file = "checkpoint_{progress}.pt"
    checkpoint_directory = "checkpoint_{progress}"
    generic_checkpoint = "checkpoint.pt"
    trajectory = "trajectory_eval_{progress}.pkl"
    diagnostics = "diagnostics_{progress}.json"
    null_diagnostics = "diagnostics_null_{progress}.json"


class BackendImportName(str, Enum):
    dec_pomdp_diagnostics = "dec_pomdp_diagnostics"


class ErrorMessage(str, Enum):
    missing_checkpoint = "No checkpoint found for progress"
    ambiguous_checkpoint = "Multiple checkpoint files found for progress"


@dataclass(frozen=True)
class CheckpointedDiagnosticPaths:
    progress: ProgressPercent
    checkpoint_path: Path
    trajectory_path: Path
    diagnostics_path: Path
    null_diagnostics_path: Path


ExportFunction = Callable[
    [Path, Path, int, int | None, int, str | None, str | None], dict[str, object]
]
ComputeFunction = Callable[
    [
        dict[str, object],
        object,
        int,
        int,
        int,
        int | None,
        float,
        tuple[str, ...],
        float,
        int,
        bool | None,
    ],
    tuple[dict[str, object], dict[str, object]],
]


def build_progress_artifact_paths(
    run_dir: Path, progress: ProgressPercent, checkpoint_path: Path | None = None
) -> CheckpointedDiagnosticPaths:
    default_checkpoint_path = (
        run_dir
        / DirectoryName.checkpoints.value
        / _format_file_name(FileNameTemplate.checkpoint_file, progress)
    )
    return CheckpointedDiagnosticPaths(
        progress=progress,
        checkpoint_path=checkpoint_path or default_checkpoint_path,
        trajectory_path=run_dir
        / DirectoryName.trajectories_by_progress.value
        / _format_file_name(FileNameTemplate.trajectory, progress),
        diagnostics_path=run_dir
        / DirectoryName.diagnostics_by_progress.value
        / _format_file_name(FileNameTemplate.diagnostics, progress),
        null_diagnostics_path=run_dir
        / DirectoryName.diagnostics_by_progress.value
        / _format_file_name(FileNameTemplate.null_diagnostics, progress),
    )


def discover_progress_checkpoint(run_dir: Path, progress: ProgressPercent) -> Path:
    try:
        return progress_checkpoint(run_dir, progress.value).path
    except FileNotFoundError:
        pass

    checkpoints_dir = run_dir / DirectoryName.checkpoints.value
    direct_checkpoint = checkpoints_dir / _format_file_name(
        FileNameTemplate.checkpoint_file, progress
    )
    if direct_checkpoint.exists():
        return direct_checkpoint

    progress_dir = checkpoints_dir / _format_file_name(
        FileNameTemplate.checkpoint_directory, progress
    )
    generic_checkpoint = progress_dir / FileNameTemplate.generic_checkpoint.value
    if generic_checkpoint.exists():
        return generic_checkpoint

    progress_named_checkpoint = progress_dir / _format_file_name(
        FileNameTemplate.checkpoint_file, progress
    )
    if progress_named_checkpoint.exists():
        return progress_named_checkpoint

    matches = sorted(progress_dir.glob(CHECKPOINT_FILE_GLOB)) if progress_dir.exists() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"{ErrorMessage.ambiguous_checkpoint.value} {progress.value}: {matches}")
    raise FileNotFoundError(f"{ErrorMessage.missing_checkpoint.value} {progress.value}")


def run_checkpointed_diagnostics(
    run_dir: Path,
    checkpoint_paths: Mapping[ProgressPercent, Path],
    diagnostics_module: object,
    env_name: str | None,
    config_id: str,
    episodes: int | None,
    target_transitions: int,
    history_k: int,
    cmi_k: int,
    null_reps: int,
    max_samples: int | None,
    posterior_alpha: float,
    metrics: tuple[str, ...],
    min_effect: float,
    parallel_workers: int,
    force_continuous_actions: bool | None,
    export_function: ExportFunction = export_trajectory_from_checkpoint,
    compute_function: ComputeFunction = compute_diagnostics_for_trajectory,
) -> list[CheckpointedDiagnosticPaths]:
    outputs: list[CheckpointedDiagnosticPaths] = []
    for progress in ProgressPercent:
        checkpoint_path = checkpoint_paths.get(progress) or discover_progress_checkpoint(
            run_dir, progress
        )
        paths = build_progress_artifact_paths(
            run_dir=run_dir,
            progress=progress,
            checkpoint_path=checkpoint_path,
        )
        trajectory = export_function(
            paths.checkpoint_path,
            paths.trajectory_path,
            progress.value,
            episodes,
            target_transitions,
            env_name,
            config_id,
        )
        diagnostics, null_diagnostics = compute_function(
            trajectory,
            diagnostics_module,
            history_k,
            cmi_k,
            null_reps,
            max_samples,
            posterior_alpha,
            metrics,
            min_effect,
            parallel_workers,
            force_continuous_actions,
        )
        write_diagnostic_outputs(
            diagnostics=diagnostics,
            null_diagnostics=null_diagnostics,
            diagnostics_path=paths.diagnostics_path,
            null_diagnostics_path=paths.null_diagnostics_path,
        )
        outputs.append(paths)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MAPPO-RNN checkpoint trajectories and diagnostics at 25/50/75/100%."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    parser.add_argument("--checkpoint-25", type=Path, default=None)
    parser.add_argument("--checkpoint-50", type=Path, default=None)
    parser.add_argument("--checkpoint-75", type=Path, default=None)
    parser.add_argument("--checkpoint-100", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=DEFAULT_OUTPUT_EPISODES)
    parser.add_argument(
        "--target-transitions", type=int, default=TARGET_DIAGNOSTIC_TRANSITIONS
    )
    parser.add_argument("--history-k", type=int, default=DEFAULT_HISTORY_K)
    parser.add_argument("--cmi-k", type=int, default=DEFAULT_CMI_K)
    parser.add_argument("--null-reps", type=int, default=DEFAULT_NULL_REPS)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--posterior-alpha", type=float, default=DEFAULT_POSTERIOR_ALPHA)
    parser.add_argument("--min-effect", type=float, default=DEFAULT_MIN_EFFECT)
    parser.add_argument("--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=[metric.value for metric in DiagnosticName],
        default=[
            DiagnosticName.oar.value,
            DiagnosticName.har.value,
            DiagnosticName.pif.value,
            DiagnosticName.dai.value,
        ],
    )
    parser.add_argument("--force-continuous-actions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics_module = importlib.import_module(
        BackendImportName.dec_pomdp_diagnostics.value
    )
    checkpoint_paths = _checkpoint_paths_from_args(args)
    outputs = run_checkpointed_diagnostics(
        run_dir=args.run_dir,
        checkpoint_paths=checkpoint_paths,
        diagnostics_module=diagnostics_module,
        env_name=args.env_name,
        config_id=args.config_id,
        episodes=args.episodes,
        target_transitions=args.target_transitions,
        history_k=args.history_k,
        cmi_k=args.cmi_k,
        null_reps=args.null_reps,
        max_samples=args.max_samples,
        posterior_alpha=args.posterior_alpha,
        metrics=tuple(args.metrics),
        min_effect=args.min_effect,
        parallel_workers=args.parallel_workers,
        force_continuous_actions=True if args.force_continuous_actions else None,
    )
    print(f"Saved checkpointed diagnostics for {len(outputs)} progress points")
    return 0


def _checkpoint_paths_from_args(args: argparse.Namespace) -> dict[ProgressPercent, Path]:
    raw_paths = {
        ProgressPercent.twenty_five: args.checkpoint_25,
        ProgressPercent.fifty: args.checkpoint_50,
        ProgressPercent.seventy_five: args.checkpoint_75,
        ProgressPercent.one_hundred: args.checkpoint_100,
    }
    return {
        progress: checkpoint_path
        for progress, checkpoint_path in raw_paths.items()
        if checkpoint_path is not None
    }


def _format_file_name(template: FileNameTemplate, progress: ProgressPercent) -> str:
    return template.value.format(progress=progress.value)


if __name__ == "__main__":
    raise SystemExit(main())
