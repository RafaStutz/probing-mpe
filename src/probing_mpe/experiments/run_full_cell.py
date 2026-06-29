from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

from probing_mpe.experiments.artifacts import (
    ArtifactFileName,
    CheckpointProgress,
    MetadataStatus,
    NormalizedCheckpoint,
    RunArtifactPaths,
    base_run_metadata,
    behavioral_metrics_are_valid,
    checkpoint_frame,
    default_final_checkpoint,
    diagnostics_are_valid,
    normalize_final_checkpoint,
    null_diagnostics_are_valid,
    progress_checkpoints,
    reloadable_checkpoint_path,
    run_artifact_paths,
    run_is_complete,
    trajectory_is_valid,
    training_checkpoint_exists,
    write_run_metadata,
)


FULL_CELL_ENV_NAME = "simple_spread_v3"
FULL_CELL_CONFIG_ID = "ippo_rnn"
FINAL_PROGRESS_PERCENT = 100
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_CONFIG_PATH = Path("configs/reduced_mpe/simple_spread_v3/ippo_rnn.yaml")
DEFAULT_BENCHMARL_ROOT = Path("/tmp/BenchMARL")
DEFAULT_OUTPUT_DIR = Path("runs")
SUCCESS_RETURN_CODE = 0
DEFAULT_DEVICE = "cuda:0"


class ConfigKey(str, Enum):
    env_name = "env_name"
    config_id = "config_id"
    benchmarl = "benchmarl"
    overrides = "overrides"


class BenchmarlKey(str, Enum):
    algorithm = "algorithm"
    task = "task"
    model = "model"
    critic_model = "critic_model"


class HydraGroupKey(str, Enum):
    critic_model = "model@critic_model"


class OverrideKey(str, Enum):
    buffer_device = "experiment.buffer_device"
    checkpoint_at_end = "experiment.checkpoint_at_end"
    create_json = "experiment.create_json"
    loggers = "experiment.loggers"
    prefer_continuous_actions = "experiment.prefer_continuous_actions"
    sampling_device = "experiment.sampling_device"
    save_folder = "experiment.save_folder"
    share_param_critic = "algorithm.share_param_critic"
    share_policy_params = "experiment.share_policy_params"
    train_device = "experiment.train_device"


class FullCellDirectoryName(str, Enum):
    checkpoint_final = "checkpoint_final"
    seed = "seed"


class FullCellArtifactName(str, Enum):
    generic_checkpoint = "checkpoint.pt"
    checkpoint_glob = "*.pt"
    trajectory_final = "trajectory_eval_final.pkl"
    diagnostics_final = "diagnostics_final.json"
    diagnostics_null_final = "diagnostics_null_final.json"
    behavioral_metrics_final = "behavioral_metrics_final.json"


class ScriptPath(str, Enum):
    export_trajectory = "scripts/export_benchmarl_trajectory.py"
    compute_diagnostics = "scripts/compute_diagnostics_from_trajectory.py"
    compute_behavioral_metrics = "scripts/compute_behavioral_metrics.py"


class CliFlag(str, Enum):
    checkpoint = "--checkpoint"
    config_path = "--config-path"
    output = "--output"
    output_dir = "--output-dir"
    progress = "--progress"
    trajectory = "--trajectory"
    diagnostics_output = "--diagnostics-output"
    null_output = "--null-output"
    env_name = "--env-name"
    config_id = "--config-id"
    benchmarl_root = "--benchmarl-root"
    device = "--device"
    seeds = "--seeds"
    python_executable = "--python-executable"
    wandb_mode = "--wandb-mode"
    dry_run = "--dry-run"
    force = "--force"


class CommandName(str, Enum):
    training = "training"
    export = "export"
    diagnostics = "diagnostics"
    behavioral = "behavioral"


class WandbMode(str, Enum):
    online = "online"
    offline = "offline"
    disabled = "disabled"


class ErrorMessage(str, Enum):
    missing_mapping = "Missing mapping"
    wrong_cell = "Config does not match the Step 8 full cell"
    required_false = "Required override must be false"
    missing_benchmarl = "Missing BenchMARL value"
    missing_checkpoint = "No final checkpoint found"
    ambiguous_checkpoint = "Multiple final checkpoints found"
    command_failed = "Command failed"


@dataclass(frozen=True)
class FullCellConfig:
    env_name: str
    config_id: str
    benchmarl: Mapping[str, object]
    overrides: Mapping[str, object]


@dataclass(frozen=True)
class FullCellOutput:
    env_name: str
    config_id: str
    seed: int
    progress_percent: int
    run_dir: Path
    checkpoint_path: Path
    trajectory_path: Path
    diagnostics_path: Path
    null_diagnostics_path: Path
    behavioral_metrics_path: Path
    metadata_path: Path


CommandRunner = Callable[[list[str], Path | None], int]
CheckpointResolver = Callable[[Path], Path]


def load_full_cell_config(config_path: Path) -> FullCellConfig:
    loaded_object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_object, dict):
        raise ValueError(f"Config file is not a mapping: {config_path}")
    loaded = {str(key): item for key, item in loaded_object.items()}

    env_name = _required_string(loaded, ConfigKey.env_name)
    config_id = _required_string(loaded, ConfigKey.config_id)
    if env_name != FULL_CELL_ENV_NAME or config_id != FULL_CELL_CONFIG_ID:
        raise ValueError(f"{ErrorMessage.wrong_cell.value}: {env_name}/{config_id}")

    return FullCellConfig(
        env_name=env_name,
        config_id=config_id,
        benchmarl=_required_mapping(loaded, ConfigKey.benchmarl),
        overrides=_required_mapping(loaded, ConfigKey.overrides),
    )


def build_training_command(
    full_cell_config: FullCellConfig,
    benchmarl_root: Path,
    run_dir: Path,
    seed: int,
    python_executable: str,
    wandb_enabled: bool,
    device: str = DEFAULT_DEVICE,
) -> list[str]:
    validate_required_overrides(full_cell_config)
    benchmarl_run = benchmarl_root / "benchmarl" / "run.py"
    command = [
        python_executable,
        str(benchmarl_run),
        f"{BenchmarlKey.algorithm.value}={_required_benchmarl_value(full_cell_config, BenchmarlKey.algorithm)}",
        f"{BenchmarlKey.task.value}={_required_benchmarl_value(full_cell_config, BenchmarlKey.task)}",
        f"{BenchmarlKey.model.value}={_required_benchmarl_value(full_cell_config, BenchmarlKey.model)}",
        f"{HydraGroupKey.critic_model.value}={_required_benchmarl_value(full_cell_config, BenchmarlKey.critic_model)}",
        f"seed={seed}",
    ]

    overrides = dict(full_cell_config.overrides)
    _apply_device_overrides(overrides, device)
    overrides[OverrideKey.save_folder.value] = str(run_dir)
    overrides[OverrideKey.checkpoint_at_end.value] = True
    overrides[OverrideKey.create_json.value] = True
    if not wandb_enabled:
        overrides[OverrideKey.loggers.value] = ["csv"]

    return command + [
        f"{key}={_format_hydra_value(value)}"
        for key, value in sorted(overrides.items())
    ]


def validate_required_overrides(full_cell_config: FullCellConfig) -> None:
    for override_key in (
        OverrideKey.share_policy_params,
        OverrideKey.share_param_critic,
        OverrideKey.prefer_continuous_actions,
    ):
        actual_value = full_cell_config.overrides.get(override_key.value)
        if actual_value is not False:
            raise ValueError(f"{ErrorMessage.required_false.value}: {override_key.value}")


def run_full_cell(
    output_dir: Path,
    config_path: Path,
    benchmarl_root: Path,
    seeds: Sequence[int],
    python_executable: str,
    wandb_enabled: bool,
    dry_run: bool,
    command_runner: CommandRunner | None = None,
    checkpoint_resolver: CheckpointResolver | None = None,
    force: bool = False,
    device: str = DEFAULT_DEVICE,
) -> list[FullCellOutput]:
    runner = command_runner or run_command
    resolver_is_injected = checkpoint_resolver is not None
    resolver = checkpoint_resolver or discover_final_checkpoint
    full_cell_config = load_full_cell_config(config_path)
    outputs: list[FullCellOutput] = []
    for seed in seeds:
        run_dir = _run_dir(output_dir, full_cell_config.env_name, full_cell_config.config_id, seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths = run_artifact_paths(
            env_name=full_cell_config.env_name,
            config_id=full_cell_config.config_id,
            seed=seed,
            run_dir=run_dir,
        )
        training_command = build_training_command(
            full_cell_config=full_cell_config,
            benchmarl_root=benchmarl_root,
            run_dir=run_dir,
            seed=seed,
            python_executable=python_executable,
            wandb_enabled=wandb_enabled,
            device=device,
        )
        output = _full_cell_output(
            full_cell_config,
            seed,
            run_dir,
            artifact_paths.checkpoint_path,
        )
        commands = _commands_for_output(output, training_command, python_executable)
        if not force and not dry_run and run_is_complete(artifact_paths, False):
            _write_metadata_for_output(
                artifact_paths=artifact_paths,
                status=MetadataStatus.complete,
                benchmarl_root=benchmarl_root,
                python_executable=python_executable,
                wandb_enabled=wandb_enabled,
                commands=commands,
                final_checkpoint=normalize_final_checkpoint(run_dir),
                include_progress=False,
            )
            outputs.append(output)
            continue

        _write_metadata_for_output(
            artifact_paths=artifact_paths,
            status=MetadataStatus.started,
            benchmarl_root=benchmarl_root,
            python_executable=python_executable,
            wandb_enabled=wandb_enabled,
            commands=commands,
            final_checkpoint=None,
            include_progress=False,
        )

        if force or dry_run or not training_checkpoint_exists(run_dir):
            _run_or_print(training_command, benchmarl_root, dry_run, runner)

        checkpoint_path = (
            artifact_paths.checkpoint_path if dry_run else resolver(run_dir)
        )
        export_checkpoint_path = (
            checkpoint_path
            if dry_run
            else reloadable_checkpoint_path(run_dir, checkpoint_path)
        )
        final_checkpoint = (
            NormalizedCheckpoint(
                source_path=checkpoint_path,
                normalized_path=checkpoint_path,
                frame=None,
            )
            if dry_run
            else _resolved_final_checkpoint(
                run_dir,
                checkpoint_path,
                allow_missing=resolver_is_injected,
            )
        )
        output = _full_cell_output(full_cell_config, seed, run_dir, export_checkpoint_path)
        commands = _commands_for_output(output, training_command, python_executable)
        _write_metadata_for_output(
            artifact_paths=artifact_paths,
            status=MetadataStatus.training_complete,
            benchmarl_root=benchmarl_root,
            python_executable=python_executable,
            wandb_enabled=wandb_enabled,
            commands=commands,
            final_checkpoint=final_checkpoint,
            include_progress=False,
        )

        if force or dry_run or not trajectory_is_valid(output.trajectory_path):
            _run_or_print(_export_command(output, python_executable), None, dry_run, runner)
        if (
            force
            or dry_run
            or not diagnostics_are_valid(output.diagnostics_path)
            or not null_diagnostics_are_valid(output.null_diagnostics_path)
        ):
            _run_or_print(_diagnostics_command(output, python_executable), None, dry_run, runner)
        if force or dry_run or not behavioral_metrics_are_valid(output.behavioral_metrics_path):
            _run_or_print(_behavioral_command(output, python_executable), None, dry_run, runner)
        _write_metadata_for_output(
            artifact_paths=artifact_paths,
            status=MetadataStatus.complete,
            benchmarl_root=benchmarl_root,
            python_executable=python_executable,
            wandb_enabled=wandb_enabled,
            commands=commands,
            final_checkpoint=final_checkpoint,
            include_progress=False,
        )
        outputs.append(output)
    return outputs


def run_command(command: list[str], cwd: Path | None) -> int:
    completed = subprocess.run(command, cwd=cwd, check=False)
    return completed.returncode


def discover_final_checkpoint(run_dir: Path) -> Path:
    return normalize_final_checkpoint(run_dir).normalized_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Step 8 full cell: simple_spread_v3/ippo_rnn seeds 0,1,2."
    )
    parser.add_argument(CliFlag.output_dir.value, type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(CliFlag.config_path.value, type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        CliFlag.benchmarl_root.value, type=Path, default=DEFAULT_BENCHMARL_ROOT
    )
    parser.add_argument(CliFlag.device.value, default=DEFAULT_DEVICE)
    parser.add_argument(CliFlag.seeds.value, nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(CliFlag.python_executable.value, default=sys.executable)
    parser.add_argument(
        CliFlag.wandb_mode.value,
        choices=[mode.value for mode in WandbMode],
        default=WandbMode.online.value,
    )
    parser.add_argument(CliFlag.dry_run.value, action="store_true")
    parser.add_argument(CliFlag.force.value, action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_full_cell(
        output_dir=args.output_dir,
        config_path=args.config_path,
        benchmarl_root=args.benchmarl_root,
        seeds=tuple(args.seeds),
        python_executable=args.python_executable,
        wandb_enabled=args.wandb_mode != WandbMode.disabled.value,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        device=args.device,
    )
    print(f"Prepared full-cell artifacts for {len(outputs)} seeds")
    return SUCCESS_RETURN_CODE


def _full_cell_output(
    full_cell_config: FullCellConfig,
    seed: int,
    run_dir: Path,
    checkpoint_path: Path,
) -> FullCellOutput:
    return FullCellOutput(
        env_name=full_cell_config.env_name,
        config_id=full_cell_config.config_id,
        seed=seed,
        progress_percent=FINAL_PROGRESS_PERCENT,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        trajectory_path=run_dir / FullCellArtifactName.trajectory_final.value,
        diagnostics_path=run_dir / FullCellArtifactName.diagnostics_final.value,
        null_diagnostics_path=run_dir / FullCellArtifactName.diagnostics_null_final.value,
        behavioral_metrics_path=run_dir / FullCellArtifactName.behavioral_metrics_final.value,
        metadata_path=run_dir / ArtifactFileName.run_metadata.value,
    )


def _export_command(output: FullCellOutput, python_executable: str) -> list[str]:
    return [
        python_executable,
        ScriptPath.export_trajectory.value,
        CliFlag.checkpoint.value,
        str(output.checkpoint_path),
        CliFlag.output.value,
        str(output.trajectory_path),
        CliFlag.progress.value,
        str(output.progress_percent),
        CliFlag.env_name.value,
        output.env_name,
        CliFlag.config_id.value,
        output.config_id,
    ]


def _diagnostics_command(output: FullCellOutput, python_executable: str) -> list[str]:
    return [
        python_executable,
        ScriptPath.compute_diagnostics.value,
        CliFlag.trajectory.value,
        str(output.trajectory_path),
        CliFlag.diagnostics_output.value,
        str(output.diagnostics_path),
        CliFlag.null_output.value,
        str(output.null_diagnostics_path),
    ]


def _behavioral_command(output: FullCellOutput, python_executable: str) -> list[str]:
    return [
        python_executable,
        ScriptPath.compute_behavioral_metrics.value,
        CliFlag.trajectory.value,
        str(output.trajectory_path),
        CliFlag.output.value,
        str(output.behavioral_metrics_path),
    ]


def _commands_for_output(
    output: FullCellOutput,
    training_command: list[str],
    python_executable: str,
) -> dict[str, list[str]]:
    return {
        CommandName.training.value: training_command,
        CommandName.export.value: _export_command(output, python_executable),
        CommandName.diagnostics.value: _diagnostics_command(output, python_executable),
        CommandName.behavioral.value: _behavioral_command(output, python_executable),
    }


def _write_metadata_for_output(
    artifact_paths: RunArtifactPaths,
    status: MetadataStatus,
    benchmarl_root: Path,
    python_executable: str,
    wandb_enabled: bool,
    commands: Mapping[str, Sequence[str]],
    final_checkpoint: NormalizedCheckpoint | None,
    include_progress: bool,
) -> None:
    progress_records: list[CheckpointProgress] = []
    if include_progress:
        try:
            progress_records = progress_checkpoints(artifact_paths.run_dir)
        except FileNotFoundError:
            progress_records = []
    write_run_metadata(
        artifact_paths.metadata_path,
        base_run_metadata(
            paths=artifact_paths,
            status=status,
            benchmarl_root=benchmarl_root,
            python_executable=python_executable,
            wandb_enabled=wandb_enabled,
            commands=commands,
            final_checkpoint=final_checkpoint,
            progress_records=progress_records,
        ),
    )


def _resolved_final_checkpoint(
    run_dir: Path,
    checkpoint_path: Path,
    allow_missing: bool,
) -> NormalizedCheckpoint:
    normalized_path = default_final_checkpoint(run_dir)
    if checkpoint_path.exists() and checkpoint_path != normalized_path:
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, normalized_path)
        return NormalizedCheckpoint(
            source_path=checkpoint_path,
            normalized_path=normalized_path,
            frame=checkpoint_frame(checkpoint_path),
        )
    if checkpoint_path.exists() and checkpoint_path == normalized_path:
        return normalize_final_checkpoint(run_dir)
    if checkpoint_path.exists():
        return NormalizedCheckpoint(
            source_path=checkpoint_path,
            normalized_path=checkpoint_path,
            frame=checkpoint_frame(checkpoint_path),
        )
    if allow_missing:
        return NormalizedCheckpoint(
            source_path=checkpoint_path,
            normalized_path=checkpoint_path,
            frame=None,
        )
    return normalize_final_checkpoint(run_dir)


def _apply_device_overrides(overrides: dict[str, object], device: str) -> None:
    overrides[OverrideKey.sampling_device.value] = device
    overrides[OverrideKey.train_device.value] = device
    overrides[OverrideKey.buffer_device.value] = device


def _run_or_print(
    command: list[str],
    cwd: Path | None,
    dry_run: bool,
    runner: CommandRunner,
) -> None:
    if dry_run:
        print(" ".join(command), flush=True)
        return
    return_code = runner(command, cwd)
    if return_code != SUCCESS_RETURN_CODE:
        raise RuntimeError(f"{ErrorMessage.command_failed.value}: {return_code}")


def _run_dir(output_dir: Path, env_name: str, config_id: str, seed: int) -> Path:
    return (
        output_dir
        / env_name
        / config_id
        / f"{FullCellDirectoryName.seed.value}_{seed}"
    )


def _default_final_checkpoint(run_dir: Path) -> Path:
    return default_final_checkpoint(run_dir)


def _required_string(source: Mapping[str, object], key: ConfigKey) -> str:
    value = source.get(key.value)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key.value} must be a non-empty string")
    return value


def _required_mapping(
    source: Mapping[str, object], key: ConfigKey
) -> Mapping[str, object]:
    value = source.get(key.value)
    if not isinstance(value, Mapping):
        raise ValueError(f"{ErrorMessage.missing_mapping.value}: {key.value}")
    return {str(item_key): item for item_key, item in value.items()}


def _required_benchmarl_value(
    full_cell_config: FullCellConfig, benchmarl_key: BenchmarlKey
) -> str:
    value = full_cell_config.benchmarl.get(benchmarl_key.value)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{ErrorMessage.missing_benchmarl.value}: {benchmarl_key.value}")
    return value


def _format_hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ",".join(_format_hydra_value(item) for item in value) + "]"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
