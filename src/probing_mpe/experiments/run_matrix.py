from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


DEFAULT_CONFIG_ROOT = Path("configs/reduced_mpe")
DEFAULT_BENCHMARL_ROOT = Path("/tmp/BenchMARL")
DEFAULT_OUTPUT_DIR = Path("runs")
DEFAULT_SEEDS = (0, 1, 2)
EXPECTED_MATRIX_RUNS = 24
CHECKPOINTED_CONFIG_ID = "mappo_rnn"
FINAL_PROGRESS_PERCENT = 100
SUCCESS_RETURN_CODE = 0
SINGLE_MATCH_COUNT = 1


class MatrixEnvName(str, Enum):
    simple_spread = "simple_spread_v3"
    simple_speaker_listener = "simple_speaker_listener_v4"


class MatrixConfigId(str, Enum):
    ippo_ff = "ippo_ff"
    ippo_rnn = "ippo_rnn"
    mappo_ff = "mappo_ff"
    mappo_rnn = "mappo_rnn"


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


class HydraScalarKey(str, Enum):
    seed = "seed"


class OverrideKey(str, Enum):
    checkpoint_at_end = "experiment.checkpoint_at_end"
    create_json = "experiment.create_json"
    loggers = "experiment.loggers"
    prefer_continuous_actions = "experiment.prefer_continuous_actions"
    save_folder = "experiment.save_folder"
    share_param_critic = "algorithm.share_param_critic"
    share_policy_params = "experiment.share_policy_params"


class MatrixDirectoryName(str, Enum):
    checkpoint_final = "checkpoint_final"
    seed = "seed"


class MatrixArtifactName(str, Enum):
    generic_checkpoint = "checkpoint.pt"
    checkpoint_glob = "*.pt"
    trajectory_final = "trajectory_eval_final.pkl"
    diagnostics_final = "diagnostics_final.json"
    diagnostics_null_final = "diagnostics_null_final.json"
    behavioral_metrics_final = "behavioral_metrics_final.json"


class MatrixScriptPath(str, Enum):
    export_trajectory = "scripts/export_benchmarl_trajectory.py"
    compute_diagnostics = "scripts/compute_diagnostics_from_trajectory.py"
    compute_behavioral_metrics = "scripts/compute_behavioral_metrics.py"
    run_checkpointed_diagnostics = "scripts/run_checkpointed_diagnostics.py"


class CliFlag(str, Enum):
    checkpoint = "--checkpoint"
    config_id = "--config-id"
    config_root = "--config-root"
    diagnostics_output = "--diagnostics-output"
    dry_run = "--dry-run"
    env_name = "--env-name"
    null_output = "--null-output"
    output = "--output"
    output_dir = "--output-dir"
    progress = "--progress"
    run_dir = "--run-dir"
    seeds = "--seeds"
    trajectory = "--trajectory"
    benchmarl_root = "--benchmarl-root"
    python_executable = "--python-executable"
    wandb_mode = "--wandb-mode"


class WandbMode(str, Enum):
    online = "online"
    offline = "offline"
    disabled = "disabled"


class ErrorMessage(str, Enum):
    ambiguous_checkpoint = "Multiple final checkpoints found"
    command_failed = "Command failed"
    missing_benchmarl = "Missing BenchMARL value"
    missing_checkpoint = "No final checkpoint found"
    missing_mapping = "Missing mapping"
    required_false = "Required override must be false"


@dataclass(frozen=True)
class MatrixConfig:
    env_name: str
    config_id: str
    benchmarl: Mapping[str, object]
    overrides: Mapping[str, object]


@dataclass(frozen=True)
class MatrixPlanEntry:
    env_name: MatrixEnvName
    config_id: MatrixConfigId
    seed: int
    config_path: Path


@dataclass(frozen=True)
class MatrixRunOutput:
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


CommandRunner = Callable[[list[str], Path | None], int]
CheckpointResolver = Callable[[Path], Path]


def build_matrix_plan(
    config_root: Path,
    seeds: Sequence[int],
) -> list[MatrixPlanEntry]:
    return [
        MatrixPlanEntry(
            env_name=env_name,
            config_id=config_id,
            seed=seed,
            config_path=config_root / env_name.value / f"{config_id.value}.yaml",
        )
        for env_name in MatrixEnvName
        for config_id in MatrixConfigId
        for seed in seeds
    ]


def load_matrix_config(config_path: Path) -> MatrixConfig:
    loaded_object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_object, dict):
        raise ValueError(f"Config file is not a mapping: {config_path}")
    loaded = {str(key): item for key, item in loaded_object.items()}

    return MatrixConfig(
        env_name=_required_string(loaded, ConfigKey.env_name),
        config_id=_required_string(loaded, ConfigKey.config_id),
        benchmarl=_required_mapping(loaded, ConfigKey.benchmarl),
        overrides=_required_mapping(loaded, ConfigKey.overrides),
    )


def build_training_command(
    matrix_config: MatrixConfig,
    benchmarl_root: Path,
    run_dir: Path,
    seed: int,
    python_executable: str,
    wandb_enabled: bool,
) -> list[str]:
    validate_required_overrides(matrix_config)
    benchmarl_run = benchmarl_root / "benchmarl" / "run.py"
    command = [
        python_executable,
        str(benchmarl_run),
        f"{BenchmarlKey.algorithm.value}={_required_benchmarl_value(matrix_config, BenchmarlKey.algorithm)}",
        f"{BenchmarlKey.task.value}={_required_benchmarl_value(matrix_config, BenchmarlKey.task)}",
        f"{BenchmarlKey.model.value}={_required_benchmarl_value(matrix_config, BenchmarlKey.model)}",
        f"{HydraGroupKey.critic_model.value}={_required_benchmarl_value(matrix_config, BenchmarlKey.critic_model)}",
        f"{HydraScalarKey.seed.value}={seed}",
    ]

    overrides = dict(matrix_config.overrides)
    overrides[OverrideKey.save_folder.value] = str(run_dir)
    overrides[OverrideKey.checkpoint_at_end.value] = True
    overrides[OverrideKey.create_json.value] = True
    if not wandb_enabled:
        overrides[OverrideKey.loggers.value] = ["csv"]

    return command + [
        f"{key}={_format_hydra_value(value)}"
        for key, value in sorted(overrides.items())
    ]


def validate_required_overrides(matrix_config: MatrixConfig) -> None:
    for override_key in (
        OverrideKey.share_policy_params,
        OverrideKey.share_param_critic,
        OverrideKey.prefer_continuous_actions,
    ):
        actual_value = matrix_config.overrides.get(override_key.value)
        if actual_value is not False:
            raise ValueError(f"{ErrorMessage.required_false.value}: {override_key.value}")


def run_matrix(
    output_dir: Path,
    config_root: Path,
    benchmarl_root: Path,
    seeds: Sequence[int],
    python_executable: str,
    wandb_enabled: bool,
    dry_run: bool,
    command_runner: CommandRunner | None = None,
    checkpoint_resolver: CheckpointResolver | None = None,
) -> list[MatrixRunOutput]:
    runner = command_runner or run_command
    resolver = checkpoint_resolver or discover_final_checkpoint
    outputs: list[MatrixRunOutput] = []

    for plan_entry in build_matrix_plan(config_root, seeds):
        matrix_config = load_matrix_config(plan_entry.config_path)
        run_dir = _run_dir(
            output_dir,
            matrix_config.env_name,
            matrix_config.config_id,
            plan_entry.seed,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        training_command = build_training_command(
            matrix_config=matrix_config,
            benchmarl_root=benchmarl_root,
            run_dir=run_dir,
            seed=plan_entry.seed,
            python_executable=python_executable,
            wandb_enabled=wandb_enabled,
        )
        _run_or_print(training_command, benchmarl_root, dry_run, runner)

        checkpoint_path = (
            _default_final_checkpoint(run_dir) if dry_run else resolver(run_dir)
        )
        output = _matrix_output(
            matrix_config=matrix_config,
            seed=plan_entry.seed,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
        )
        _run_or_print(_export_command(output, python_executable), None, dry_run, runner)
        _run_or_print(_diagnostics_command(output, python_executable), None, dry_run, runner)
        _run_or_print(_behavioral_command(output, python_executable), None, dry_run, runner)
        if matrix_config.config_id == CHECKPOINTED_CONFIG_ID:
            _run_or_print(
                _checkpointed_diagnostics_command(output, python_executable),
                None,
                dry_run,
                runner,
            )
        outputs.append(output)

    return outputs


def run_command(command: list[str], cwd: Path | None) -> int:
    completed = subprocess.run(command, cwd=cwd, check=False)
    return completed.returncode


def discover_final_checkpoint(run_dir: Path) -> Path:
    preferred = _default_final_checkpoint(run_dir)
    if preferred.exists():
        return preferred

    checkpoint_dir = run_dir / MatrixDirectoryName.checkpoint_final.value
    matches = (
        sorted(checkpoint_dir.glob(MatrixArtifactName.checkpoint_glob.value))
        if checkpoint_dir.exists()
        else []
    )
    if len(matches) == SINGLE_MATCH_COUNT:
        return matches[0]
    if len(matches) > SINGLE_MATCH_COUNT:
        raise ValueError(f"{ErrorMessage.ambiguous_checkpoint.value}: {matches}")

    recursive_matches = sorted(run_dir.rglob(MatrixArtifactName.checkpoint_glob.value))
    if len(recursive_matches) == SINGLE_MATCH_COUNT:
        return recursive_matches[0]
    if len(recursive_matches) > SINGLE_MATCH_COUNT:
        raise ValueError(f"{ErrorMessage.ambiguous_checkpoint.value}: {recursive_matches}")
    raise FileNotFoundError(f"{ErrorMessage.missing_checkpoint.value}: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full reduced-MPE BenchMARL matrix."
    )
    parser.add_argument(CliFlag.output_dir.value, type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(CliFlag.config_root.value, type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument(
        CliFlag.benchmarl_root.value, type=Path, default=DEFAULT_BENCHMARL_ROOT
    )
    parser.add_argument(CliFlag.seeds.value, nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(CliFlag.python_executable.value, default=sys.executable)
    parser.add_argument(
        CliFlag.wandb_mode.value,
        choices=[mode.value for mode in WandbMode],
        default=WandbMode.online.value,
    )
    parser.add_argument(CliFlag.dry_run.value, action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_matrix(
        output_dir=args.output_dir,
        config_root=args.config_root,
        benchmarl_root=args.benchmarl_root,
        seeds=tuple(args.seeds),
        python_executable=args.python_executable,
        wandb_enabled=args.wandb_mode != WandbMode.disabled.value,
        dry_run=bool(args.dry_run),
    )
    print(f"Prepared matrix artifacts for {len(outputs)} runs")
    return SUCCESS_RETURN_CODE


def _matrix_output(
    matrix_config: MatrixConfig,
    seed: int,
    run_dir: Path,
    checkpoint_path: Path,
) -> MatrixRunOutput:
    return MatrixRunOutput(
        env_name=matrix_config.env_name,
        config_id=matrix_config.config_id,
        seed=seed,
        progress_percent=FINAL_PROGRESS_PERCENT,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        trajectory_path=run_dir / MatrixArtifactName.trajectory_final.value,
        diagnostics_path=run_dir / MatrixArtifactName.diagnostics_final.value,
        null_diagnostics_path=run_dir / MatrixArtifactName.diagnostics_null_final.value,
        behavioral_metrics_path=run_dir / MatrixArtifactName.behavioral_metrics_final.value,
    )


def _export_command(output: MatrixRunOutput, python_executable: str) -> list[str]:
    return [
        python_executable,
        MatrixScriptPath.export_trajectory.value,
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


def _diagnostics_command(output: MatrixRunOutput, python_executable: str) -> list[str]:
    return [
        python_executable,
        MatrixScriptPath.compute_diagnostics.value,
        CliFlag.trajectory.value,
        str(output.trajectory_path),
        CliFlag.diagnostics_output.value,
        str(output.diagnostics_path),
        CliFlag.null_output.value,
        str(output.null_diagnostics_path),
    ]


def _behavioral_command(output: MatrixRunOutput, python_executable: str) -> list[str]:
    return [
        python_executable,
        MatrixScriptPath.compute_behavioral_metrics.value,
        CliFlag.trajectory.value,
        str(output.trajectory_path),
        CliFlag.output.value,
        str(output.behavioral_metrics_path),
    ]


def _checkpointed_diagnostics_command(
    output: MatrixRunOutput,
    python_executable: str,
) -> list[str]:
    return [
        python_executable,
        MatrixScriptPath.run_checkpointed_diagnostics.value,
        CliFlag.run_dir.value,
        str(output.run_dir),
        CliFlag.env_name.value,
        output.env_name,
        CliFlag.config_id.value,
        output.config_id,
    ]


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
    return output_dir / env_name / config_id / f"{MatrixDirectoryName.seed.value}_{seed}"


def _default_final_checkpoint(run_dir: Path) -> Path:
    return (
        run_dir
        / MatrixDirectoryName.checkpoint_final.value
        / MatrixArtifactName.generic_checkpoint.value
    )


def _required_string(source: Mapping[str, object], key: ConfigKey) -> str:
    value = source.get(key.value)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key.value} must be a non-empty string")
    return value


def _required_mapping(
    source: Mapping[str, object],
    key: ConfigKey,
) -> Mapping[str, object]:
    value = source.get(key.value)
    if not isinstance(value, Mapping):
        raise ValueError(f"{ErrorMessage.missing_mapping.value}: {key.value}")
    return {str(item_key): item for item_key, item in value.items()}


def _required_benchmarl_value(
    matrix_config: MatrixConfig,
    benchmarl_key: BenchmarlKey,
) -> str:
    value = matrix_config.benchmarl.get(benchmarl_key.value)
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
