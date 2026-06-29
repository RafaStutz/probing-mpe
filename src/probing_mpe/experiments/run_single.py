from __future__ import annotations

import argparse
import sys
from enum import Enum
from pathlib import Path

from probing_mpe.experiments.artifacts import (
    MetadataStatus,
    NormalizedCheckpoint,
    behavioral_metrics_are_valid,
    checkpointed_artifacts_are_valid,
    diagnostics_are_valid,
    null_diagnostics_are_valid,
    run_artifact_paths,
    run_is_complete,
    trajectory_is_valid,
)
from probing_mpe.experiments.run_matrix import (
    CHECKPOINTED_CONFIG_ID,
    DEFAULT_BENCHMARL_ROOT,
    DEFAULT_CONFIG_ROOT,
    DEFAULT_DEVICE,
    _behavioral_command,
    _checkpointed_diagnostics_command,
    _commands_for_output,
    _diagnostics_command,
    _export_command,
    _matrix_output,
    _run_dir,
    _run_or_print,
    _write_metadata_for_output,
    build_training_command,
    discover_final_checkpoint,
    load_matrix_config,
    run_command,
)


SUCCESS_RETURN_CODE = 0
CONFIG_SUFFIX = ".yaml"


class CliFlag(str, Enum):
    env = "--env"
    config = "--config"
    seed = "--seed"
    wandb = "--wandb"
    output_dir = "--output-dir"
    config_root = "--config-root"
    benchmarl_root = "--benchmarl-root"
    dry_run = "--dry-run"
    force = "--force"
    device = "--device"


class WandbMode(str, Enum):
    disabled = "disabled"


class ErrorMessage(str, Enum):
    config_missing = "Config file not found"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single training job for the replication plan."
    )
    parser.add_argument(
        CliFlag.env.value,
        required=True,
        help="Environment name, for example simple_spread_v3.",
    )
    parser.add_argument(
        CliFlag.config.value,
        required=True,
        help="Config ID, for example ippo_rnn.",
    )
    parser.add_argument(CliFlag.seed.value, type=int, required=True)
    parser.add_argument(
        CliFlag.wandb.value,
        default="online",
        help="W&B mode: online, offline, or disabled.",
    )
    parser.add_argument(
        CliFlag.output_dir.value,
        type=Path,
        default=Path("/workspace/results"),
    )
    parser.add_argument(
        CliFlag.config_root.value,
        type=Path,
        default=DEFAULT_CONFIG_ROOT,
    )
    parser.add_argument(
        CliFlag.benchmarl_root.value,
        type=Path,
        default=DEFAULT_BENCHMARL_ROOT,
    )
    parser.add_argument(CliFlag.dry_run.value, action="store_true")
    parser.add_argument(CliFlag.force.value, action="store_true")
    parser.add_argument(CliFlag.device.value, default=DEFAULT_DEVICE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config_root / args.env / f"{args.config}{CONFIG_SUFFIX}"
    if not config_path.exists():
        print(f"{ErrorMessage.config_missing.value}: {config_path}", file=sys.stderr)
        return 1

    matrix_config = load_matrix_config(config_path)
    run_dir = _run_dir(args.output_dir, args.env, args.config, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = run_artifact_paths(
        env_name=args.env,
        config_id=args.config,
        seed=args.seed,
        run_dir=run_dir,
    )

    wandb_enabled = args.wandb != WandbMode.disabled.value
    checkpointed_required = args.config == CHECKPOINTED_CONFIG_ID
    training_command = build_training_command(
        matrix_config=matrix_config,
        benchmarl_root=args.benchmarl_root,
        run_dir=run_dir,
        seed=args.seed,
        python_executable=sys.executable,
        wandb_enabled=wandb_enabled,
        device=args.device,
    )
    output = _matrix_output(
        matrix_config=matrix_config,
        seed=args.seed,
        run_dir=run_dir,
        checkpoint_path=artifact_paths.checkpoint_path,
    )
    commands = _commands_for_output(
        output=output,
        training_command=training_command,
        python_executable=sys.executable,
        checkpointed_required=checkpointed_required,
    )

    if not args.force and not args.dry_run and run_is_complete(
        artifact_paths,
        checkpointed_required,
    ):
        _write_metadata_for_output(
            artifact_paths=artifact_paths,
            status=MetadataStatus.complete,
            benchmarl_root=args.benchmarl_root,
            python_executable=sys.executable,
            wandb_enabled=wandb_enabled,
            commands=commands,
            final_checkpoint=NormalizedCheckpoint(
                source_path=artifact_paths.checkpoint_path,
                normalized_path=artifact_paths.checkpoint_path,
                frame=None,
            ),
            include_progress=checkpointed_required,
        )
        print(f"Run single already complete. Results in {run_dir}")
        return SUCCESS_RETURN_CODE

    _write_metadata_for_output(
        artifact_paths=artifact_paths,
        status=MetadataStatus.started,
        benchmarl_root=args.benchmarl_root,
        python_executable=sys.executable,
        wandb_enabled=wandb_enabled,
        commands=commands,
        final_checkpoint=None,
        include_progress=False,
    )

    if args.force or args.dry_run or not artifact_paths.checkpoint_path.exists():
        print(f"Starting training run for {args.env} / {args.config} / seed {args.seed}...")
        _run_or_print(training_command, args.benchmarl_root, args.dry_run, run_command)

    checkpoint_path = (
        artifact_paths.checkpoint_path
        if args.dry_run
        else discover_final_checkpoint(run_dir)
    )
    output = _matrix_output(
        matrix_config=matrix_config,
        seed=args.seed,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
    )
    commands = _commands_for_output(
        output=output,
        training_command=training_command,
        python_executable=sys.executable,
        checkpointed_required=checkpointed_required,
    )
    final_checkpoint = NormalizedCheckpoint(
        source_path=checkpoint_path,
        normalized_path=checkpoint_path,
        frame=None,
    )
    _write_metadata_for_output(
        artifact_paths=artifact_paths,
        status=MetadataStatus.training_complete,
        benchmarl_root=args.benchmarl_root,
        python_executable=sys.executable,
        wandb_enabled=wandb_enabled,
        commands=commands,
        final_checkpoint=final_checkpoint,
        include_progress=False,
    )

    if args.force or args.dry_run or not trajectory_is_valid(output.trajectory_path):
        print("Exporting trajectory...")
        _run_or_print(_export_command(output, sys.executable), None, args.dry_run, run_command)

    if (
        args.force
        or args.dry_run
        or not diagnostics_are_valid(output.diagnostics_path)
        or not null_diagnostics_are_valid(output.null_diagnostics_path)
    ):
        print("Computing diagnostics...")
        _run_or_print(
            _diagnostics_command(output, sys.executable),
            None,
            args.dry_run,
            run_command,
        )

    if args.force or args.dry_run or not behavioral_metrics_are_valid(
        output.behavioral_metrics_path
    ):
        print("Computing behavioral metrics...")
        _run_or_print(
            _behavioral_command(output, sys.executable),
            None,
            args.dry_run,
            run_command,
        )

    if checkpointed_required and (
        args.force or args.dry_run or not checkpointed_artifacts_are_valid(run_dir)
    ):
        print("Running checkpointed diagnostics for MAPPO-RNN...")
        _run_or_print(
            _checkpointed_diagnostics_command(output, sys.executable),
            None,
            args.dry_run,
            run_command,
        )

    _write_metadata_for_output(
        artifact_paths=artifact_paths,
        status=MetadataStatus.complete,
        benchmarl_root=args.benchmarl_root,
        python_executable=sys.executable,
        wandb_enabled=wandb_enabled,
        commands=commands,
        final_checkpoint=final_checkpoint,
        include_progress=checkpointed_required and not args.dry_run,
    )

    print(f"Run single completed successfully. Results in {run_dir}")
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    sys.exit(main())
