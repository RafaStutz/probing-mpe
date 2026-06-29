from __future__ import annotations

import tempfile
import unittest
import json
import pickle
from enum import Enum
from pathlib import Path

import numpy as np

from probing_mpe.evaluation import DiagnosticJsonKey
from probing_mpe.experiments.artifacts import (
    ArtifactFileName,
    CheckpointMetadataKey,
    DirectoryName,
    MetadataKey,
    run_artifact_paths,
)
from probing_mpe.experiments.run_matrix import (
    CHECKPOINTED_CONFIG_ID,
    DEFAULT_CONFIG_ROOT,
    DEFAULT_SEEDS,
    EXPECTED_MATRIX_RUNS,
    MatrixConfigId,
    MatrixEnvName,
    MatrixScriptPath,
    build_matrix_plan,
    build_training_command,
    load_matrix_config,
    run_matrix,
)
from probing_mpe.metrics import BehavioralMetricKey
from probing_mpe.trajectories import GroupName, TrajectoryKey


BENCHMARL_ROOT = Path("/tmp/BenchMARL")
PYTHON_EXECUTABLE = "python"
COMMANDS_PER_STANDARD_RUN = 4
CHECKPOINTED_RUN_COUNT = 6
CHECKPOINTED_COMMANDS_PER_RUN = 1
SINGLE_SEED = (0,)
SINGLE_SEED_MATRIX_RUNS = 8
SINGLE_SEED_CHECKPOINTED_RUN_COUNT = 2
BENCHMARL_EXPERIMENT_DIR = "benchmarl_experiment"
FINAL_FRAME = 10_000_000
EXPECTED_MATRIX_COMMANDS = (
    EXPECTED_MATRIX_RUNS * COMMANDS_PER_STANDARD_RUN
    + CHECKPOINTED_RUN_COUNT * CHECKPOINTED_COMMANDS_PER_RUN
)
FIRST_OUTPUT_INDEX = 0
LAST_OUTPUT_INDEX = -1
CHECKPOINT_CONTENT = "checkpoint"


class HydraToken(str, Enum):
    mappo = "algorithm=mappo"
    speaker_listener = "task=pettingzoo/simple_speaker_listener"
    mlp = "model=layers/mlp"
    critic_model = "model@critic_model=layers/mlp"
    seed_two = "seed=2"
    no_wandb = "experiment.loggers=[csv]"
    no_policy_sharing = "experiment.share_policy_params=false"
    no_critic_sharing = "algorithm.share_param_critic=false"
    discrete_actions = "experiment.prefer_continuous_actions=false"
    sampling_device = "experiment.sampling_device=cuda:0"
    train_device = "experiment.train_device=cuda:0"
    buffer_device = "experiment.buffer_device=cuda:0"


class CommandToken(str, Enum):
    training_script = "run.py"
    checkpointed = "scripts/run_checkpointed_diagnostics.py"
    export_script = "scripts/export_benchmarl_trajectory.py"
    diagnostics_script = "scripts/compute_diagnostics_from_trajectory.py"
    behavioral_script = "scripts/compute_behavioral_metrics.py"
    final_trajectory = "trajectory_eval_final.pkl"
    final_diagnostics = "diagnostics_final.json"
    final_behavioral = "behavioral_metrics_final.json"


class MatrixRunnerTest(unittest.TestCase):
    def test_build_matrix_plan_contains_full_24_run_design(self) -> None:
        plan = build_matrix_plan(DEFAULT_CONFIG_ROOT, DEFAULT_SEEDS)

        self.assertEqual(len(plan), EXPECTED_MATRIX_RUNS)
        self.assertEqual(plan[FIRST_OUTPUT_INDEX].env_name, MatrixEnvName.simple_spread)
        self.assertEqual(plan[FIRST_OUTPUT_INDEX].config_id, MatrixConfigId.ippo_ff)
        self.assertEqual(plan[FIRST_OUTPUT_INDEX].seed, DEFAULT_SEEDS[FIRST_OUTPUT_INDEX])
        self.assertEqual(
            plan[LAST_OUTPUT_INDEX].env_name,
            MatrixEnvName.simple_speaker_listener,
        )
        self.assertEqual(plan[LAST_OUTPUT_INDEX].config_id, MatrixConfigId.mappo_rnn)
        self.assertEqual(plan[LAST_OUTPUT_INDEX].seed, DEFAULT_SEEDS[LAST_OUTPUT_INDEX])

    def test_build_training_command_uses_selected_config_and_disables_wandb(self) -> None:
        config_path = (
            DEFAULT_CONFIG_ROOT
            / MatrixEnvName.simple_speaker_listener.value
            / f"{MatrixConfigId.mappo_ff.value}.yaml"
        )
        matrix_config = load_matrix_config(config_path)

        command = build_training_command(
            matrix_config=matrix_config,
            benchmarl_root=BENCHMARL_ROOT,
            run_dir=Path("/tmp/run"),
            seed=DEFAULT_SEEDS[LAST_OUTPUT_INDEX],
            python_executable=PYTHON_EXECUTABLE,
            wandb_enabled=False,
        )

        self.assertIn(HydraToken.mappo.value, command)
        self.assertIn(HydraToken.speaker_listener.value, command)
        self.assertIn(HydraToken.mlp.value, command)
        self.assertIn(HydraToken.critic_model.value, command)
        self.assertIn(HydraToken.seed_two.value, command)
        self.assertIn(HydraToken.no_wandb.value, command)
        self.assertIn(HydraToken.no_policy_sharing.value, command)
        self.assertIn(HydraToken.no_critic_sharing.value, command)
        self.assertIn(HydraToken.discrete_actions.value, command)
        self.assertIn(HydraToken.sampling_device.value, command)
        self.assertIn(HydraToken.train_device.value, command)
        self.assertIn(HydraToken.buffer_device.value, command)

    def test_run_matrix_runs_all_final_artifacts_and_checkpointed_mappo_rnn(self) -> None:
        commands: list[list[str]] = []

        def run_command(command: list[str], cwd: Path | None) -> int:
            commands.append(command)
            return 0

        def resolve_checkpoint(run_dir: Path) -> Path:
            return run_dir / "checkpoint_final" / "checkpoint.pt"

        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = run_matrix(
                output_dir=Path(temp_dir),
                config_root=DEFAULT_CONFIG_ROOT,
                benchmarl_root=BENCHMARL_ROOT,
                seeds=DEFAULT_SEEDS,
                python_executable=PYTHON_EXECUTABLE,
                wandb_enabled=False,
                dry_run=False,
                command_runner=run_command,
                checkpoint_resolver=resolve_checkpoint,
            )

        self.assertEqual(len(outputs), EXPECTED_MATRIX_RUNS)
        self.assertEqual(len(commands), EXPECTED_MATRIX_COMMANDS)
        self.assertTrue(
            all(
                output.trajectory_path.name == CommandToken.final_trajectory.value
                for output in outputs
            )
        )
        self.assertTrue(
            all(
                output.diagnostics_path.name == CommandToken.final_diagnostics.value
                for output in outputs
            )
        )
        self.assertTrue(
            all(
                output.behavioral_metrics_path.name == CommandToken.final_behavioral.value
                for output in outputs
            )
        )

        checkpointed_commands = [
            command
            for command in commands
            if MatrixScriptPath.run_checkpointed_diagnostics.value in command
        ]
        self.assertEqual(len(checkpointed_commands), CHECKPOINTED_RUN_COUNT)
        self.assertTrue(
            all(CHECKPOINTED_CONFIG_ID in command for command in checkpointed_commands)
        )

    def test_run_matrix_skips_complete_runs_by_default_and_writes_metadata(self) -> None:
        commands: list[list[str]] = []

        def run_command(command: list[str], cwd: Path | None) -> int:
            commands.append(command)
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            _write_complete_matrix_artifacts(output_dir, checkpointed=True)

            outputs = run_matrix(
                output_dir=output_dir,
                config_root=DEFAULT_CONFIG_ROOT,
                benchmarl_root=BENCHMARL_ROOT,
                seeds=SINGLE_SEED,
                python_executable=PYTHON_EXECUTABLE,
                wandb_enabled=False,
                dry_run=False,
                command_runner=run_command,
                force=False,
            )

            metadata_path = (
                output_dir
                / MatrixEnvName.simple_spread.value
                / MatrixConfigId.ippo_ff.value
                / "seed_0"
                / ArtifactFileName.run_metadata.value
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(len(outputs), SINGLE_SEED_MATRIX_RUNS)
        self.assertEqual(commands, [])
        self.assertEqual(metadata[MetadataKey.status.value], "complete")

    def test_run_matrix_resumes_from_final_checkpoint_and_runs_missing_postprocessing(self) -> None:
        commands: list[list[str]] = []

        def run_command(command: list[str], cwd: Path | None) -> int:
            commands.append(command)
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            _write_matrix_checkpoints(output_dir)

            run_matrix(
                output_dir=output_dir,
                config_root=DEFAULT_CONFIG_ROOT,
                benchmarl_root=BENCHMARL_ROOT,
                seeds=SINGLE_SEED,
                python_executable=PYTHON_EXECUTABLE,
                wandb_enabled=False,
                dry_run=False,
                command_runner=run_command,
                force=False,
            )

        training_commands = [
            command
            for command in commands
            if command[1].endswith(CommandToken.training_script.value)
        ]
        self.assertEqual(training_commands, [])
        self.assertEqual(
            sum(CommandToken.export_script.value in command for command in commands),
            SINGLE_SEED_MATRIX_RUNS,
        )
        self.assertEqual(
            sum(CommandToken.diagnostics_script.value in command for command in commands),
            SINGLE_SEED_MATRIX_RUNS,
        )
        self.assertEqual(
            sum(CommandToken.behavioral_script.value in command for command in commands),
            SINGLE_SEED_MATRIX_RUNS,
        )
        self.assertEqual(
            sum(CommandToken.checkpointed.value in command for command in commands),
            SINGLE_SEED_CHECKPOINTED_RUN_COUNT,
        )

    def test_run_matrix_exports_from_original_benchmarl_checkpoint_after_normalization(self) -> None:
        commands: list[list[str]] = []

        def run_command(command: list[str], cwd: Path | None) -> int:
            commands.append(command)
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            raw_checkpoints = _write_reloadable_matrix_checkpoints(output_dir)

            run_matrix(
                output_dir=output_dir,
                config_root=DEFAULT_CONFIG_ROOT,
                benchmarl_root=BENCHMARL_ROOT,
                seeds=SINGLE_SEED,
                python_executable=PYTHON_EXECUTABLE,
                wandb_enabled=False,
                dry_run=False,
                command_runner=run_command,
                force=False,
            )
            metadata_path = (
                output_dir
                / MatrixEnvName.simple_spread.value
                / MatrixConfigId.ippo_ff.value
                / "seed_0"
                / ArtifactFileName.run_metadata.value
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_frame = metadata[MetadataKey.final_checkpoint.value][
                CheckpointMetadataKey.frame.value
            ]

        export_commands = [
            command
            for command in commands
            if CommandToken.export_script.value in command
        ]
        exported_checkpoints = {
            Path(command[command.index("--checkpoint") + 1])
            for command in export_commands
        }
        self.assertTrue(raw_checkpoints.issubset(exported_checkpoints))
        self.assertEqual(metadata_frame, FINAL_FRAME)

    def test_run_matrix_resumes_from_raw_benchmarl_checkpoint_without_retraining(self) -> None:
        commands: list[list[str]] = []

        def run_command(command: list[str], cwd: Path | None) -> int:
            commands.append(command)
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            raw_checkpoints = _write_raw_matrix_checkpoints(output_dir)

            run_matrix(
                output_dir=output_dir,
                config_root=DEFAULT_CONFIG_ROOT,
                benchmarl_root=BENCHMARL_ROOT,
                seeds=SINGLE_SEED,
                python_executable=PYTHON_EXECUTABLE,
                wandb_enabled=False,
                dry_run=False,
                command_runner=run_command,
                force=False,
            )

        training_commands = [
            command
            for command in commands
            if command[1].endswith(CommandToken.training_script.value)
        ]
        export_commands = [
            command
            for command in commands
            if CommandToken.export_script.value in command
        ]
        exported_checkpoints = {
            Path(command[command.index("--checkpoint") + 1])
            for command in export_commands
        }
        self.assertEqual(training_commands, [])
        self.assertTrue(raw_checkpoints.issubset(exported_checkpoints))

    def test_run_matrix_force_ignores_complete_resume_state(self) -> None:
        commands: list[list[str]] = []

        def run_command(command: list[str], cwd: Path | None) -> int:
            commands.append(command)
            return 0

        def resolve_checkpoint(run_dir: Path) -> Path:
            return run_dir / "checkpoint_final" / "checkpoint.pt"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            _write_complete_matrix_artifacts(output_dir, checkpointed=True)

            run_matrix(
                output_dir=output_dir,
                config_root=DEFAULT_CONFIG_ROOT,
                benchmarl_root=BENCHMARL_ROOT,
                seeds=SINGLE_SEED,
                python_executable=PYTHON_EXECUTABLE,
                wandb_enabled=False,
                dry_run=False,
                command_runner=run_command,
                checkpoint_resolver=resolve_checkpoint,
                force=True,
            )

        training_commands = [
            command
            for command in commands
            if command[1].endswith(CommandToken.training_script.value)
        ]
        self.assertEqual(len(training_commands), SINGLE_SEED_MATRIX_RUNS)


if __name__ == "__main__":
    unittest.main()


def _write_complete_matrix_artifacts(output_dir: Path, checkpointed: bool) -> None:
    _write_matrix_checkpoints(output_dir)
    for plan_entry in build_matrix_plan(DEFAULT_CONFIG_ROOT, SINGLE_SEED):
        paths = run_artifact_paths(
            env_name=plan_entry.env_name.value,
            config_id=plan_entry.config_id.value,
            seed=plan_entry.seed,
            run_dir=(
                output_dir
                / plan_entry.env_name.value
                / plan_entry.config_id.value
                / "seed_0"
            ),
        )
        _write_valid_final_artifacts(paths)
        if checkpointed and plan_entry.config_id.value == CHECKPOINTED_CONFIG_ID:
            _write_valid_checkpointed_artifacts(paths.run_dir)


def _write_matrix_checkpoints(output_dir: Path) -> None:
    for plan_entry in build_matrix_plan(DEFAULT_CONFIG_ROOT, SINGLE_SEED):
        paths = run_artifact_paths(
            env_name=plan_entry.env_name.value,
            config_id=plan_entry.config_id.value,
            seed=plan_entry.seed,
            run_dir=(
                output_dir
                / plan_entry.env_name.value
                / plan_entry.config_id.value
                / "seed_0"
            ),
        )
        paths.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        paths.checkpoint_path.write_text(CHECKPOINT_CONTENT, encoding="utf-8")


def _write_reloadable_matrix_checkpoints(output_dir: Path) -> set[Path]:
    raw_checkpoints: set[Path] = set()
    for plan_entry in build_matrix_plan(DEFAULT_CONFIG_ROOT, SINGLE_SEED):
        paths = run_artifact_paths(
            env_name=plan_entry.env_name.value,
            config_id=plan_entry.config_id.value,
            seed=plan_entry.seed,
            run_dir=(
                output_dir
                / plan_entry.env_name.value
                / plan_entry.config_id.value
                / "seed_0"
            ),
        )
        raw_checkpoint = (
            paths.run_dir
            / BENCHMARL_EXPERIMENT_DIR
            / DirectoryName.checkpoints.value
            / f"checkpoint_{FINAL_FRAME}.pt"
        )
        raw_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        raw_checkpoint.write_text(CHECKPOINT_CONTENT, encoding="utf-8")
        paths.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        paths.checkpoint_path.write_text(CHECKPOINT_CONTENT, encoding="utf-8")
        raw_checkpoints.add(raw_checkpoint)
    return raw_checkpoints


def _write_raw_matrix_checkpoints(output_dir: Path) -> set[Path]:
    raw_checkpoints: set[Path] = set()
    for plan_entry in build_matrix_plan(DEFAULT_CONFIG_ROOT, SINGLE_SEED):
        paths = run_artifact_paths(
            env_name=plan_entry.env_name.value,
            config_id=plan_entry.config_id.value,
            seed=plan_entry.seed,
            run_dir=(
                output_dir
                / plan_entry.env_name.value
                / plan_entry.config_id.value
                / "seed_0"
            ),
        )
        raw_checkpoint = (
            paths.run_dir
            / BENCHMARL_EXPERIMENT_DIR
            / DirectoryName.checkpoints.value
            / f"checkpoint_{FINAL_FRAME}.pt"
        )
        raw_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        raw_checkpoint.write_text(CHECKPOINT_CONTENT, encoding="utf-8")
        raw_checkpoints.add(raw_checkpoint)
    return raw_checkpoints


def _write_valid_final_artifacts(paths: object) -> None:
    trajectory_path = getattr(paths, "trajectory_path")
    with trajectory_path.open("wb") as trajectory_file:
        pickle.dump(_valid_trajectory(), trajectory_file)
    getattr(paths, "diagnostics_path").write_text(
        json.dumps({DiagnosticJsonKey.diagnostics.value: {}}),
        encoding="utf-8",
    )
    getattr(paths, "null_diagnostics_path").write_text(
        json.dumps({DiagnosticJsonKey.null_diagnostics.value: {}}),
        encoding="utf-8",
    )
    getattr(paths, "behavioral_metrics_path").write_text(
        json.dumps({BehavioralMetricKey.behavioral_metrics.value: {}}),
        encoding="utf-8",
    )


def _write_valid_checkpointed_artifacts(run_dir: Path) -> None:
    for progress in (25, 50, 75, 100):
        trajectory_path = (
            run_dir
            / DirectoryName.trajectories_by_progress.value
            / f"trajectory_eval_{progress}.pkl"
        )
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        with trajectory_path.open("wb") as trajectory_file:
            pickle.dump(_valid_trajectory(), trajectory_file)
        diagnostics_dir = run_dir / DirectoryName.diagnostics_by_progress.value
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        (diagnostics_dir / f"diagnostics_{progress}.json").write_text(
            json.dumps({DiagnosticJsonKey.diagnostics.value: {}}),
            encoding="utf-8",
        )
        (diagnostics_dir / f"diagnostics_null_{progress}.json").write_text(
            json.dumps({DiagnosticJsonKey.null_diagnostics.value: {}}),
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
        TrajectoryKey.seed.value: 0,
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
