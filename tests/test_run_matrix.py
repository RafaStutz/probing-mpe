from __future__ import annotations

import tempfile
import unittest
from enum import Enum
from pathlib import Path

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


BENCHMARL_ROOT = Path("/tmp/BenchMARL")
PYTHON_EXECUTABLE = "python"
COMMANDS_PER_STANDARD_RUN = 4
CHECKPOINTED_RUN_COUNT = 6
CHECKPOINTED_COMMANDS_PER_RUN = 1
EXPECTED_MATRIX_COMMANDS = (
    EXPECTED_MATRIX_RUNS * COMMANDS_PER_STANDARD_RUN
    + CHECKPOINTED_RUN_COUNT * CHECKPOINTED_COMMANDS_PER_RUN
)
FIRST_OUTPUT_INDEX = 0
LAST_OUTPUT_INDEX = -1


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


class CommandToken(str, Enum):
    checkpointed = "scripts/run_checkpointed_diagnostics.py"
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


if __name__ == "__main__":
    unittest.main()
