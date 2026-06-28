from __future__ import annotations

import tempfile
import unittest
from enum import Enum
from pathlib import Path

from probing_mpe.experiments.run_full_cell import (
    DEFAULT_SEEDS,
    FULL_CELL_CONFIG_ID,
    FULL_CELL_ENV_NAME,
    FINAL_PROGRESS_PERCENT,
    FullCellArtifactName,
    FullCellDirectoryName,
    build_training_command,
    discover_final_checkpoint,
    load_full_cell_config,
    run_full_cell,
)


CONFIG_PATH = Path("configs/reduced_mpe/simple_spread_v3/ippo_rnn.yaml")
BENCHMARL_ROOT = Path("/tmp/BenchMARL")
PYTHON_EXECUTABLE = "python"
FIRST_SEED = 0
SECOND_SEED = 1
EXPECTED_COMMANDS_PER_SEED = 4


class HydraOverride(str, Enum):
    algorithm = "algorithm=ippo"
    task = "task=pettingzoo/simple_spread"
    model = "model=layers/gru"
    critic_model = "model@critic_model=layers/mlp"
    seed_one = "seed=1"
    max_frames = "experiment.max_n_frames=10000000"
    loggers_csv = "experiment.loggers=[csv]"


class CommandToken(str, Enum):
    export_script = "scripts/export_benchmarl_trajectory.py"
    diagnostics_script = "scripts/compute_diagnostics_from_trajectory.py"
    behavioral_script = "scripts/compute_behavioral_metrics.py"


class FullCellRunnerTest(unittest.TestCase):
    def test_build_training_command_uses_full_budget_and_disabled_wandb(self) -> None:
        full_cell_config = load_full_cell_config(CONFIG_PATH)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / FullCellDirectoryName.seed.value
            command = build_training_command(
                full_cell_config=full_cell_config,
                benchmarl_root=BENCHMARL_ROOT,
                run_dir=run_dir,
                seed=SECOND_SEED,
                python_executable=PYTHON_EXECUTABLE,
                wandb_enabled=False,
            )

        self.assertEqual(
            command[:2], [PYTHON_EXECUTABLE, str(BENCHMARL_ROOT / "benchmarl" / "run.py")]
        )
        self.assertIn(HydraOverride.algorithm.value, command)
        self.assertIn(HydraOverride.task.value, command)
        self.assertIn(HydraOverride.model.value, command)
        self.assertIn(HydraOverride.critic_model.value, command)
        self.assertIn(HydraOverride.seed_one.value, command)
        self.assertIn(HydraOverride.max_frames.value, command)
        self.assertIn(HydraOverride.loggers_csv.value, command)

    def test_discover_final_checkpoint_prefers_plan_checkpoint_final(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            checkpoint_path = (
                run_dir
                / FullCellDirectoryName.checkpoint_final.value
                / FullCellArtifactName.generic_checkpoint.value
            )
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_text(FullCellArtifactName.generic_checkpoint.value, encoding="utf-8")

            discovered = discover_final_checkpoint(run_dir)

        self.assertEqual(discovered, checkpoint_path)

    def test_run_full_cell_runs_three_seed_pipeline_in_order(self) -> None:
        commands: list[list[str]] = []
        checkpoint_by_seed = {
            seed: Path(f"/checkpoints/seed_{seed}/checkpoint.pt") for seed in DEFAULT_SEEDS
        }

        def run_command(command: list[str], cwd: Path | None) -> int:
            commands.append(command)
            return 0

        def resolve_checkpoint(run_dir: Path) -> Path:
            seed = int(run_dir.name.removeprefix(f"{FullCellDirectoryName.seed.value}_"))
            return checkpoint_by_seed[seed]

        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = run_full_cell(
                output_dir=Path(temp_dir),
                config_path=CONFIG_PATH,
                benchmarl_root=BENCHMARL_ROOT,
                seeds=DEFAULT_SEEDS,
                python_executable=PYTHON_EXECUTABLE,
                wandb_enabled=False,
                dry_run=False,
                command_runner=run_command,
                checkpoint_resolver=resolve_checkpoint,
            )

        self.assertEqual(len(outputs), len(DEFAULT_SEEDS))
        self.assertEqual(
            len(commands), len(DEFAULT_SEEDS) * EXPECTED_COMMANDS_PER_SEED
        )
        self.assertEqual([output.seed for output in outputs], list(DEFAULT_SEEDS))
        self.assertEqual(
            [output.run_dir.name for output in outputs],
            [f"{FullCellDirectoryName.seed.value}_{seed}" for seed in DEFAULT_SEEDS],
        )
        self.assertEqual(outputs[FIRST_SEED].env_name, FULL_CELL_ENV_NAME)
        self.assertEqual(outputs[FIRST_SEED].config_id, FULL_CELL_CONFIG_ID)
        self.assertEqual(outputs[FIRST_SEED].progress_percent, FINAL_PROGRESS_PERCENT)
        self.assertTrue(
            any(CommandToken.export_script.value in command for command in commands)
        )
        self.assertTrue(
            any(CommandToken.diagnostics_script.value in command for command in commands)
        )
        self.assertTrue(
            any(CommandToken.behavioral_script.value in command for command in commands)
        )


if __name__ == "__main__":
    unittest.main()
