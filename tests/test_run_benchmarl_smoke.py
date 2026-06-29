from pathlib import Path
import tempfile
import unittest

import yaml

from probing_mpe.experiments.run_benchmarl_smoke import (
    ConfigKey,
    OverrideKey,
    SmokeBudget,
    build_command,
    load_smoke_config,
    prepare_output_dir,
    validate_required_overrides,
)


CONFIG_PATH = Path("configs/reduced_mpe/simple_spread_v3/ippo_rnn.yaml")
BENCHMARL_ROOT = Path("/tmp/BenchMARL")
OUTPUT_DIR = Path("/tmp/probing_mpe_smoke")
SEED = 0


class RunBenchmarlSmokeTests(unittest.TestCase):
    def test_build_command_uses_smoke_budget_and_no_sharing(self) -> None:
        smoke_config = load_smoke_config(CONFIG_PATH)

        command = build_command(
            smoke_config=smoke_config,
            benchmarl_root=BENCHMARL_ROOT,
            output_dir=OUTPUT_DIR,
            seed=SEED,
            python_executable="python",
        )

        self.assertEqual(
            command[:2], ["python", str(BENCHMARL_ROOT / "benchmarl" / "run.py")]
        )
        self.assertIn("algorithm=ippo", command)
        self.assertIn("task=pettingzoo/simple_spread", command)
        self.assertIn("model=layers/gru", command)
        self.assertIn("model@critic_model=layers/mlp", command)
        self.assertNotIn("critic_model=layers/mlp", command)
        self.assertIn("seed=0", command)
        self.assertIn(
            f"{OverrideKey.max_n_frames.value}={SmokeBudget.max_frames}", command
        )
        self.assertIn(
            f"{OverrideKey.on_policy_n_minibatch_iters.value}={SmokeBudget.optimizer_epochs}",
            command,
        )
        self.assertIn(
            f"{OverrideKey.evaluation_episodes.value}={SmokeBudget.evaluation_episodes}",
            command,
        )
        self.assertIn(f"{OverrideKey.loggers.value}=[csv]", command)
        self.assertIn(f"{OverrideKey.share_policy_params.value}=false", command)
        self.assertIn(f"{OverrideKey.share_param_critic.value}=false", command)
        self.assertIn(f"{OverrideKey.prefer_continuous_actions.value}=false", command)
        self.assertIn(f"{OverrideKey.sampling_device.value}=cuda:0", command)
        self.assertIn(f"{OverrideKey.train_device.value}=cuda:0", command)
        self.assertIn(f"{OverrideKey.buffer_device.value}=cuda:0", command)

    def test_validate_required_overrides_rejects_parameter_sharing(self) -> None:
        data = yaml.safe_load(CONFIG_PATH.read_text())
        data[ConfigKey.overrides.value][OverrideKey.share_policy_params.value] = True
        bad_config = OUTPUT_DIR / "bad_parameter_sharing.yaml"
        bad_config.parent.mkdir(parents=True, exist_ok=True)
        bad_config.write_text(yaml.safe_dump(data), encoding="utf-8")

        smoke_config = load_smoke_config(bad_config)

        with self.assertRaisesRegex(ValueError, "share_policy_params"):
            validate_required_overrides(smoke_config)

    def test_validate_required_overrides_rejects_continuous_actions(self) -> None:
        data = yaml.safe_load(CONFIG_PATH.read_text())
        data[ConfigKey.overrides.value][
            OverrideKey.prefer_continuous_actions.value
        ] = True
        bad_config = OUTPUT_DIR / "bad_continuous_actions.yaml"
        bad_config.parent.mkdir(parents=True, exist_ok=True)
        bad_config.write_text(yaml.safe_dump(data), encoding="utf-8")

        smoke_config = load_smoke_config(bad_config)

        with self.assertRaisesRegex(ValueError, "prefer_continuous_actions"):
            validate_required_overrides(smoke_config)

    def test_prepare_output_dir_creates_save_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "benchmarl_smoke"

            prepare_output_dir(output_dir)

            self.assertTrue(output_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
