from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence, cast

import yaml


DEFAULT_CONFIG_PATH = Path("configs/reduced_mpe/simple_spread_v3/ippo_rnn.yaml")
DEFAULT_BENCHMARL_ROOT = Path("/tmp/BenchMARL")
DEFAULT_OUTPUT_DIR = Path("/tmp/probing_mpe_benchmarl_smoke")
DEFAULT_SEED = 0
DEFAULT_DEVICE = "cuda:0"
SMOKE_MAX_FRAMES = 4096
SMOKE_BATCH_MULTIPLIER = 2
SMOKE_OPTIMIZER_EPOCHS = 1
SMOKE_EVALUATION_EPISODES = 2


class ConfigKey(str, Enum):
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
    checkpoint_interval = "experiment.checkpoint_interval"
    create_json = "experiment.create_json"
    evaluation_episodes = "experiment.evaluation_episodes"
    evaluation_interval = "experiment.evaluation_interval"
    keep_checkpoints_num = "experiment.keep_checkpoints_num"
    loggers = "experiment.loggers"
    max_n_frames = "experiment.max_n_frames"
    on_policy_collected_frames_per_batch = (
        "experiment.on_policy_collected_frames_per_batch"
    )
    on_policy_n_minibatch_iters = "experiment.on_policy_n_minibatch_iters"
    prefer_continuous_actions = "experiment.prefer_continuous_actions"
    render = "experiment.render"
    sampling_device = "experiment.sampling_device"
    save_folder = "experiment.save_folder"
    share_param_critic = "algorithm.share_param_critic"
    share_policy_params = "experiment.share_policy_params"
    train_device = "experiment.train_device"


class SmokeBudget:
    max_frames = SMOKE_MAX_FRAMES
    batch_multiplier = SMOKE_BATCH_MULTIPLIER
    optimizer_epochs = SMOKE_OPTIMIZER_EPOCHS
    evaluation_episodes = SMOKE_EVALUATION_EPISODES


@dataclass(frozen=True)
class SmokeConfig:
    benchmarl: Mapping[str, object]
    overrides: Mapping[str, object]


def load_smoke_config(config_path: Path) -> SmokeConfig:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file is not a mapping: {config_path}")

    benchmarl = loaded.get(ConfigKey.benchmarl.value)
    overrides = loaded.get(ConfigKey.overrides.value)
    if not isinstance(benchmarl, dict):
        raise ValueError(f"Missing mapping: {ConfigKey.benchmarl.value}")
    if not isinstance(overrides, dict):
        raise ValueError(f"Missing mapping: {ConfigKey.overrides.value}")

    return SmokeConfig(
        benchmarl=cast(Mapping[str, object], benchmarl),
        overrides=cast(Mapping[str, object], overrides),
    )


def validate_required_overrides(smoke_config: SmokeConfig) -> None:
    expected_false_keys = (
        OverrideKey.share_policy_params,
        OverrideKey.share_param_critic,
        OverrideKey.prefer_continuous_actions,
    )
    for override_key in expected_false_keys:
        actual_value = smoke_config.overrides.get(override_key.value)
        if actual_value is not False:
            raise ValueError(f"{override_key.value} must be false, got {actual_value}")


def build_command(
    smoke_config: SmokeConfig,
    benchmarl_root: Path,
    output_dir: Path,
    seed: int,
    python_executable: str,
    device: str = DEFAULT_DEVICE,
) -> list[str]:
    validate_required_overrides(smoke_config)

    benchmarl_run = benchmarl_root / "benchmarl" / "run.py"
    base_command = [
        python_executable,
        str(benchmarl_run),
        f"{BenchmarlKey.algorithm.value}={_required_benchmarl_value(smoke_config, BenchmarlKey.algorithm)}",
        f"{BenchmarlKey.task.value}={_required_benchmarl_value(smoke_config, BenchmarlKey.task)}",
        f"{BenchmarlKey.model.value}={_required_benchmarl_value(smoke_config, BenchmarlKey.model)}",
        f"{HydraGroupKey.critic_model.value}={_required_benchmarl_value(smoke_config, BenchmarlKey.critic_model)}",
        f"seed={seed}",
    ]

    frames_per_batch = _required_positive_int(
        smoke_config, OverrideKey.on_policy_collected_frames_per_batch
    )
    smoke_max_frames = max(
        SmokeBudget.max_frames, frames_per_batch * SmokeBudget.batch_multiplier
    )

    smoke_overrides = dict(smoke_config.overrides)
    _apply_device_overrides(smoke_overrides, device)
    smoke_overrides[OverrideKey.max_n_frames.value] = smoke_max_frames
    smoke_overrides[
        OverrideKey.on_policy_n_minibatch_iters.value
    ] = SmokeBudget.optimizer_epochs
    smoke_overrides[
        OverrideKey.evaluation_episodes.value
    ] = SmokeBudget.evaluation_episodes
    smoke_overrides[OverrideKey.evaluation_interval.value] = frames_per_batch
    smoke_overrides[OverrideKey.checkpoint_interval.value] = frames_per_batch
    smoke_overrides[OverrideKey.checkpoint_at_end.value] = True
    smoke_overrides[OverrideKey.keep_checkpoints_num.value] = None
    smoke_overrides[OverrideKey.loggers.value] = ["csv"]
    smoke_overrides[OverrideKey.render.value] = False
    smoke_overrides[OverrideKey.create_json.value] = True
    smoke_overrides[OverrideKey.save_folder.value] = str(output_dir)

    return base_command + [
        f"{key}={_format_hydra_value(value)}"
        for key, value in sorted(smoke_overrides.items())
    ]


def run_command(command: Sequence[str], benchmarl_root: Path) -> int:
    if not (benchmarl_root / "benchmarl" / "run.py").exists():
        raise FileNotFoundError(f"BenchMARL checkout not found: {benchmarl_root}")
    completed = subprocess.run(command, cwd=benchmarl_root, check=False)
    return completed.returncode


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or print the disk-safe BenchMARL smoke-test command."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--benchmarl-root", type=Path, default=DEFAULT_BENCHMARL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    smoke_config = load_smoke_config(args.config)
    command = build_command(
        smoke_config=smoke_config,
        benchmarl_root=args.benchmarl_root,
        output_dir=args.output_dir,
        seed=args.seed,
        python_executable=args.python_executable,
        device=args.device,
    )
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0
    prepare_output_dir(args.output_dir)
    return run_command(command, args.benchmarl_root)


def _required_benchmarl_value(
    smoke_config: SmokeConfig, benchmarl_key: BenchmarlKey
) -> str:
    value = smoke_config.benchmarl.get(benchmarl_key.value)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing BenchMARL value: {benchmarl_key.value}")
    return value


def _required_positive_int(
    smoke_config: SmokeConfig, override_key: OverrideKey
) -> int:
    value = smoke_config.overrides.get(override_key.value)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{override_key.value} must be a positive integer")
    return value


def _format_hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ",".join(_format_hydra_value(item) for item in value) + "]"
    return str(value)


def _apply_device_overrides(overrides: dict[str, object], device: str) -> None:
    overrides[OverrideKey.sampling_device.value] = device
    overrides[OverrideKey.train_device.value] = device
    overrides[OverrideKey.buffer_device.value] = device


if __name__ == "__main__":
    raise SystemExit(main())
