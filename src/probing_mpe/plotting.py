from __future__ import annotations

import argparse
import binascii
import csv
import json
import math
import os
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "probing_mpe_matplotlib"))

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch


DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_PLOTS_DIR = Path("plots")
SUMMARY_FILE_NAME = "analysis_summary.json"
PER_RUN_METRICS_FILE_NAME = "per_run_metrics.json"
SCENARIO_VERDICT_FILE_NAME = "scenario_verdict.json"
LEARNING_CURVES_PLOT_NAME = "learning_curves_return.png"
MEMORY_GAP_PLOT_NAME = "memory_gap.png"
DIAGNOSTIC_EVOLUTION_PLOT_NAME = "diagnostic_evolution_mappo_rnn.png"
FINAL_DIAGNOSTIC_PLOT_NAME = "final_diagnostics_above_null.png"
BEHAVIORAL_SPREAD_PLOT_NAME = "behavioral_metrics_simple_spread.png"
BEHAVIORAL_SPEAKER_LISTENER_PLOT_NAME = "behavioral_metrics_speaker_listener.png"
EVAL_RETURN_SCALAR_FILE_NAME = "eval_reward_episode_reward_mean.csv"
SCALARS_DIRECTORY_NAME = "scalars"
REQUIRED_PLOT_NAMES = (
    LEARNING_CURVES_PLOT_NAME,
    MEMORY_GAP_PLOT_NAME,
    DIAGNOSTIC_EVOLUTION_PLOT_NAME,
    FINAL_DIAGNOSTIC_PLOT_NAME,
    BEHAVIORAL_SPREAD_PLOT_NAME,
    BEHAVIORAL_SPEAKER_LISTENER_PLOT_NAME,
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JSON_INDENT = 2
SEED_PREFIX = "seed_"
CONFIG_SEPARATOR = "/"
MAPPO_RNN_CONFIG_ID = "mappo_rnn"
IPPO_ALGORITHM_LABEL = "IPPO"
MAPPO_ALGORITHM_LABEL = "MAPPO"
PNG_COLOR_MODE_RGB = 2
PNG_BIT_DEPTH = 8
PNG_COMPRESSION_METHOD = 0
PNG_FILTER_METHOD = 0
PNG_INTERLACE_METHOD = 0
PNG_NO_FILTER = 0
PNG_CHANNELS = 3
PLOT_WIDTH = 960
PLOT_HEIGHT = 540
PLOT_MARGIN_LEFT = 80
PLOT_MARGIN_RIGHT = 40
PLOT_MARGIN_TOP = 44
PLOT_MARGIN_BOTTOM = 76
PLOT_AXIS_WIDTH = 2
BAR_WIDTH = 18
LINE_WIDTH = 2
POINT_SIZE = 4
MINIMUM_PLOT_RANGE = 1.0
DEFAULT_EMPTY_MAX = 1.0
PLOT_FIGURE_WIDTH = 11.0
PLOT_FIGURE_HEIGHT = 6.4
PLOT_DPI = 140
BAR_VALUE_PADDING_RATIO = 0.02
TICK_LABEL_ROTATION_DEGREES = 35
TICK_LABEL_FONT_SIZE = 8
ANNOTATION_FONT_SIZE = 7
TITLE_FONT_SIZE = 13
AXIS_LABEL_FONT_SIZE = 10
GRID_ALPHA = 0.28
NAN_BAR_ALPHA = 0.35


class EnvName(str, Enum):
    simple_spread = "simple_spread_v3"
    simple_speaker_listener = "simple_speaker_listener_v4"


class ConfigId(str, Enum):
    ippo_ff = "ippo_ff"
    ippo_rnn = "ippo_rnn"
    mappo_ff = "mappo_ff"
    mappo_rnn = "mappo_rnn"


class DiagnosticLabel(str, Enum):
    OAR = "OAR"
    HAR = "HAR"
    PIF = "PIF"
    AA = "AA"
    DAI = "DAI"


class PlotDiagnosticLabel(str, Enum):
    OAR = "OAR"
    HAR = "HAR"
    PIF = "PIF"
    DAI = "DAI"


class BehavioralMetricName(str, Enum):
    return_mean = "eval/return_mean"
    coverage_success_rate = "eval/coverage_success_rate"
    final_landmark_distance_mean = "eval/final_landmark_distance_mean"
    collision_rate = "eval/collision_rate"
    target_success_rate = "eval/target_success_rate"
    final_target_distance_mean = "eval/final_target_distance_mean"
    wrong_landmark_rate = "eval/wrong_landmark_rate"


class PlotTitle(str, Enum):
    final_returns = "Final evaluation return"
    memory_gap = "Memory benefit: RNN return minus FF return"
    diagnostic_evolution = "MAPPO-RNN diagnostic evolution by checkpoint"
    above_null = "Final diagnostic margin above permutation null"
    behavioral_spread = "Simple Spread behavioral metrics"
    behavioral_speaker_listener = "Simple Speaker Listener behavioral metrics"


class AxisLabel(str, Enum):
    run_config = "Run configuration"
    algorithm_pair = "Environment / algorithm"
    progress = "Training progress"
    diagnostic_fraction = "Fraction above null"
    diagnostic_margin = "Diagnostic minus null"
    return_mean = "Mean return"
    return_delta = "Return delta"
    diagnostic_value = "Mean normalized diagnostic"
    behavioral_value = "Metric value"
    training_step = "Training frames"


class TrainingOverride(str, Enum):
    frames_per_batch = "experiment.on_policy_collected_frames_per_batch="


class DisplayLabel(str, Enum):
    simple_spread = "Spread"
    simple_speaker_listener = "Speaker/listener"
    ippo_ff = "IPPO FF"
    ippo_rnn = "IPPO RNN"
    mappo_ff = "MAPPO FF"
    mappo_rnn = "MAPPO RNN"
    coverage_success_rate = "Coverage success"
    final_landmark_distance_mean = "Landmark distance"
    collision_rate = "Collision rate"
    target_success_rate = "Target success"
    final_target_distance_mean = "Target distance"
    wrong_landmark_rate = "Wrong landmark"


class JsonKey(str, Enum):
    metadata = "metadata"
    diagnostics = "diagnostics"
    null_diagnostics = "null_diagnostics"
    behavioral_metrics = "behavioral_metrics"
    final_checkpoint = "final_checkpoint"
    source_path = "source_path"
    env_name = "env_name"
    config_id = "config_id"
    seed = "seed"
    normalized = "normalized"
    normalized_mean = "normalized_mean"
    training_progress_percent = "training_progress_percent"
    commands = "commands"
    training = "training"


class SummaryKey(str, Enum):
    run_count = "run_count"
    missing_artifacts = "missing_artifacts"
    per_run_metrics = "per_run_metrics"
    memory_gap = "memory_gap"
    learning_curves = "learning_curves"
    final_diagnostics_above_null = "final_diagnostics_above_null"
    final_diagnostics_margin_above_null = "final_diagnostics_margin_above_null"
    diagnostic_evolution = "diagnostic_evolution"
    scenario_verdict = "scenario_verdict"


class AggregateKey(str, Enum):
    mean = "mean"
    std = "std"
    individual_seed_values = "individual_seed_values"


class ScenarioVerdictKey(str, Enum):
    memory_benefit_descriptive = "memory_benefit_descriptive"
    hidden_teammate_information_detected = "hidden_teammate_information_detected"
    synchronous_coordination_detected = "synchronous_coordination_detected"
    temporal_coordination_detected = "temporal_coordination_detected"
    notes = "notes"


class RunMetricKey(str, Enum):
    env_name = "env_name"
    config_id = "config_id"
    seed = "seed"
    diagnostics = "diagnostics"
    null_diagnostics = "null_diagnostics"
    above_null = "above_null"
    behavioral_metrics = "behavioral_metrics"
    return_metrics = "return_metrics"
    learning_curve = "learning_curve"


class LearningCurveKey(str, Enum):
    step = "step"
    value = "value"


class PngChunkType(bytes, Enum):
    ihdr = b"IHDR"
    idat = b"IDAT"
    iend = b"IEND"


class ArtifactFileName(str, Enum):
    diagnostics_final = "diagnostics_final.json"
    diagnostics_null_final = "diagnostics_null_final.json"
    behavioral_metrics_final = "behavioral_metrics_final.json"
    run_metadata = "run_metadata.json"


class DirectoryName(str, Enum):
    diagnostics_by_progress = "diagnostics_by_progress"


class RgbColor(tuple[int, int, int], Enum):
    background = (255, 255, 255)
    axis = (31, 41, 55)
    grid = (229, 231, 235)
    ippo_ff = (37, 99, 235)
    ippo_rnn = (20, 184, 166)
    mappo_ff = (234, 88, 12)
    mappo_rnn = (147, 51, 234)
    diagnostic_oar = (37, 99, 235)
    diagnostic_har = (22, 163, 74)
    diagnostic_pif = (234, 88, 12)
    diagnostic_aa = (147, 51, 234)
    diagnostic_dai = (220, 38, 38)
    muted = (156, 163, 175)


@dataclass(frozen=True)
class RunArtifacts:
    env_name: str
    config_id: str
    seed: int
    run_dir: Path
    diagnostics_path: Path
    null_diagnostics_path: Path
    behavioral_metrics_path: Path


@dataclass(frozen=True)
class RunMetrics:
    env_name: str
    config_id: str
    seed: int
    diagnostics: dict[str, float]
    null_diagnostics: dict[str, float]
    above_null: dict[str, bool]
    behavioral_metrics: dict[str, float]
    learning_curve: list[tuple[int, float]]


@dataclass
class Canvas:
    width: int
    height: int
    pixels: bytearray


def build_analysis_outputs(runs_dir: Path, plots_dir: Path) -> dict[str, object]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    run_metrics, missing_artifacts = load_run_metrics(runs_dir)
    memory_gap = compute_memory_gap(run_metrics)
    learning_curves = compute_learning_curves(run_metrics)
    final_above_null = compute_final_diagnostics_above_null(run_metrics)
    final_margin_above_null = compute_final_diagnostics_margin_above_null(run_metrics)
    diagnostic_evolution = load_diagnostic_evolution(runs_dir)
    scenario_verdict = compute_scenario_verdict(final_above_null, memory_gap)

    summary: dict[str, object] = {
        SummaryKey.run_count.value: len(run_metrics),
        SummaryKey.missing_artifacts.value: missing_artifacts,
        SummaryKey.per_run_metrics.value: [
            _run_metrics_to_jsonable(metrics) for metrics in run_metrics
        ],
        SummaryKey.memory_gap.value: memory_gap,
        SummaryKey.learning_curves.value: learning_curves,
        SummaryKey.final_diagnostics_above_null.value: final_above_null,
        SummaryKey.final_diagnostics_margin_above_null.value: final_margin_above_null,
        SummaryKey.diagnostic_evolution.value: diagnostic_evolution,
        SummaryKey.scenario_verdict.value: scenario_verdict,
    }
    _write_json(plots_dir / SUMMARY_FILE_NAME, summary)
    _write_json(
        plots_dir / PER_RUN_METRICS_FILE_NAME,
        summary[SummaryKey.per_run_metrics.value],
    )
    _write_json(plots_dir / SCENARIO_VERDICT_FILE_NAME, scenario_verdict)
    write_required_plots(
        plots_dir=plots_dir,
        run_metrics=run_metrics,
        memory_gap=memory_gap,
        final_margin_above_null=final_margin_above_null,
        diagnostic_evolution=diagnostic_evolution,
    )
    return summary


def load_run_metrics(runs_dir: Path) -> tuple[list[RunMetrics], list[str]]:
    run_metrics: list[RunMetrics] = []
    missing_artifacts: list[str] = []
    for run_artifacts in discover_run_artifacts(runs_dir):
        missing = [
            path
            for path in (
                run_artifacts.diagnostics_path,
                run_artifacts.null_diagnostics_path,
                run_artifacts.behavioral_metrics_path,
            )
            if not path.exists()
        ]
        if missing:
            missing_artifacts.extend(str(path) for path in missing)
            continue
        diagnostics_json = _load_json(run_artifacts.diagnostics_path)
        null_json = _load_json(run_artifacts.null_diagnostics_path)
        behavioral_json = _load_json(run_artifacts.behavioral_metrics_path)
        diagnostics = _diagnostic_values(diagnostics_json)
        null_diagnostics = _null_diagnostic_values(null_json)
        run_metrics.append(
            RunMetrics(
                env_name=run_artifacts.env_name,
                config_id=run_artifacts.config_id,
                seed=run_artifacts.seed,
                diagnostics=diagnostics,
                null_diagnostics=null_diagnostics,
                above_null={
                    metric_name: _above_null(
                        value,
                        null_diagnostics.get(metric_name, math.nan),
                    )
                    for metric_name, value in diagnostics.items()
                },
                behavioral_metrics=_behavioral_values(behavioral_json),
                learning_curve=_load_learning_curve(run_artifacts.run_dir),
            )
        )
    return run_metrics, missing_artifacts


def discover_run_artifacts(runs_dir: Path) -> list[RunArtifacts]:
    artifacts: list[RunArtifacts] = []
    if not runs_dir.exists():
        return artifacts
    for env_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        for config_dir in sorted(path for path in env_dir.iterdir() if path.is_dir()):
            for seed_dir in sorted(path for path in config_dir.iterdir() if path.is_dir()):
                if not seed_dir.name.startswith(SEED_PREFIX):
                    continue
                artifacts.append(
                    RunArtifacts(
                        env_name=env_dir.name,
                        config_id=config_dir.name,
                        seed=_seed_from_directory(seed_dir),
                        run_dir=seed_dir,
                        diagnostics_path=(
                            seed_dir / ArtifactFileName.diagnostics_final.value
                        ),
                        null_diagnostics_path=(
                            seed_dir / ArtifactFileName.diagnostics_null_final.value
                        ),
                        behavioral_metrics_path=(
                            seed_dir / ArtifactFileName.behavioral_metrics_final.value
                        ),
                    )
                )
    return artifacts


def compute_memory_gap(run_metrics: Sequence[RunMetrics]) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for env_name in sorted({metrics.env_name for metrics in run_metrics}):
        for algorithm_label, ff_config, rnn_config in _memory_gap_pairs():
            seed_values: list[float] = []
            for seed in sorted({metrics.seed for metrics in run_metrics}):
                ff_return = _return_for(run_metrics, env_name, ff_config, seed)
                rnn_return = _return_for(run_metrics, env_name, rnn_config, seed)
                if math.isfinite(ff_return) and math.isfinite(rnn_return):
                    seed_values.append(rnn_return - ff_return)
            output[f"{env_name}{CONFIG_SEPARATOR}{algorithm_label}"] = {
                AggregateKey.mean.value: _mean(seed_values),
                AggregateKey.std.value: _std(seed_values),
                AggregateKey.individual_seed_values.value: seed_values,
            }
    return output


def compute_learning_curves(
    run_metrics: Sequence[RunMetrics],
) -> dict[str, list[dict[str, float | int]]]:
    return {
        _run_key(metrics): [
            {
                LearningCurveKey.step.value: step,
                LearningCurveKey.value.value: value,
            }
            for step, value in metrics.learning_curve
        ]
        for metrics in run_metrics
    }


def compute_final_diagnostics_above_null(
    run_metrics: Sequence[RunMetrics],
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for env_name in sorted({metrics.env_name for metrics in run_metrics}):
        for config_id in [config.value for config in ConfigId]:
            matching = [
                metrics
                for metrics in run_metrics
                if metrics.env_name == env_name and metrics.config_id == config_id
            ]
            if not matching:
                continue
            output[f"{env_name}{CONFIG_SEPARATOR}{config_id}"] = {
                _normalized_metric_name(label.value): _count_label(
                    [
                        metrics.above_null.get(_normalized_metric_name(label.value), False)
                        for metrics in matching
                    ],
                    len(matching),
                )
                for label in (
                    DiagnosticLabel.HAR,
                    DiagnosticLabel.PIF,
                    DiagnosticLabel.AA,
                    DiagnosticLabel.DAI,
                )
            }
    return output


def compute_final_diagnostics_margin_above_null(
    run_metrics: Sequence[RunMetrics],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for env_name in sorted({metrics.env_name for metrics in run_metrics}):
        for config_id in [config.value for config in ConfigId]:
            matching = [
                metrics
                for metrics in run_metrics
                if metrics.env_name == env_name and metrics.config_id == config_id
            ]
            if not matching:
                continue
            output[f"{env_name}{CONFIG_SEPARATOR}{config_id}"] = {
                _normalized_metric_name(label.value): _mean(
                    [
                        metrics.diagnostics.get(
                            _normalized_metric_name(label.value),
                            math.nan,
                        )
                        - metrics.null_diagnostics.get(
                            _normalized_metric_name(label.value),
                            math.nan,
                        )
                        for metrics in matching
                    ]
                )
                for label in (
                    DiagnosticLabel.HAR,
                    DiagnosticLabel.PIF,
                    DiagnosticLabel.AA,
                    DiagnosticLabel.DAI,
                )
            }
    return output


def load_diagnostic_evolution(runs_dir: Path) -> dict[str, dict[str, list[float]]]:
    output: dict[str, dict[str, list[float]]] = {}
    for env_name in [env.value for env in EnvName]:
        progress_values: dict[str, list[float]] = {}
        for progress in (25, 50, 75, 100):
            progress_values[str(progress)] = []
        for seed_dir in sorted(
            (runs_dir / env_name / MAPPO_RNN_CONFIG_ID).glob(f"{SEED_PREFIX}*")
        ):
            diagnostics_dir = seed_dir / DirectoryName.diagnostics_by_progress.value
            for progress in (25, 50, 75, 100):
                diagnostics_path = diagnostics_dir / f"diagnostics_{progress}.json"
                if not diagnostics_path.exists():
                    continue
                diagnostics = _diagnostic_values(_load_json(diagnostics_path))
                values = [
                    diagnostics.get(_normalized_metric_name(label.value), math.nan)
                    for label in PlotDiagnosticLabel
                ]
                progress_values[str(progress)].append(_mean_finite(values))
        output[env_name] = progress_values
    return output


def compute_scenario_verdict(
    final_above_null: Mapping[str, Mapping[str, str]],
    memory_gap: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for env_name in [env.value for env in EnvName]:
        ippo_gap = _mapping_float(memory_gap, f"{env_name}{CONFIG_SEPARATOR}{IPPO_ALGORITHM_LABEL}")
        mappo_gap = _mapping_float(memory_gap, f"{env_name}{CONFIG_SEPARATOR}{MAPPO_ALGORITHM_LABEL}")
        mappo_rnn_key = f"{env_name}{CONFIG_SEPARATOR}{ConfigId.mappo_rnn.value}"
        above_null = final_above_null.get(mappo_rnn_key, {})
        output[env_name] = {
            ScenarioVerdictKey.memory_benefit_descriptive.value: (
                ippo_gap > 0.0 or mappo_gap > 0.0
            ),
            ScenarioVerdictKey.hidden_teammate_information_detected.value: (
                above_null.get(_normalized_metric_name(DiagnosticLabel.PIF.value), "0/0")
            ),
            ScenarioVerdictKey.synchronous_coordination_detected.value: (
                above_null.get(_normalized_metric_name(DiagnosticLabel.AA.value), "0/0")
            ),
            ScenarioVerdictKey.temporal_coordination_detected.value: (
                above_null.get(_normalized_metric_name(DiagnosticLabel.HAR.value), "0/0")
            ),
            ScenarioVerdictKey.notes.value: "Descriptive only; reduced seed count.",
        }
    return output


def write_required_plots(
    plots_dir: Path,
    run_metrics: Sequence[RunMetrics],
    memory_gap: Mapping[str, Mapping[str, object]],
    final_margin_above_null: Mapping[str, Mapping[str, float]],
    diagnostic_evolution: Mapping[str, Mapping[str, Sequence[float]]],
) -> None:
    _plot_learning_curves(run_metrics, plots_dir / LEARNING_CURVES_PLOT_NAME)
    _plot_memory_gap(memory_gap, plots_dir / MEMORY_GAP_PLOT_NAME)
    _plot_diagnostic_evolution(
        diagnostic_evolution,
        plots_dir / DIAGNOSTIC_EVOLUTION_PLOT_NAME,
    )
    _plot_above_null(final_margin_above_null, plots_dir / FINAL_DIAGNOSTIC_PLOT_NAME)
    _plot_behavioral(
        run_metrics,
        EnvName.simple_spread.value,
        (
            BehavioralMetricName.coverage_success_rate.value,
            BehavioralMetricName.final_landmark_distance_mean.value,
            BehavioralMetricName.collision_rate.value,
        ),
        plots_dir / BEHAVIORAL_SPREAD_PLOT_NAME,
    )
    _plot_behavioral(
        run_metrics,
        EnvName.simple_speaker_listener.value,
        (
            BehavioralMetricName.target_success_rate.value,
            BehavioralMetricName.final_target_distance_mean.value,
            BehavioralMetricName.wrong_landmark_rate.value,
        ),
        plots_dir / BEHAVIORAL_SPEAKER_LISTENER_PLOT_NAME,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reduced MPE replication tables and plot artifacts."
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_analysis_outputs(
        runs_dir=args.runs_dir,
        plots_dir=args.plots_dir,
    )
    print(
        f"Saved {len(REQUIRED_PLOT_NAMES)} plots for {summary[SummaryKey.run_count.value]} runs"
    )
    return 0


def _plot_learning_curves(run_metrics: Sequence[RunMetrics], output_path: Path) -> None:
    series = [
        metrics.learning_curve
        for metrics in run_metrics
        if metrics.learning_curve
    ]
    colors = [
        _config_color(metrics.config_id)
        for metrics in run_metrics
        if metrics.learning_curve
    ]
    labels = [
        f"{_display_env(metrics.env_name)} {_display_config(metrics.config_id)} seed {metrics.seed}"
        for metrics in run_metrics
        if metrics.learning_curve
    ]
    _write_xy_line_plot(
        series=series,
        colors=colors,
        output_path=output_path,
        title=PlotTitle.final_returns.value,
        x_label=AxisLabel.training_step.value,
        y_label=AxisLabel.return_mean.value,
        series_labels=labels,
    )


def _plot_memory_gap(
    memory_gap: Mapping[str, Mapping[str, object]],
    output_path: Path,
) -> None:
    sorted_keys = sorted(memory_gap)
    values = [
        float(values_by_key.get(AggregateKey.mean.value, math.nan))
        for _, values_by_key in sorted(memory_gap.items())
    ]
    colors = [
        RgbColor.ippo_rnn.value if IPPO_ALGORITHM_LABEL in key else RgbColor.mappo_rnn.value
        for key in sorted_keys
    ]
    _write_bar_plot(
        values=values,
        colors=colors,
        output_path=output_path,
        title=PlotTitle.memory_gap.value,
        x_label=AxisLabel.algorithm_pair.value,
        y_label=AxisLabel.return_delta.value,
        x_tick_labels=[_display_summary_key(key) for key in sorted_keys],
        legend_items=(
            (IPPO_ALGORITHM_LABEL, RgbColor.ippo_rnn.value),
            (MAPPO_ALGORITHM_LABEL, RgbColor.mappo_rnn.value),
        ),
    )


def _plot_diagnostic_evolution(
    diagnostic_evolution: Mapping[str, Mapping[str, Sequence[float]]],
    output_path: Path,
) -> None:
    series: list[list[float]] = []
    colors: list[tuple[int, int, int]] = []
    series_labels: list[str] = []
    for env_name in [env.value for env in EnvName]:
        progress_values = diagnostic_evolution.get(env_name, {})
        values = [
            _mean_finite(progress_values.get(str(progress), []))
            for progress in (25, 50, 75, 100)
        ]
        series.append(values)
        series_labels.append(_display_env(env_name))
        colors.append(
            RgbColor.mappo_rnn.value
            if env_name == EnvName.simple_spread.value
            else RgbColor.diagnostic_dai.value
        )
    _write_line_plot(
        series=series,
        colors=colors,
        output_path=output_path,
        title=PlotTitle.diagnostic_evolution.value,
        x_label=AxisLabel.progress.value,
        y_label=AxisLabel.diagnostic_value.value,
        x_tick_labels=("25%", "50%", "75%", "100%"),
        series_labels=series_labels,
    )


def _plot_above_null(
    final_margin_above_null: Mapping[str, Mapping[str, float]],
    output_path: Path,
) -> None:
    values: list[float] = []
    colors: list[tuple[int, int, int]] = []
    labels: list[str] = []
    for row_key in sorted(final_margin_above_null):
        row = final_margin_above_null[row_key]
        for diagnostic_name in (
            _normalized_metric_name(DiagnosticLabel.HAR.value),
            _normalized_metric_name(DiagnosticLabel.PIF.value),
            _normalized_metric_name(DiagnosticLabel.AA.value),
            _normalized_metric_name(DiagnosticLabel.DAI.value),
        ):
            values.append(row.get(diagnostic_name, math.nan))
            colors.append(_diagnostic_color(diagnostic_name))
            labels.append(f"{_display_summary_key(row_key)}\n{diagnostic_name}")
    _write_bar_plot(
        values=values,
        colors=colors,
        output_path=output_path,
        title=PlotTitle.above_null.value,
        x_label=AxisLabel.run_config.value,
        y_label=AxisLabel.diagnostic_margin.value,
        x_tick_labels=labels,
        legend_items=_diagnostic_legend_items(),
    )


def _plot_behavioral(
    run_metrics: Sequence[RunMetrics],
    env_name: str,
    metric_names: Sequence[str],
    output_path: Path,
) -> None:
    labels = [
        f"{_display_behavioral_metric(metric_name)}\n{_display_config(config_id)}"
        for metric_name in metric_names
        for config_id in [config.value for config in ConfigId]
    ]
    values = [
        _mean(
            [
                metrics.behavioral_metrics.get(metric_name, math.nan)
                for metrics in run_metrics
                if metrics.env_name == env_name and metrics.config_id == config_id
            ]
        )
        for metric_name in metric_names
        for config_id in [config.value for config in ConfigId]
    ]
    title = (
        PlotTitle.behavioral_spread.value
        if env_name == EnvName.simple_spread.value
        else PlotTitle.behavioral_speaker_listener.value
    )
    _write_bar_plot(
        values=values,
        colors=_config_colors_repeated(),
        output_path=output_path,
        title=title,
        x_label=AxisLabel.run_config.value,
        y_label=AxisLabel.behavioral_value.value,
        x_tick_labels=labels,
        legend_items=_config_legend_items(),
    )


def _write_bar_plot(
    values: Sequence[float],
    colors: Sequence[tuple[int, int, int]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    x_tick_labels: Sequence[str],
    legend_items: Sequence[tuple[str, tuple[int, int, int]]] = (),
) -> None:
    finite_values = [value for value in values if math.isfinite(value)]
    figure, axes = _new_figure(title, x_label, y_label)
    x_values = list(range(len(values)))
    plot_values = [value if math.isfinite(value) else 0.0 for value in values]
    bar_colors = [
        _matplotlib_color(colors[index % len(colors)] if colors else RgbColor.muted.value)
        for index in x_values
    ]
    bars = axes.bar(x_values, plot_values, color=bar_colors)
    for bar_index, value in enumerate(values):
        if not math.isfinite(value):
            bars[bar_index].set_alpha(NAN_BAR_ALPHA)
            continue
        offset = _annotation_offset(finite_values)
        vertical_alignment = "bottom" if value >= 0.0 else "top"
        axes.text(
            bar_index,
            value + offset if value >= 0.0 else value - offset,
            _format_number(value),
            ha="center",
            va=vertical_alignment,
            fontsize=ANNOTATION_FONT_SIZE,
            color=_matplotlib_color(RgbColor.axis.value),
        )
    _style_axis(
        axes=axes,
        x_tick_labels=x_tick_labels,
        legend_items=legend_items,
    )
    _set_y_limits(axes, finite_values)
    _save_figure(figure, output_path)


def _write_line_plot(
    series: Sequence[Sequence[float]],
    colors: Sequence[tuple[int, int, int]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    x_tick_labels: Sequence[str],
    series_labels: Sequence[str],
) -> None:
    figure, axes = _new_figure(title, x_label, y_label)
    all_values = [value for values in series for value in values if math.isfinite(value)]
    for series_index, values in enumerate(series):
        x_values: list[int] = []
        y_values: list[float] = []
        for value_index, value in enumerate(values):
            if not math.isfinite(value):
                continue
            x_values.append(value_index)
            y_values.append(value)
        color = colors[series_index % len(colors)] if colors else RgbColor.muted.value
        label = (
            series_labels[series_index]
            if series_index < len(series_labels)
            else f"Series {series_index + 1}"
        )
        axes.plot(
            x_values,
            y_values,
            color=_matplotlib_color(color),
            marker="o",
            linewidth=LINE_WIDTH,
            label=label,
        )
    _style_axis(
        axes=axes,
        x_tick_labels=x_tick_labels,
        legend_items=(),
    )
    if series_labels:
        axes.legend(loc="best", fontsize=TICK_LABEL_FONT_SIZE)
    _set_y_limits(axes, all_values)
    _save_figure(figure, output_path)


def _write_xy_line_plot(
    series: Sequence[Sequence[tuple[int, float]]],
    colors: Sequence[tuple[int, int, int]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series_labels: Sequence[str],
) -> None:
    figure, axes = _new_figure(title, x_label, y_label)
    all_values = [value for values in series for _, value in values if math.isfinite(value)]
    for series_index, values in enumerate(series):
        x_values = [step for step, value in values if math.isfinite(value)]
        y_values = [value for _, value in values if math.isfinite(value)]
        color = colors[series_index % len(colors)] if colors else RgbColor.muted.value
        label = (
            series_labels[series_index]
            if series_index < len(series_labels)
            else f"Series {series_index + 1}"
        )
        axes.plot(
            x_values,
            y_values,
            color=_matplotlib_color(color),
            marker="o",
            markersize=3,
            linewidth=LINE_WIDTH,
            label=label,
        )
    axes.tick_params(axis="x", labelsize=TICK_LABEL_FONT_SIZE)
    axes.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    axes.legend(loc="best", fontsize=TICK_LABEL_FONT_SIZE)
    _set_y_limits(axes, all_values)
    _save_figure(figure, output_path)


def _new_figure(title: str, x_label: str, y_label: str) -> tuple[Figure, Axes]:
    figure, axes = plt.subplots(
        figsize=(PLOT_FIGURE_WIDTH, PLOT_FIGURE_HEIGHT),
        dpi=PLOT_DPI,
    )
    axes.set_title(title, fontsize=TITLE_FONT_SIZE)
    axes.set_xlabel(x_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes.set_ylabel(y_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes.grid(axis="y", alpha=GRID_ALPHA)
    return figure, axes


def _style_axis(
    axes: Axes,
    x_tick_labels: Sequence[str],
    legend_items: Sequence[tuple[str, tuple[int, int, int]]],
) -> None:
    axes.set_xticks(list(range(len(x_tick_labels))))
    axes.set_xticklabels(
        list(x_tick_labels),
        rotation=TICK_LABEL_ROTATION_DEGREES,
        ha="right",
        fontsize=TICK_LABEL_FONT_SIZE,
    )
    axes.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    if legend_items:
        handles = [
            Patch(facecolor=_matplotlib_color(color), label=label)
            for label, color in legend_items
        ]
        axes.legend(handles=handles, loc="best", fontsize=TICK_LABEL_FONT_SIZE)


def _set_y_limits(axes: Axes, values: Sequence[float]) -> None:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        axes.set_ylim(0.0, DEFAULT_EMPTY_MAX)
        return
    minimum_value = min([0.0] + finite_values)
    maximum_value = max([0.0] + finite_values)
    value_range = max(maximum_value - minimum_value, MINIMUM_PLOT_RANGE)
    padding = value_range * BAR_VALUE_PADDING_RATIO
    axes.set_ylim(minimum_value - padding, maximum_value + padding)


def _annotation_offset(values: Sequence[float]) -> float:
    if not values:
        return MINIMUM_PLOT_RANGE * BAR_VALUE_PADDING_RATIO
    value_range = max(max(values) - min(values), MINIMUM_PLOT_RANGE)
    return value_range * BAR_VALUE_PADDING_RATIO


def _save_figure(figure: Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, format="png")
    plt.close(figure)


def _matplotlib_color(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def _new_canvas() -> Canvas:
    pixels = bytearray(RgbColor.background.value * (PLOT_WIDTH * PLOT_HEIGHT))
    return Canvas(width=PLOT_WIDTH, height=PLOT_HEIGHT, pixels=pixels)


def _draw_axes(canvas: Canvas) -> None:
    axis_color = RgbColor.axis.value
    grid_color = RgbColor.grid.value
    left = PLOT_MARGIN_LEFT
    right = canvas.width - PLOT_MARGIN_RIGHT
    top = PLOT_MARGIN_TOP
    bottom = canvas.height - PLOT_MARGIN_BOTTOM
    _fill_rect(canvas, left, top, right, top + PLOT_AXIS_WIDTH, grid_color)
    _fill_rect(canvas, left, bottom, right, bottom + PLOT_AXIS_WIDTH, axis_color)
    _fill_rect(canvas, left, top, left + PLOT_AXIS_WIDTH, bottom, axis_color)


def _fill_rect(
    canvas: Canvas,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int],
) -> None:
    clipped_left = max(0, min(canvas.width, left))
    clipped_right = max(0, min(canvas.width, right))
    clipped_top = max(0, min(canvas.height, top))
    clipped_bottom = max(0, min(canvas.height, bottom))
    for y in range(clipped_top, clipped_bottom):
        for x in range(clipped_left, clipped_right):
            _set_pixel(canvas, x, y, color)


def _draw_line(
    canvas: Canvas,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _fill_rect(
            canvas,
            x0 - LINE_WIDTH,
            y0 - LINE_WIDTH,
            x0 + LINE_WIDTH,
            y0 + LINE_WIDTH,
            color,
        )
        if x0 == x1 and y0 == y1:
            break
        doubled_error = error * 2
        if doubled_error >= dy:
            error += dy
            x0 += step_x
        if doubled_error <= dx:
            error += dx
            y0 += step_y


def _set_pixel(
    canvas: Canvas,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    index = (y * canvas.width + x) * PNG_CHANNELS
    canvas.pixels[index : index + PNG_CHANNELS] = bytes(color)


def _write_png(output_path: Path, canvas: Canvas) -> None:
    rows = []
    row_bytes = canvas.width * PNG_CHANNELS
    for y in range(canvas.height):
        start = y * row_bytes
        rows.append(bytes([PNG_NO_FILTER]) + canvas.pixels[start : start + row_bytes])
    compressed = zlib.compress(b"".join(rows))
    ihdr = struct.pack(
        ">IIBBBBB",
        canvas.width,
        canvas.height,
        PNG_BIT_DEPTH,
        PNG_COLOR_MODE_RGB,
        PNG_COMPRESSION_METHOD,
        PNG_FILTER_METHOD,
        PNG_INTERLACE_METHOD,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        PNG_SIGNATURE
        + _png_chunk(PngChunkType.ihdr.value, ihdr)
        + _png_chunk(PngChunkType.idat.value, compressed)
        + _png_chunk(PngChunkType.iend.value, b"")
    )


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(chunk_type + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", crc)
    )


def _value_to_y(
    value: float,
    minimum_value: float,
    value_range: float,
    plot_height: int,
) -> int:
    ratio = (value - minimum_value) / value_range
    return int(PLOT_MARGIN_TOP + plot_height * (1.0 - ratio))


def _diagnostic_values(source: Mapping[str, object]) -> dict[str, float]:
    diagnostics = _nested_mapping(source, JsonKey.diagnostics)
    return {
        _normalized_metric_name(label.value): _numeric(
            _nested_mapping(diagnostics, label.value).get(JsonKey.normalized.value)
        )
        for label in DiagnosticLabel
    }


def _null_diagnostic_values(source: Mapping[str, object]) -> dict[str, float]:
    null_diagnostics = _nested_mapping(source, JsonKey.null_diagnostics)
    return {
        _normalized_metric_name(label.value): _numeric(
            _nested_mapping(null_diagnostics, label.value).get(
                JsonKey.normalized_mean.value
            )
        )
        for label in DiagnosticLabel
    }


def _behavioral_values(source: Mapping[str, object]) -> dict[str, float]:
    values = _nested_mapping(source, JsonKey.behavioral_metrics)
    return {str(key): _numeric(item) for key, item in values.items()}


def _nested_mapping(
    source: Mapping[str, object],
    key: JsonKey | str,
) -> Mapping[str, object]:
    key_value = key.value if isinstance(key, JsonKey) else key
    value = source.get(key_value, {})
    return value if isinstance(value, Mapping) else {}


def _load_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"JSON artifact must contain a mapping: {path}")
    return {str(key): value for key, value in loaded.items()}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=JSON_INDENT, allow_nan=True),
        encoding="utf-8",
    )


def _run_metrics_to_jsonable(metrics: RunMetrics) -> dict[str, object]:
    return {
        RunMetricKey.env_name.value: metrics.env_name,
        RunMetricKey.config_id.value: metrics.config_id,
        RunMetricKey.seed.value: metrics.seed,
        RunMetricKey.diagnostics.value: metrics.diagnostics,
        RunMetricKey.null_diagnostics.value: metrics.null_diagnostics,
        RunMetricKey.above_null.value: metrics.above_null,
        RunMetricKey.behavioral_metrics.value: metrics.behavioral_metrics,
        RunMetricKey.return_metrics.value: {
            BehavioralMetricName.return_mean.value: metrics.behavioral_metrics.get(
                BehavioralMetricName.return_mean.value,
                math.nan,
            )
        },
        RunMetricKey.learning_curve.value: [
            {
                LearningCurveKey.step.value: step,
                LearningCurveKey.value.value: value,
            }
            for step, value in metrics.learning_curve
        ],
    }


def _memory_gap_pairs() -> tuple[tuple[str, str, str], ...]:
    return (
        (
            IPPO_ALGORITHM_LABEL,
            ConfigId.ippo_ff.value,
            ConfigId.ippo_rnn.value,
        ),
        (
            MAPPO_ALGORITHM_LABEL,
            ConfigId.mappo_ff.value,
            ConfigId.mappo_rnn.value,
        ),
    )


def _return_for(
    run_metrics: Sequence[RunMetrics],
    env_name: str,
    config_id: str,
    seed: int,
) -> float:
    for metrics in run_metrics:
        if (
            metrics.env_name == env_name
            and metrics.config_id == config_id
            and metrics.seed == seed
        ):
            return metrics.behavioral_metrics.get(
                BehavioralMetricName.return_mean.value,
                math.nan,
            )
    return math.nan


def _config_colors_repeated() -> list[tuple[int, int, int]]:
    return [
        RgbColor.ippo_ff.value,
        RgbColor.ippo_rnn.value,
        RgbColor.mappo_ff.value,
        RgbColor.mappo_rnn.value,
    ]


def _config_color(config_id: str) -> tuple[int, int, int]:
    if config_id == ConfigId.ippo_ff.value:
        return RgbColor.ippo_ff.value
    if config_id == ConfigId.ippo_rnn.value:
        return RgbColor.ippo_rnn.value
    if config_id == ConfigId.mappo_ff.value:
        return RgbColor.mappo_ff.value
    if config_id == ConfigId.mappo_rnn.value:
        return RgbColor.mappo_rnn.value
    return RgbColor.muted.value


def _diagnostic_color(name: str) -> tuple[int, int, int]:
    if name.startswith(DiagnosticLabel.HAR.value):
        return RgbColor.diagnostic_har.value
    if name.startswith(DiagnosticLabel.PIF.value):
        return RgbColor.diagnostic_pif.value
    if name.startswith(DiagnosticLabel.AA.value):
        return RgbColor.diagnostic_aa.value
    if name.startswith(DiagnosticLabel.DAI.value):
        return RgbColor.diagnostic_dai.value
    return RgbColor.diagnostic_oar.value


def _config_legend_items() -> tuple[tuple[str, tuple[int, int, int]], ...]:
    return (
        (DisplayLabel.ippo_ff.value, RgbColor.ippo_ff.value),
        (DisplayLabel.ippo_rnn.value, RgbColor.ippo_rnn.value),
        (DisplayLabel.mappo_ff.value, RgbColor.mappo_ff.value),
        (DisplayLabel.mappo_rnn.value, RgbColor.mappo_rnn.value),
    )


def _diagnostic_legend_items() -> tuple[tuple[str, tuple[int, int, int]], ...]:
    return (
        (DiagnosticLabel.HAR.value, RgbColor.diagnostic_har.value),
        (DiagnosticLabel.PIF.value, RgbColor.diagnostic_pif.value),
        (DiagnosticLabel.AA.value, RgbColor.diagnostic_aa.value),
        (DiagnosticLabel.DAI.value, RgbColor.diagnostic_dai.value),
    )


def _display_env(env_name: str) -> str:
    if env_name == EnvName.simple_spread.value:
        return DisplayLabel.simple_spread.value
    if env_name == EnvName.simple_speaker_listener.value:
        return DisplayLabel.simple_speaker_listener.value
    return env_name


def _display_config(config_id: str) -> str:
    if config_id == ConfigId.ippo_ff.value:
        return DisplayLabel.ippo_ff.value
    if config_id == ConfigId.ippo_rnn.value:
        return DisplayLabel.ippo_rnn.value
    if config_id == ConfigId.mappo_ff.value:
        return DisplayLabel.mappo_ff.value
    if config_id == ConfigId.mappo_rnn.value:
        return DisplayLabel.mappo_rnn.value
    return config_id


def _display_behavioral_metric(metric_name: str) -> str:
    if metric_name == BehavioralMetricName.coverage_success_rate.value:
        return DisplayLabel.coverage_success_rate.value
    if metric_name == BehavioralMetricName.final_landmark_distance_mean.value:
        return DisplayLabel.final_landmark_distance_mean.value
    if metric_name == BehavioralMetricName.collision_rate.value:
        return DisplayLabel.collision_rate.value
    if metric_name == BehavioralMetricName.target_success_rate.value:
        return DisplayLabel.target_success_rate.value
    if metric_name == BehavioralMetricName.final_target_distance_mean.value:
        return DisplayLabel.final_target_distance_mean.value
    if metric_name == BehavioralMetricName.wrong_landmark_rate.value:
        return DisplayLabel.wrong_landmark_rate.value
    return metric_name


def _display_summary_key(summary_key: str) -> str:
    parts = summary_key.split(CONFIG_SEPARATOR)
    if len(parts) != 2:
        return summary_key
    return f"{_display_env(parts[0])}\n{_display_config(parts[1])}"


def _run_key(metrics: RunMetrics) -> str:
    return (
        f"{metrics.env_name}{CONFIG_SEPARATOR}{metrics.config_id}"
        f"{CONFIG_SEPARATOR}{SEED_PREFIX}{metrics.seed}"
    )


def _load_learning_curve(run_dir: Path) -> list[tuple[int, float]]:
    scalar_path = _learning_curve_path(run_dir)
    if scalar_path is None:
        return []
    frame_multiplier = _learning_curve_frame_multiplier(run_dir)
    points: list[tuple[int, float]] = []
    with scalar_path.open(newline="", encoding="utf-8") as scalar_file:
        for row in csv.reader(scalar_file):
            if len(row) < 2:
                continue
            step = _numeric(row[0])
            value = _numeric(row[1])
            if math.isfinite(step) and math.isfinite(value):
                points.append((int(step * frame_multiplier), value))
    return points


def _learning_curve_frame_multiplier(run_dir: Path) -> int:
    metadata_path = run_dir / ArtifactFileName.run_metadata.value
    if not metadata_path.exists():
        return 1
    metadata = _load_json(metadata_path)
    commands = metadata.get(JsonKey.commands.value)
    if not isinstance(commands, Mapping):
        return 1
    training_command = commands.get(JsonKey.training.value)
    if not isinstance(training_command, Sequence) or isinstance(training_command, str):
        return 1
    for item in training_command:
        if not isinstance(item, str):
            continue
        if item.startswith(TrainingOverride.frames_per_batch.value):
            value = item[len(TrainingOverride.frames_per_batch.value) :]
            if value.isdigit():
                return int(value)
    return 1


def _learning_curve_path(run_dir: Path) -> Path | None:
    metadata_path = run_dir / ArtifactFileName.run_metadata.value
    if metadata_path.exists():
        metadata = _load_json(metadata_path)
        checkpoint = _nested_mapping(metadata, JsonKey.final_checkpoint)
        source_path = checkpoint.get(JsonKey.source_path.value)
        if isinstance(source_path, str):
            parts = Path(source_path).parts
            if len(parts) >= 3:
                experiment_name = parts[-3]
                candidate = (
                    run_dir
                    / experiment_name
                    / experiment_name
                    / SCALARS_DIRECTORY_NAME
                    / EVAL_RETURN_SCALAR_FILE_NAME
                )
                if candidate.exists():
                    return candidate
    candidates = sorted(
        run_dir.glob(
            f"*/*/{SCALARS_DIRECTORY_NAME}/{EVAL_RETURN_SCALAR_FILE_NAME}"
        )
    )
    return candidates[-1] if candidates else None


def _mapping_float(
    source: Mapping[str, Mapping[str, object]],
    key: str,
) -> float:
    value = source.get(key, {}).get(AggregateKey.mean.value, math.nan)
    return _numeric(value)


def _normalized_metric_name(label: str) -> str:
    return f"{label}norm"


def _seed_from_directory(seed_dir: Path) -> int:
    return int(seed_dir.name.removeprefix(SEED_PREFIX))


def _count_label(values: Sequence[bool], denominator: int) -> str:
    return f"{sum(1 for value in values if value)}/{denominator}"


def _fraction_from_count(value: str) -> float:
    parts = value.split(CONFIG_SEPARATOR)
    if len(parts) != 2:
        return 0.0
    numerator = _numeric(parts[0])
    denominator = _numeric(parts[1])
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _above_null(value: float, null_value: float) -> bool:
    return math.isfinite(value) and math.isfinite(null_value) and value > null_value


def _mean(values: Sequence[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return math.nan
    return float(sum(finite_values) / len(finite_values))


def _mean_finite(values: Sequence[float]) -> float:
    return _mean(values)


def _std(values: Sequence[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return math.nan
    mean_value = _mean(finite_values)
    variance = sum((value - mean_value) ** 2 for value in finite_values) / len(
        finite_values
    )
    return float(math.sqrt(variance))


def _numeric(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return math.nan
    return math.nan


def _format_number(value: float) -> str:
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
