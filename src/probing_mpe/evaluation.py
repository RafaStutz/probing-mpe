from __future__ import annotations

import argparse
import importlib
import json
import math
import pickle
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from probing_mpe.trajectories import TrajectoryKey, validate_trajectory_schema


DEFAULT_HISTORY_K = 3
DEFAULT_CMI_K = 25
DEFAULT_NULL_REPS = 5
DEFAULT_MAX_SAMPLES = 8000
DEFAULT_POSTERIOR_ALPHA = 0.5
DEFAULT_MIN_EFFECT = 0.01
DEFAULT_PARALLEL_WORKERS = 1
JSON_INDENT = 2


class BackendModuleName(str, Enum):
    dec_pomdp_diagnostics = "dec_pomdp_diagnostics"
    wandb = "wandb"


class DiagnosticName(str, Enum):
    oar = "oar"
    har = "har"
    pif = "pif"
    aa = "aa"
    dai = "dai"


class DiagnosticLabel(str, Enum):
    OAR = "OAR"
    HAR = "HAR"
    PIF = "PIF"
    AA = "AA"
    DAI = "DAI"


class DiagnosticJsonKey(str, Enum):
    metadata = "metadata"
    diagnostics = "diagnostics"
    null_diagnostics = "null_diagnostics"
    flags = "flags"
    raw_row = "raw_row"
    undefined_normalized = "undefined_normalized"
    null_reps = "null_reps"
    replicate_values_available = "replicate_values_available"
    replicate_values_reason = "replicate_values_reason"
    raw = "raw"
    normalized = "normalized"
    raw_mean = "raw_mean"
    normalized_mean = "normalized_mean"
    raw_column = "raw_column"
    normalized_column = "normalized_column"
    raw_null_column = "raw_null_column"
    normalized_null_column = "normalized_null_column"
    metric = "metric"
    undefined_reason = "undefined_reason"


class MetadataKey(str, Enum):
    env_name = "env_name"
    algorithm = "algorithm"
    policy_architecture = "policy_architecture"
    config_id = "config_id"
    seed = "seed"
    parameter_sharing = "parameter_sharing"
    training_progress_percent = "training_progress_percent"
    diagnostics_library = "diagnostics_library"


class ResultAttribute(str, Enum):
    flags = "flags"
    raw_row = "raw_row"


class BackendFunctionName(str, Enum):
    UserData = "UserData"
    compute_diagnostics = "compute_diagnostics"
    init = "init"
    log = "log"
    finish = "finish"


class WandbMetricPrefix(str, Enum):
    diagnostics_final = "diagnostics_final"


class UndefinedReason(str, Enum):
    zero_conditional_action_entropy = "zero or degenerate action entropy"


class NullReplicateReason(str, Enum):
    backend_means_only = (
        "dec_pomdp_diagnostics.UserData API exposes permutation-null means, "
        "not individual null replicate values"
    )


class MetricColumn(str, Enum):
    oar_raw = "oar_max"
    oar_norm = "oarR_max"
    har_ohist_raw = "har_ohist_max"
    har_ohist_norm = "harRcond_ohist_max"
    har_hidden_raw = "har_hidden_max"
    har_hidden_norm = "harRcond_hidden_max"
    pif_ohist_raw = "pifOA_ohist_max"
    pif_ohist_norm = "pifOARcond_ohist_max"
    pif_hidden_raw = "pif_hidden_max"
    pif_hidden_norm = "pifRcond_hidden_max"
    aa_raw = "aa_max"
    aa_norm = "aaRcond_max"
    dai_ohist_raw = "daiOA_ohist_max"
    dai_ohist_norm = "daiOARcond_ohist_max"
    dai_hidden_raw = "dai_hidden_max"
    dai_hidden_norm = "daiRcond_hidden_max"


@dataclass(frozen=True)
class MetricColumns:
    raw: MetricColumn
    normalized: MetricColumn


def load_trajectory(path: Path) -> dict[str, object]:
    with path.open("rb") as trajectory_file:
        loaded = pickle.load(trajectory_file)
    if not isinstance(loaded, Mapping):
        raise ValueError("Trajectory file must contain a mapping")
    trajectory = dict(loaded)
    validate_trajectory_schema(trajectory)
    return trajectory


def trajectory_to_user_data(
    trajectory: Mapping[str, object], diagnostics_module: object
) -> object:
    validate_trajectory_schema(trajectory)
    user_data_factory = getattr(diagnostics_module, BackendFunctionName.UserData.value)
    if not callable(user_data_factory):
        raise ValueError("Diagnostic backend does not expose UserData")

    hidden_states = trajectory[TrajectoryKey.hidden_states.value]
    return user_data_factory(
        observations=_agent_arrays(trajectory, TrajectoryKey.observations),
        actions=_agent_arrays(trajectory, TrajectoryKey.actions_diagnostic),
        timesteps=_agent_arrays(trajectory, TrajectoryKey.timesteps),
        episode_ids=_agent_arrays(trajectory, TrajectoryKey.episode_ids),
        hidden_states=(
            _mapping_to_arrays(hidden_states)
            if isinstance(hidden_states, Mapping)
            else None
        ),
        env_name=str(trajectory[TrajectoryKey.env_name.value]),
        alg_name=str(trajectory[TrajectoryKey.config_id.value]),
        seed=int(trajectory[TrajectoryKey.seed.value]),
        scenario_name=str(trajectory[TrajectoryKey.env_name.value]),
    )


def compute_diagnostics_for_trajectory(
    trajectory: Mapping[str, object],
    diagnostics_module: object,
    history_k: int,
    cmi_k: int,
    null_reps: int,
    max_samples: int | None,
    posterior_alpha: float,
    metrics: Sequence[str],
    min_effect: float,
    parallel_workers: int,
    force_continuous_actions: bool | None,
) -> tuple[dict[str, object], dict[str, object]]:
    user_data = trajectory_to_user_data(trajectory, diagnostics_module)
    compute_diagnostics = getattr(
        diagnostics_module, BackendFunctionName.compute_diagnostics.value
    )
    if not callable(compute_diagnostics):
        raise ValueError("Diagnostic backend does not expose compute_diagnostics")

    result = compute_diagnostics(
        user_data,
        history_k=history_k,
        cmi_k=cmi_k,
        null_reps=null_reps,
        metrics=tuple(metrics),
        force_continuous_A=force_continuous_actions,
        max_samples=max_samples,
        parallel_workers=parallel_workers,
        posterior_alpha=posterior_alpha,
        min_effect=min_effect,
    )
    raw_row = _result_mapping(result, ResultAttribute.raw_row)
    flags = _result_mapping(result, ResultAttribute.flags)
    metadata = _metadata(trajectory)
    diagnostics = _diagnostic_values(trajectory, raw_row)
    null_diagnostics = _null_diagnostic_values(trajectory, raw_row)

    return (
        {
            DiagnosticJsonKey.metadata.value: metadata,
            DiagnosticJsonKey.diagnostics.value: diagnostics,
            DiagnosticJsonKey.flags.value: flags,
            DiagnosticJsonKey.undefined_normalized.value: _undefined_normalized(
                diagnostics
            ),
            DiagnosticJsonKey.raw_row.value: raw_row,
        },
        {
            DiagnosticJsonKey.metadata.value: metadata,
            DiagnosticJsonKey.null_reps.value: null_reps,
            DiagnosticJsonKey.null_diagnostics.value: null_diagnostics,
            DiagnosticJsonKey.replicate_values_available.value: False,
            DiagnosticJsonKey.replicate_values_reason.value: (
                NullReplicateReason.backend_means_only.value
            ),
        },
    )


def write_diagnostic_outputs(
    diagnostics: Mapping[str, object],
    null_diagnostics: Mapping[str, object],
    diagnostics_path: Path,
    null_diagnostics_path: Path,
) -> None:
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    null_diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(_to_jsonable(diagnostics), indent=JSON_INDENT, allow_nan=True),
        encoding="utf-8",
    )
    null_diagnostics_path.write_text(
        json.dumps(_to_jsonable(null_diagnostics), indent=JSON_INDENT, allow_nan=True),
        encoding="utf-8",
    )


def flatten_wandb_metrics(
    diagnostics: Mapping[str, object], null_diagnostics: Mapping[str, object]
) -> dict[str, object]:
    metric_values = _nested_mapping(diagnostics, DiagnosticJsonKey.diagnostics)
    null_metric_values = _nested_mapping(
        null_diagnostics, DiagnosticJsonKey.null_diagnostics
    )
    output: dict[str, object] = {}
    for diagnostic_label in DiagnosticLabel:
        values = _nested_mapping(metric_values, diagnostic_label.value)
        null_values = _nested_mapping(null_metric_values, diagnostic_label.value)
        raw_key = _wandb_metric_name(diagnostic_label.value)
        normalized_key = _wandb_metric_name(f"{diagnostic_label.value}norm")
        null_key = _wandb_metric_name(f"{diagnostic_label.value}norm_null_mean")
        above_null_key = _wandb_metric_name(f"{diagnostic_label.value}norm_above_null")
        output[raw_key] = values.get(DiagnosticJsonKey.raw.value)
        output[normalized_key] = values.get(DiagnosticJsonKey.normalized.value)
        output[null_key] = null_values.get(DiagnosticJsonKey.normalized_mean.value)
        output[above_null_key] = _above_null(
            values.get(DiagnosticJsonKey.normalized.value),
            null_values.get(DiagnosticJsonKey.normalized_mean.value),
        )
    return output


def log_diagnostics_to_wandb(
    diagnostics: Mapping[str, object],
    null_diagnostics: Mapping[str, object],
    enabled: bool,
    project: str | None,
    run_name: str | None,
    mode: str | None,
) -> None:
    if not enabled:
        return
    wandb_module = importlib.import_module(BackendModuleName.wandb.value)
    created_run = False
    if getattr(wandb_module, "run", None) is None:
        init = getattr(wandb_module, BackendFunctionName.init.value)
        if not callable(init):
            raise ValueError("wandb backend does not expose init")
        init(project=project, name=run_name, mode=mode)
        created_run = True

    log = getattr(wandb_module, BackendFunctionName.log.value)
    if not callable(log):
        raise ValueError("wandb backend does not expose log")
    log(flatten_wandb_metrics(diagnostics, null_diagnostics))

    if created_run:
        finish = getattr(wandb_module, BackendFunctionName.finish.value)
        if callable(finish):
            finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Dec-POMDP diagnostics from a BenchMARL trajectory export."
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    parser.add_argument("--null-output", type=Path, required=True)
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
        default=[metric.value for metric in DiagnosticName],
    )
    parser.add_argument("--force-continuous-actions", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics_module = importlib.import_module(
        BackendModuleName.dec_pomdp_diagnostics.value
    )
    trajectory = load_trajectory(args.trajectory)
    diagnostics, null_diagnostics = compute_diagnostics_for_trajectory(
        trajectory=trajectory,
        diagnostics_module=diagnostics_module,
        history_k=args.history_k,
        cmi_k=args.cmi_k,
        null_reps=args.null_reps,
        max_samples=args.max_samples,
        posterior_alpha=args.posterior_alpha,
        metrics=tuple(args.metrics),
        min_effect=args.min_effect,
        parallel_workers=args.parallel_workers,
        force_continuous_actions=(
            True if args.force_continuous_actions else None
        ),
    )
    write_diagnostic_outputs(
        diagnostics=diagnostics,
        null_diagnostics=null_diagnostics,
        diagnostics_path=args.diagnostics_output,
        null_diagnostics_path=args.null_output,
    )
    log_diagnostics_to_wandb(
        diagnostics=diagnostics,
        null_diagnostics=null_diagnostics,
        enabled=bool(args.wandb),
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        mode=args.wandb_mode,
    )
    print(f"Saved {args.diagnostics_output} and {args.null_output}")
    return 0


def _metadata(trajectory: Mapping[str, object]) -> dict[str, object]:
    return {
        MetadataKey.env_name.value: trajectory[TrajectoryKey.env_name.value],
        MetadataKey.algorithm.value: trajectory[TrajectoryKey.algorithm.value],
        MetadataKey.policy_architecture.value: trajectory[
            TrajectoryKey.policy_architecture.value
        ],
        MetadataKey.config_id.value: trajectory[TrajectoryKey.config_id.value],
        MetadataKey.seed.value: trajectory[TrajectoryKey.seed.value],
        MetadataKey.parameter_sharing.value: trajectory[
            TrajectoryKey.parameter_sharing.value
        ],
        MetadataKey.training_progress_percent.value: trajectory[
            TrajectoryKey.training_progress_percent.value
        ],
        MetadataKey.diagnostics_library.value: BackendModuleName.dec_pomdp_diagnostics.value,
    }


def _diagnostic_values(
    trajectory: Mapping[str, object], raw_row: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for label, columns in _metric_columns(trajectory).items():
        output[label.value] = {
            DiagnosticJsonKey.raw.value: _numeric_value(raw_row, columns.raw.value),
            DiagnosticJsonKey.normalized.value: _numeric_value(
                raw_row, columns.normalized.value
            ),
            DiagnosticJsonKey.raw_column.value: columns.raw.value,
            DiagnosticJsonKey.normalized_column.value: columns.normalized.value,
        }
    return output


def _null_diagnostic_values(
    trajectory: Mapping[str, object], raw_row: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for label, columns in _metric_columns(trajectory).items():
        raw_null_column = _null_column(columns.raw)
        normalized_null_column = _null_column(columns.normalized)
        output[label.value] = {
            DiagnosticJsonKey.raw_mean.value: _numeric_value(raw_row, raw_null_column),
            DiagnosticJsonKey.normalized_mean.value: _numeric_value(
                raw_row, normalized_null_column
            ),
            DiagnosticJsonKey.raw_null_column.value: raw_null_column,
            DiagnosticJsonKey.normalized_null_column.value: normalized_null_column,
        }
    return output


def _metric_columns(
    trajectory: Mapping[str, object],
) -> dict[DiagnosticLabel, MetricColumns]:
    is_rnn = trajectory[TrajectoryKey.policy_architecture.value] == "rnn"
    return {
        DiagnosticLabel.OAR: MetricColumns(MetricColumn.oar_raw, MetricColumn.oar_norm),
        DiagnosticLabel.HAR: MetricColumns(
            MetricColumn.har_hidden_raw if is_rnn else MetricColumn.har_ohist_raw,
            MetricColumn.har_hidden_norm if is_rnn else MetricColumn.har_ohist_norm,
        ),
        DiagnosticLabel.PIF: MetricColumns(
            MetricColumn.pif_hidden_raw if is_rnn else MetricColumn.pif_ohist_raw,
            MetricColumn.pif_hidden_norm if is_rnn else MetricColumn.pif_ohist_norm,
        ),
        DiagnosticLabel.AA: MetricColumns(MetricColumn.aa_raw, MetricColumn.aa_norm),
        DiagnosticLabel.DAI: MetricColumns(
            MetricColumn.dai_hidden_raw if is_rnn else MetricColumn.dai_ohist_raw,
            MetricColumn.dai_hidden_norm if is_rnn else MetricColumn.dai_ohist_norm,
        ),
    }


def _undefined_normalized(
    diagnostics: Mapping[str, Mapping[str, object]]
) -> list[dict[str, str]]:
    undefined: list[dict[str, str]] = []
    for diagnostic_label, values in diagnostics.items():
        normalized_value = values.get(DiagnosticJsonKey.normalized.value)
        if _is_nan(normalized_value):
            undefined.append(
                {
                    DiagnosticJsonKey.metric.value: f"{diagnostic_label}norm",
                    DiagnosticJsonKey.undefined_reason.value: (
                        UndefinedReason.zero_conditional_action_entropy.value
                    ),
                }
            )
    return undefined


def _result_mapping(result: object, attribute: ResultAttribute) -> dict[str, object]:
    value = getattr(result, attribute.value)
    if not isinstance(value, Mapping):
        raise ValueError(f"Diagnostic result {attribute.value} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _agent_arrays(
    trajectory: Mapping[str, object], trajectory_key: TrajectoryKey
) -> dict[str, np.ndarray]:
    value = trajectory[trajectory_key.value]
    if not isinstance(value, Mapping):
        raise ValueError(f"{trajectory_key.value} must be a mapping")
    return _mapping_to_arrays(value)


def _mapping_to_arrays(value: Mapping[object, object]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("Agent keys must be strings")
        arrays[key] = np.asarray(item)
    return arrays


def _numeric_value(row: Mapping[str, object], column: str) -> float:
    value = row.get(column, math.nan)
    if value is None:
        return math.nan
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int | float):
        return float(value)
    return math.nan


def _null_column(column: MetricColumn) -> str:
    return f"{column.value}_null"


def _nested_mapping(
    source: Mapping[str, object], key: DiagnosticJsonKey | str
) -> Mapping[str, object]:
    key_value = key.value if isinstance(key, DiagnosticJsonKey) else key
    value = source.get(key_value, {})
    if isinstance(value, Mapping):
        return value
    return {}


def _wandb_metric_name(name: str) -> str:
    return f"{WandbMetricPrefix.diagnostics_final.value}/{name}"


def _above_null(value: object, null_value: object) -> bool:
    if not isinstance(value, int | float) or not isinstance(null_value, int | float):
        return False
    if not (math.isfinite(float(value)) and math.isfinite(float(null_value))):
        return False
    return float(value) > float(null_value)


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _to_jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
