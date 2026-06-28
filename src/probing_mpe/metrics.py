from __future__ import annotations

import argparse
import importlib
import json
import math
import pickle
import sys
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

import numpy as np

from probing_mpe.trajectories import TrajectoryKey, validate_trajectory_schema


JSON_INDENT = 2
WORLD_DIMENSIONS = 2
SIMPLE_SPREAD_AGENT_COUNT = 3
SIMPLE_SPREAD_LANDMARK_COUNT = 3
SIMPLE_SPREAD_VELOCITY_START = 0
SIMPLE_SPREAD_POSITION_START = 2
SIMPLE_SPREAD_LANDMARK_START = 4
SIMPLE_SPREAD_AGENT_SIZE = 0.15
SIMPLE_SPREAD_COLLISION_DISTANCE = SIMPLE_SPREAD_AGENT_SIZE * 2.0
SIMPLE_SPEAKER_LISTENER_LANDMARK_COUNT = 3
SIMPLE_SPEAKER_LISTENER_GOAL_COLOR_DIMENSIONS = 3
SIMPLE_SPEAKER_LISTENER_LISTENER_LANDMARK_START = 2


class BackendModuleName(str, Enum):
    wandb = "wandb"


class BackendFunctionName(str, Enum):
    init = "init"
    log = "log"
    finish = "finish"


class EnvironmentName(str, Enum):
    simple_spread = "simple_spread_v3"
    simple_speaker_listener = "simple_speaker_listener_v4"


class AgentRole(str, Enum):
    agent = "agent"
    speaker = "speaker"
    listener = "listener"


class BehavioralMetricKey(str, Enum):
    metadata = "metadata"
    behavioral_metrics = "behavioral_metrics"
    omitted_metrics = "omitted_metrics"
    metric = "metric"
    reason = "reason"


class MetadataKey(str, Enum):
    env_name = "env_name"
    algorithm = "algorithm"
    policy_architecture = "policy_architecture"
    config_id = "config_id"
    seed = "seed"
    parameter_sharing = "parameter_sharing"
    training_progress_percent = "training_progress_percent"


class MetricName(str, Enum):
    return_mean = "eval/return_mean"
    return_std = "eval/return_std"
    return_min = "eval/return_min"
    return_max = "eval/return_max"
    coverage_success_rate = "eval/coverage_success_rate"
    final_landmark_distance_mean = "eval/final_landmark_distance_mean"
    collision_rate = "eval/collision_rate"
    duplicate_coverage_rate = "eval/duplicate_coverage_rate"
    target_success_rate = "eval/target_success_rate"
    final_target_distance_mean = "eval/final_target_distance_mean"
    wrong_landmark_rate = "eval/wrong_landmark_rate"
    speaker_message_target_association = "eval/speaker_message_target_association"


class OmissionReason(str, Enum):
    unknown_environment = "No environment-specific behavioral metrics are defined for"
    missing_simple_spread_shape = "Simple Spread observations do not expose the expected position and landmark fields"
    missing_speaker_listener_roles = "Speaker-Listener trajectory does not include speaker and listener roles"
    missing_speaker_listener_shape = "Speaker-Listener observations do not expose target color and listener landmark fields"


def load_trajectory(path: Path) -> dict[str, object]:
    with path.open("rb") as trajectory_file:
        loaded = pickle.load(trajectory_file)
    if not isinstance(loaded, Mapping):
        raise ValueError("Trajectory file must contain a mapping")
    trajectory = dict(loaded)
    validate_trajectory_schema(trajectory)
    return trajectory


def compute_behavioral_metrics(trajectory: Mapping[str, object]) -> dict[str, object]:
    validate_trajectory_schema(trajectory)
    metrics = _return_metrics(trajectory)
    omitted_metrics: list[dict[str, str]] = []
    env_name = str(trajectory[TrajectoryKey.env_name.value])

    if env_name == EnvironmentName.simple_spread.value:
        spread_metrics, spread_omissions = _simple_spread_metrics(trajectory)
        metrics.update(spread_metrics)
        omitted_metrics.extend(spread_omissions)
    elif env_name == EnvironmentName.simple_speaker_listener.value:
        speaker_metrics, speaker_omissions = _speaker_listener_metrics(trajectory)
        metrics.update(speaker_metrics)
        omitted_metrics.extend(speaker_omissions)
    else:
        omitted_metrics.append(
            _omission(
                MetricName.coverage_success_rate,
                f"{OmissionReason.unknown_environment.value} {env_name}",
            )
        )

    return {
        BehavioralMetricKey.metadata.value: _metadata(trajectory),
        BehavioralMetricKey.behavioral_metrics.value: metrics,
        BehavioralMetricKey.omitted_metrics.value: omitted_metrics,
    }


def write_behavioral_metrics(result: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_to_jsonable(result), indent=JSON_INDENT, allow_nan=True),
        encoding="utf-8",
    )


def flatten_wandb_metrics(result: Mapping[str, object]) -> dict[str, object]:
    metrics = result.get(BehavioralMetricKey.behavioral_metrics.value, {})
    if not isinstance(metrics, Mapping):
        return {}
    return {str(key): value for key, value in metrics.items()}


def log_behavioral_metrics_to_wandb(
    result: Mapping[str, object],
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
    log(flatten_wandb_metrics(result))

    if created_run:
        finish = getattr(wandb_module, BackendFunctionName.finish.value)
        if callable(finish):
            finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute behavioral metrics from a BenchMARL trajectory export."
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectory = load_trajectory(args.trajectory)
    result = compute_behavioral_metrics(trajectory)
    write_behavioral_metrics(result, args.output)
    log_behavioral_metrics_to_wandb(
        result=result,
        enabled=bool(args.wandb),
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        mode=args.wandb_mode,
    )
    print(f"Saved {args.output}")
    return 0


def _return_metrics(trajectory: Mapping[str, object]) -> dict[str, float]:
    rewards = _agent_arrays(trajectory, TrajectoryKey.rewards)
    episode_ids = _agent_arrays(trajectory, TrajectoryKey.episode_ids)
    reference_agent = sorted(rewards)[0]
    returns: list[float] = []
    for episode_id in sorted(np.unique(episode_ids[reference_agent]).tolist()):
        per_agent_returns = []
        for agent_name, agent_rewards in rewards.items():
            agent_episode_ids = episode_ids[agent_name]
            per_agent_returns.append(float(np.sum(agent_rewards[agent_episode_ids == episode_id])))
        returns.append(float(np.mean(per_agent_returns)))

    returns_array = np.asarray(returns, dtype=float)
    return {
        MetricName.return_mean.value: float(np.mean(returns_array)),
        MetricName.return_std.value: float(np.std(returns_array)),
        MetricName.return_min.value: float(np.min(returns_array)),
        MetricName.return_max.value: float(np.max(returns_array)),
    }


def _simple_spread_metrics(
    trajectory: Mapping[str, object]
) -> tuple[dict[str, float], list[dict[str, str]]]:
    observations = _agent_arrays(trajectory, TrajectoryKey.observations)
    if len(observations) != SIMPLE_SPREAD_AGENT_COUNT or any(
        observation.shape[1] < _simple_spread_min_observation_size()
        for observation in observations.values()
    ):
        return {}, [
            _omission(
                MetricName.coverage_success_rate,
                OmissionReason.missing_simple_spread_shape.value,
            )
        ]

    episode_ids = _agent_arrays(trajectory, TrajectoryKey.episode_ids)
    timesteps = _agent_arrays(trajectory, TrajectoryKey.timesteps)
    reference_agent = sorted(observations)[0]
    coverage_successes: list[float] = []
    final_distance_means: list[float] = []
    duplicate_coverages: list[float] = []
    collision_counts: list[float] = []

    for episode_id in sorted(np.unique(episode_ids[reference_agent]).tolist()):
        final_indices = _final_indices_for_episode(episode_ids, timesteps, episode_id)
        final_observations = {
            agent_name: observations[agent_name][final_indices[agent_name]]
            for agent_name in observations
        }
        landmark_distances = _simple_spread_landmark_distances(final_observations)
        nearest_landmark_distances = np.min(landmark_distances, axis=0)
        final_distance_means.append(float(np.mean(nearest_landmark_distances)))
        coverage_successes.append(
            float(np.all(nearest_landmark_distances <= SIMPLE_SPREAD_AGENT_SIZE))
        )
        duplicate_coverages.append(
            float(_has_duplicate_coverage(landmark_distances, nearest_landmark_distances))
        )
        collision_counts.append(float(_simple_spread_collision_count(final_observations)))

    return (
        {
            MetricName.coverage_success_rate.value: float(np.mean(coverage_successes)),
            MetricName.final_landmark_distance_mean.value: float(np.mean(final_distance_means)),
            MetricName.collision_rate.value: float(np.mean(collision_counts)),
            MetricName.duplicate_coverage_rate.value: float(np.mean(duplicate_coverages)),
        },
        [],
    )


def _speaker_listener_metrics(
    trajectory: Mapping[str, object]
) -> tuple[dict[str, float], list[dict[str, str]]]:
    observations = _agent_arrays(trajectory, TrajectoryKey.observations)
    actions = _agent_arrays(trajectory, TrajectoryKey.actions_diagnostic)
    role_map = _string_mapping(trajectory, TrajectoryKey.agent_role_map)
    speaker_name = _agent_for_role(role_map, AgentRole.speaker)
    listener_name = _agent_for_role(role_map, AgentRole.listener)
    if speaker_name is None or listener_name is None:
        return {}, [
            _omission(
                MetricName.target_success_rate,
                OmissionReason.missing_speaker_listener_roles.value,
            )
        ]

    speaker_observations = observations[speaker_name]
    listener_observations = observations[listener_name]
    if (
        speaker_observations.shape[1] < SIMPLE_SPEAKER_LISTENER_GOAL_COLOR_DIMENSIONS
        or listener_observations.shape[1] < _speaker_listener_min_listener_observation_size()
    ):
        return {}, [
            _omission(
                MetricName.target_success_rate,
                OmissionReason.missing_speaker_listener_shape.value,
            )
        ]

    episode_ids = _agent_arrays(trajectory, TrajectoryKey.episode_ids)
    timesteps = _agent_arrays(trajectory, TrajectoryKey.timesteps)
    final_target_distances: list[float] = []
    target_successes: list[float] = []
    wrong_landmark_values: list[float] = []
    for episode_id in sorted(np.unique(episode_ids[speaker_name]).tolist()):
        final_indices = _final_indices_for_episode(episode_ids, timesteps, episode_id)
        target_index = _target_index(speaker_observations[final_indices[speaker_name]])
        distances = _speaker_listener_landmark_distances(
            listener_observations[final_indices[listener_name]]
        )
        target_distance = float(distances[target_index])
        final_target_distances.append(target_distance)
        nearest_index = int(np.argmin(distances))
        target_successes.append(float(nearest_index == target_index))
        wrong_landmark_values.append(float(nearest_index != target_index))

    target_indices = np.asarray(
        [_target_index(observation) for observation in speaker_observations],
        dtype=np.int64,
    )
    speaker_actions = np.asarray(actions[speaker_name], dtype=np.int64)

    return (
        {
            MetricName.target_success_rate.value: float(np.mean(target_successes)),
            MetricName.final_target_distance_mean.value: float(np.mean(final_target_distances)),
            MetricName.wrong_landmark_rate.value: float(np.mean(wrong_landmark_values)),
            MetricName.speaker_message_target_association.value: _message_target_association(
                speaker_actions, target_indices
            ),
        },
        [],
    )


def _simple_spread_landmark_distances(
    observations: Mapping[str, np.ndarray]
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for observation in observations.values():
        distances = []
        for landmark_index in range(SIMPLE_SPREAD_LANDMARK_COUNT):
            offset = SIMPLE_SPREAD_LANDMARK_START + landmark_index * WORLD_DIMENSIONS
            distances.append(float(np.linalg.norm(observation[offset : offset + WORLD_DIMENSIONS])))
        rows.append(np.asarray(distances, dtype=float))
    return np.stack(rows, axis=0)


def _simple_spread_collision_count(observations: Mapping[str, np.ndarray]) -> int:
    agent_names = sorted(observations)
    positions = {
        agent_name: observations[agent_name][
            SIMPLE_SPREAD_POSITION_START : SIMPLE_SPREAD_POSITION_START + WORLD_DIMENSIONS
        ]
        for agent_name in agent_names
    }
    collisions = 0
    for left_index, left_agent in enumerate(agent_names):
        for right_agent in agent_names[left_index + 1 :]:
            distance = float(np.linalg.norm(positions[left_agent] - positions[right_agent]))
            if distance < SIMPLE_SPREAD_COLLISION_DISTANCE:
                collisions += 1
    return collisions


def _has_duplicate_coverage(
    landmark_distances: np.ndarray, nearest_landmark_distances: np.ndarray
) -> bool:
    nearest_landmarks_by_agent = np.argmin(landmark_distances, axis=1)
    unique_nearest = set(int(value) for value in nearest_landmarks_by_agent.tolist())
    has_duplicate = len(unique_nearest) < len(nearest_landmarks_by_agent)
    has_uncovered = bool(np.any(nearest_landmark_distances > SIMPLE_SPREAD_AGENT_SIZE))
    return has_duplicate and has_uncovered


def _speaker_listener_landmark_distances(observation: np.ndarray) -> np.ndarray:
    distances = []
    for landmark_index in range(SIMPLE_SPEAKER_LISTENER_LANDMARK_COUNT):
        offset = (
            SIMPLE_SPEAKER_LISTENER_LISTENER_LANDMARK_START
            + landmark_index * WORLD_DIMENSIONS
        )
        distances.append(float(np.linalg.norm(observation[offset : offset + WORLD_DIMENSIONS])))
    return np.asarray(distances, dtype=float)


def _message_target_association(actions: np.ndarray, target_indices: np.ndarray) -> float:
    correct = 0
    for message in np.unique(actions).tolist():
        mask = actions == message
        targets_for_message = target_indices[mask]
        if len(targets_for_message) == 0:
            continue
        _, counts = np.unique(targets_for_message, return_counts=True)
        correct += int(np.max(counts))
    return float(correct / len(actions)) if len(actions) else math.nan


def _target_index(speaker_observation: np.ndarray) -> int:
    return int(np.argmax(speaker_observation[:SIMPLE_SPEAKER_LISTENER_GOAL_COLOR_DIMENSIONS]))


def _final_indices_for_episode(
    episode_ids: Mapping[str, np.ndarray],
    timesteps: Mapping[str, np.ndarray],
    episode_id: int,
) -> dict[str, int]:
    output: dict[str, int] = {}
    for agent_name, agent_episode_ids in episode_ids.items():
        matching_indices = np.where(agent_episode_ids == episode_id)[0]
        agent_timesteps = timesteps[agent_name][matching_indices]
        output[agent_name] = int(matching_indices[int(np.argmax(agent_timesteps))])
    return output


def _simple_spread_min_observation_size() -> int:
    return SIMPLE_SPREAD_LANDMARK_START + SIMPLE_SPREAD_LANDMARK_COUNT * WORLD_DIMENSIONS


def _speaker_listener_min_listener_observation_size() -> int:
    return (
        SIMPLE_SPEAKER_LISTENER_LISTENER_LANDMARK_START
        + SIMPLE_SPEAKER_LISTENER_LANDMARK_COUNT * WORLD_DIMENSIONS
    )


def _agent_for_role(role_map: Mapping[str, str], role: AgentRole) -> str | None:
    for agent_name, agent_role in role_map.items():
        if agent_role == role.value:
            return agent_name
    return None


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
    }


def _agent_arrays(
    trajectory: Mapping[str, object], trajectory_key: TrajectoryKey
) -> dict[str, np.ndarray]:
    value = trajectory[trajectory_key.value]
    if not isinstance(value, Mapping):
        raise ValueError(f"{trajectory_key.value} must be a mapping")
    output: dict[str, np.ndarray] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("Agent keys must be strings")
        output[key] = np.asarray(item)
    return output


def _string_mapping(
    trajectory: Mapping[str, object], trajectory_key: TrajectoryKey
) -> dict[str, str]:
    value = trajectory[trajectory_key.value]
    if not isinstance(value, Mapping):
        raise ValueError(f"{trajectory_key.value} must be a mapping")
    output: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, str):
            output[key] = item
    return output


def _omission(metric: MetricName, reason: str) -> dict[str, str]:
    return {
        BehavioralMetricKey.metric.value: metric.value,
        BehavioralMetricKey.reason.value: reason,
    }


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
