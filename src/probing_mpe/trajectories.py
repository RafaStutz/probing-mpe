from __future__ import annotations

import argparse
import math
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


TARGET_DIAGNOSTIC_TRANSITIONS = 8000
DEFAULT_PROGRESS_PERCENT = 100
RNN_HIDDEN_PREFIX = "_hidden_gru_"
SINGLE_LAYER_SIZE = 1
RNN_LAYER_AXIS = -2


class TrajectoryKey(str, Enum):
    env_name = "env_name"
    algorithm = "algorithm"
    policy_architecture = "policy_architecture"
    config_id = "config_id"
    seed = "seed"
    parameter_sharing = "parameter_sharing"
    training_progress_percent = "training_progress_percent"
    observations = "observations"
    actions_raw = "actions_raw"
    actions_diagnostic = "actions_diagnostic"
    rewards = "rewards"
    timesteps = "timesteps"
    episode_ids = "episode_ids"
    dones = "dones"
    infos = "infos"
    global_state = "global_state"
    hidden_states = "hidden_states"
    agent_role_map = "agent_role_map"
    action_space_description = "action_space_description"


class TensorKey(str, Enum):
    observation = "observation"
    action = "action"
    reward = "reward"
    done = "done"
    state = "state"
    info = "info"
    next = "next"


class GroupName(str, Enum):
    agent = "agent"
    speaker = "speaker"
    listener = "listener"


class PolicyArchitecture(str, Enum):
    ff = "ff"
    rnn = "rnn"


class ActionDescriptionKey(str, Enum):
    type = "type"
    encoding = "encoding"
    raw_spec = "raw_spec"
    meaning = "meaning"


class ActionEncoding(str, Enum):
    categorical_discrete = "categorical_discrete"
    integer_index = "integer_index"


@dataclass(frozen=True)
class AgentChunk:
    observations: Mapping[str, np.ndarray]
    actions_raw: Mapping[str, np.ndarray]
    actions_diagnostic: Mapping[str, np.ndarray]
    rewards: Mapping[str, np.ndarray]
    timesteps: Mapping[str, np.ndarray]
    episode_ids: Mapping[str, np.ndarray]
    dones: Mapping[str, np.ndarray]
    infos: Mapping[str, list[dict[str, object]]]
    hidden_states: Mapping[str, np.ndarray] | None
    global_state: np.ndarray | None


@dataclass(frozen=True)
class MergedTrajectoryArrays:
    observations: dict[str, np.ndarray]
    actions_raw: dict[str, np.ndarray]
    actions_diagnostic: dict[str, np.ndarray]
    rewards: dict[str, np.ndarray]
    timesteps: dict[str, np.ndarray]
    episode_ids: dict[str, np.ndarray]
    dones: dict[str, np.ndarray]
    infos: dict[str, list[dict[str, object]]]
    hidden_states: dict[str, np.ndarray] | None
    global_state: np.ndarray | None


def build_agent_role_map(group_map: Mapping[str, Sequence[str]]) -> dict[str, str]:
    role_map: dict[str, str] = {}
    for group_name, agent_names in group_map.items():
        if group_name == GroupName.speaker.value:
            role = GroupName.speaker.value
        elif group_name == GroupName.listener.value:
            role = GroupName.listener.value
        else:
            role = GroupName.agent.value
        for agent_name in agent_names:
            role_map[agent_name] = role
    return role_map


def build_action_space_description(
    group_map: Mapping[str, Sequence[str]], action_spec: object
) -> dict[str, dict[str, str]]:
    action_spec_text = str(action_spec)
    descriptions: dict[str, dict[str, str]] = {}
    for group_name, agent_names in group_map.items():
        meaning = _action_meaning(group_name)
        for agent_name in agent_names:
            descriptions[agent_name] = {
                ActionDescriptionKey.type.value: ActionEncoding.categorical_discrete.value,
                ActionDescriptionKey.encoding.value: ActionEncoding.integer_index.value,
                ActionDescriptionKey.raw_spec.value: action_spec_text,
                ActionDescriptionKey.meaning.value: meaning,
            }
    return descriptions


def merge_agent_chunks(
    chunks: Sequence[AgentChunk], include_hidden_states: bool
) -> MergedTrajectoryArrays:
    agent_names = _agent_names_from_chunks(chunks)
    hidden_states: dict[str, np.ndarray] | None = {} if include_hidden_states else None
    global_state_chunks = [
        chunk.global_state for chunk in chunks if chunk.global_state is not None
    ]

    merged = MergedTrajectoryArrays(
        observations=_concatenate_agent_arrays(chunks, agent_names, TrajectoryKey.observations),
        actions_raw=_concatenate_agent_arrays(chunks, agent_names, TrajectoryKey.actions_raw),
        actions_diagnostic=_concatenate_agent_arrays(
            chunks, agent_names, TrajectoryKey.actions_diagnostic
        ),
        rewards=_concatenate_agent_arrays(chunks, agent_names, TrajectoryKey.rewards),
        timesteps=_concatenate_agent_arrays(chunks, agent_names, TrajectoryKey.timesteps),
        episode_ids=_concatenate_agent_arrays(
            chunks, agent_names, TrajectoryKey.episode_ids
        ),
        dones=_concatenate_agent_arrays(chunks, agent_names, TrajectoryKey.dones),
        infos=_concatenate_agent_infos(chunks, agent_names),
        hidden_states=hidden_states,
        global_state=(
            np.concatenate(global_state_chunks, axis=0) if global_state_chunks else None
        ),
    )

    if include_hidden_states:
        if merged.hidden_states is None:
            raise ValueError("hidden_states merge failed")
        for agent_name in agent_names:
            arrays: list[np.ndarray] = []
            for chunk in chunks:
                if chunk.hidden_states is None:
                    raise ValueError("hidden_states are required for recurrent exports")
                arrays.append(chunk.hidden_states[agent_name])
            merged.hidden_states[agent_name] = np.concatenate(arrays, axis=0)

    return merged


def validate_trajectory_schema(trajectory: Mapping[str, object]) -> None:
    for trajectory_key in TrajectoryKey:
        if trajectory_key.value not in trajectory:
            raise ValueError(f"Missing trajectory field: {trajectory_key.value}")

    observations = _agent_array_mapping(trajectory, TrajectoryKey.observations)
    actions_raw = _agent_array_mapping(trajectory, TrajectoryKey.actions_raw)
    actions_diagnostic = _agent_array_mapping(
        trajectory, TrajectoryKey.actions_diagnostic
    )
    rewards = _agent_array_mapping(trajectory, TrajectoryKey.rewards)
    timesteps = _agent_array_mapping(trajectory, TrajectoryKey.timesteps)
    episode_ids = _agent_array_mapping(trajectory, TrajectoryKey.episode_ids)
    dones = _agent_array_mapping(trajectory, TrajectoryKey.dones)
    role_map = _string_mapping(trajectory, TrajectoryKey.agent_role_map)

    agent_names = set(observations)
    for field_name, mapping in (
        (TrajectoryKey.actions_raw.value, actions_raw),
        (TrajectoryKey.actions_diagnostic.value, actions_diagnostic),
        (TrajectoryKey.rewards.value, rewards),
        (TrajectoryKey.timesteps.value, timesteps),
        (TrajectoryKey.episode_ids.value, episode_ids),
        (TrajectoryKey.dones.value, dones),
    ):
        if set(mapping) != agent_names:
            raise ValueError(f"{field_name} agents do not match observations")

    if set(role_map) != agent_names:
        raise ValueError("agent_role_map agents do not match observations")

    reference_timesteps: np.ndarray | None = None
    reference_episode_ids: np.ndarray | None = None
    for agent_name in sorted(agent_names):
        sample_count = len(observations[agent_name])
        for field_name, mapping in (
            (TrajectoryKey.actions_raw.value, actions_raw),
            (TrajectoryKey.actions_diagnostic.value, actions_diagnostic),
            (TrajectoryKey.rewards.value, rewards),
            (TrajectoryKey.timesteps.value, timesteps),
            (TrajectoryKey.episode_ids.value, episode_ids),
            (TrajectoryKey.dones.value, dones),
        ):
            if len(mapping[agent_name]) != sample_count:
                raise ValueError(f"{field_name} length mismatch for {agent_name}")

        if reference_timesteps is None:
            reference_timesteps = timesteps[agent_name]
            reference_episode_ids = episode_ids[agent_name]
        elif not (
            np.array_equal(reference_timesteps, timesteps[agent_name])
            and np.array_equal(reference_episode_ids, episode_ids[agent_name])
        ):
            raise ValueError("episode IDs and timesteps must align across agents")

    policy_architecture = trajectory[TrajectoryKey.policy_architecture.value]
    if policy_architecture == PolicyArchitecture.rnn.value:
        hidden_states_object = trajectory[TrajectoryKey.hidden_states.value]
        if not isinstance(hidden_states_object, Mapping):
            raise ValueError("hidden_states are required for recurrent trajectories")
        hidden_states = _mapping_to_arrays(hidden_states_object)
        if set(hidden_states) != agent_names:
            raise ValueError("hidden_states agents do not match observations")
        for agent_name in sorted(agent_names):
            if len(hidden_states[agent_name]) != len(observations[agent_name]):
                raise ValueError(f"hidden_states length mismatch for {agent_name}")


def export_trajectory_from_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    progress_percent: int,
    episodes: int | None,
    target_transitions: int,
    env_name_override: str | None,
    config_id_override: str | None,
) -> dict[str, object]:
    from benchmarl.experiment import Experiment

    experiment = Experiment.reload_from_file(
        str(checkpoint_path),
        experiment_patch={
            "evaluation_episodes": episodes or 1,
            "render": False,
        },
    )
    try:
        episode_count = episodes or _episodes_for_target(
            target_transitions=target_transitions,
            max_steps=experiment.max_steps,
        )
        rollouts = _collect_rollouts(experiment, episode_count)
        policy_architecture = _policy_architecture(experiment)
        group_map = {
            str(group_name): [str(agent_name) for agent_name in agent_names]
            for group_name, agent_names in experiment.group_map.items()
        }
        chunks = [
            _chunk_from_rollout(
                rollout=rollout,
                group_map=group_map,
                episode_id=episode_id,
                include_hidden_states=policy_architecture == PolicyArchitecture.rnn.value,
            )
            for episode_id, rollout in enumerate(rollouts)
        ]
        merged = merge_agent_chunks(
            chunks=chunks,
            include_hidden_states=policy_architecture == PolicyArchitecture.rnn.value,
        )
        trajectory = _build_trajectory(
            experiment=experiment,
            merged=merged,
            group_map=group_map,
            progress_percent=progress_percent,
            policy_architecture=policy_architecture,
            env_name_override=env_name_override,
            config_id_override=config_id_override,
        )
        validate_trajectory_schema(trajectory)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as output_file:
            pickle.dump(trajectory, output_file)
        return trajectory
    finally:
        close = getattr(experiment.test_env, "close", None)
        if callable(close):
            close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a BenchMARL checkpoint evaluation trajectory."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=int, default=DEFAULT_PROGRESS_PERCENT)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--target-transitions", type=int, default=TARGET_DIAGNOSTIC_TRANSITIONS
    )
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--config-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectory = export_trajectory_from_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        progress_percent=args.progress,
        episodes=args.episodes,
        target_transitions=args.target_transitions,
        env_name_override=args.env_name,
        config_id_override=args.config_id,
    )
    observations = _agent_array_mapping_object(trajectory[TrajectoryKey.observations.value])
    sample_count = min(len(values) for values in observations.values())
    print(
        f"Saved {args.output} with {len(observations)} agents and {sample_count} samples per agent"
    )
    return 0


def _collect_rollouts(experiment: object, episodes: int) -> list[object]:
    import torch
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    rollouts: list[object] = []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(episodes):
            rollouts.append(
                experiment.test_env.rollout(
                    max_steps=experiment.max_steps,
                    policy=experiment.policy,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
            )
    return rollouts


def _chunk_from_rollout(
    rollout: object,
    group_map: Mapping[str, Sequence[str]],
    episode_id: int,
    include_hidden_states: bool,
) -> AgentChunk:
    observations: dict[str, np.ndarray] = {}
    actions_raw: dict[str, np.ndarray] = {}
    actions_diagnostic: dict[str, np.ndarray] = {}
    rewards: dict[str, np.ndarray] = {}
    timesteps: dict[str, np.ndarray] = {}
    episode_ids: dict[str, np.ndarray] = {}
    dones: dict[str, np.ndarray] = {}
    infos: dict[str, list[dict[str, object]]] = {}
    hidden_states: dict[str, np.ndarray] | None = {} if include_hidden_states else None

    for group_name, agent_names in group_map.items():
        group_observations = _to_numpy(_get_rollout_value(rollout, (group_name, TensorKey.observation.value)))
        group_actions = _to_numpy(_get_rollout_value(rollout, (group_name, TensorKey.action.value)))
        group_rewards = _squeeze_last(
            _to_numpy(
                _get_rollout_value(
                    rollout,
                    (
                        TensorKey.next.value,
                        group_name,
                        TensorKey.reward.value,
                    ),
                )
            )
        )
        group_dones = _squeeze_last(
            _to_numpy(
                _get_rollout_value(
                    rollout,
                    (
                        TensorKey.next.value,
                        group_name,
                        TensorKey.done.value,
                    ),
                )
            )
        )
        group_hidden = (
            _to_numpy(_get_rollout_value(rollout, (group_name, _hidden_state_key(rollout, group_name))))
            if include_hidden_states
            else None
        )

        step_count = len(group_observations)
        for agent_index, agent_name in enumerate(agent_names):
            observations[agent_name] = group_observations[:, agent_index]
            raw_action = group_actions[:, agent_index]
            actions_raw[agent_name] = raw_action
            actions_diagnostic[agent_name] = _diagnostic_action(raw_action)
            rewards[agent_name] = group_rewards[:, agent_index]
            timesteps[agent_name] = np.arange(step_count, dtype=np.int64)
            episode_ids[agent_name] = np.full(step_count, episode_id, dtype=np.int64)
            dones[agent_name] = group_dones[:, agent_index].astype(bool)
            infos[agent_name] = [{} for _ in range(step_count)]
            if include_hidden_states:
                if hidden_states is None or group_hidden is None:
                    raise ValueError("hidden_states are required for recurrent exports")
                hidden_states[agent_name] = _agent_hidden_states(
                    group_hidden[:, agent_index]
                )

    global_state = _optional_numpy(rollout, TensorKey.state.value)
    return AgentChunk(
        observations=observations,
        actions_raw=actions_raw,
        actions_diagnostic=actions_diagnostic,
        rewards=rewards,
        timesteps=timesteps,
        episode_ids=episode_ids,
        dones=dones,
        infos=infos,
        hidden_states=hidden_states,
        global_state=global_state,
    )


def _build_trajectory(
    experiment: object,
    merged: MergedTrajectoryArrays,
    group_map: Mapping[str, Sequence[str]],
    progress_percent: int,
    policy_architecture: str,
    env_name_override: str | None,
    config_id_override: str | None,
) -> dict[str, object]:
    env_name = env_name_override or _env_name(experiment)
    algorithm = _algorithm_name(experiment)
    config_id = config_id_override or f"{algorithm}_{policy_architecture}"
    return {
        TrajectoryKey.env_name.value: env_name,
        TrajectoryKey.algorithm.value: algorithm,
        TrajectoryKey.policy_architecture.value: policy_architecture,
        TrajectoryKey.config_id.value: config_id,
        TrajectoryKey.seed.value: int(experiment.seed),
        TrajectoryKey.parameter_sharing.value: bool(
            experiment.config.share_policy_params
        ),
        TrajectoryKey.training_progress_percent.value: progress_percent,
        TrajectoryKey.observations.value: merged.observations,
        TrajectoryKey.actions_raw.value: merged.actions_raw,
        TrajectoryKey.actions_diagnostic.value: merged.actions_diagnostic,
        TrajectoryKey.rewards.value: merged.rewards,
        TrajectoryKey.timesteps.value: merged.timesteps,
        TrajectoryKey.episode_ids.value: merged.episode_ids,
        TrajectoryKey.dones.value: merged.dones,
        TrajectoryKey.infos.value: merged.infos,
        TrajectoryKey.global_state.value: merged.global_state,
        TrajectoryKey.hidden_states.value: merged.hidden_states,
        TrajectoryKey.agent_role_map.value: build_agent_role_map(group_map),
        TrajectoryKey.action_space_description.value: build_action_space_description(
            group_map, experiment.action_spec
        ),
    }


def _episodes_for_target(target_transitions: int, max_steps: int) -> int:
    return max(1, math.ceil(target_transitions / max_steps))


def _policy_architecture(experiment: object) -> str:
    return (
        PolicyArchitecture.rnn.value
        if bool(experiment.model_config.is_rnn)
        else PolicyArchitecture.ff.value
    )


def _env_name(experiment: object) -> str:
    task_config = getattr(experiment.task, "config", {})
    if isinstance(task_config, Mapping):
        task_name = task_config.get("task")
        if isinstance(task_name, str):
            return task_name
    return str(experiment.task)


def _algorithm_name(experiment: object) -> str:
    name = experiment.algorithm_config.associated_class().__name__.lower()
    return name


def _get_rollout_value(rollout: object, key: str | tuple[str, ...]) -> object:
    get = getattr(rollout, "get")
    return get(key)


def _optional_numpy(rollout: object, key: str | tuple[str, ...]) -> np.ndarray | None:
    try:
        return _to_numpy(_get_rollout_value(rollout, key))
    except KeyError:
        return None


def _to_numpy(value: object) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        return np.asarray(numpy_method())
    return np.asarray(value)


def _squeeze_last(values: np.ndarray) -> np.ndarray:
    if values.ndim > 0 and values.shape[-1] == SINGLE_LAYER_SIZE:
        return np.squeeze(values, axis=-1)
    return values


def _agent_hidden_states(values: np.ndarray) -> np.ndarray:
    if values.ndim >= 2 and values.shape[RNN_LAYER_AXIS] == SINGLE_LAYER_SIZE:
        return np.squeeze(values, axis=RNN_LAYER_AXIS)
    return values


def _diagnostic_action(action: np.ndarray) -> np.ndarray:
    squeezed = _squeeze_last(action)
    return squeezed.astype(np.int64, copy=False)


def _hidden_state_key(rollout: object, group_name: str) -> str:
    keys_method = getattr(rollout, "keys")
    keys = keys_method(True, True)
    for key in keys:
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and key[0] == group_name
            and isinstance(key[1], str)
            and key[1].startswith(RNN_HIDDEN_PREFIX)
        ):
            return key[1]
    raise ValueError(f"No GRU hidden state found for group {group_name}")


def _agent_names_from_chunks(chunks: Sequence[AgentChunk]) -> list[str]:
    if not chunks:
        raise ValueError("At least one trajectory chunk is required")
    return sorted(chunks[0].observations)


def _concatenate_agent_arrays(
    chunks: Sequence[AgentChunk],
    agent_names: Sequence[str],
    trajectory_key: TrajectoryKey,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for agent_name in agent_names:
        arrays = [
            _chunk_agent_mapping(chunk, trajectory_key)[agent_name]
            for chunk in chunks
        ]
        output[agent_name] = np.concatenate(arrays, axis=0)
    return output


def _concatenate_agent_infos(
    chunks: Sequence[AgentChunk], agent_names: Sequence[str]
) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for agent_name in agent_names:
        values: list[dict[str, object]] = []
        for chunk in chunks:
            values.extend(chunk.infos[agent_name])
        output[agent_name] = values
    return output


def _chunk_agent_mapping(
    chunk: AgentChunk, trajectory_key: TrajectoryKey
) -> Mapping[str, np.ndarray]:
    if trajectory_key == TrajectoryKey.observations:
        return chunk.observations
    if trajectory_key == TrajectoryKey.actions_raw:
        return chunk.actions_raw
    if trajectory_key == TrajectoryKey.actions_diagnostic:
        return chunk.actions_diagnostic
    if trajectory_key == TrajectoryKey.rewards:
        return chunk.rewards
    if trajectory_key == TrajectoryKey.timesteps:
        return chunk.timesteps
    if trajectory_key == TrajectoryKey.episode_ids:
        return chunk.episode_ids
    if trajectory_key == TrajectoryKey.dones:
        return chunk.dones
    raise ValueError(f"Unsupported chunk mapping: {trajectory_key.value}")


def _agent_array_mapping(
    trajectory: Mapping[str, object], trajectory_key: TrajectoryKey
) -> dict[str, np.ndarray]:
    return _agent_array_mapping_object(trajectory[trajectory_key.value])


def _agent_array_mapping_object(value: object) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError("Expected per-agent mapping")
    return _mapping_to_arrays(value)


def _mapping_to_arrays(value: Mapping[object, object]) -> dict[str, np.ndarray]:
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
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{trajectory_key.value} must map strings to strings")
        output[key] = item
    return output


def _action_meaning(group_name: str) -> str:
    if group_name == GroupName.speaker.value:
        return "speaker communication symbol"
    if group_name == GroupName.listener.value:
        return "listener movement action"
    return "movement action"


if __name__ == "__main__":
    raise SystemExit(main())
