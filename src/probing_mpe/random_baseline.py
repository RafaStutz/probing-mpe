from __future__ import annotations

import argparse
import pickle
import math
from pathlib import Path
from typing import Sequence, Mapping

import numpy as np
import torch

from probing_mpe.trajectories import (
    TrajectoryKey,
    PolicyArchitecture,
    TARGET_DIAGNOSTIC_TRANSITIONS,
    DEFAULT_PROGRESS_PERCENT,
    _chunk_from_rollout,
    merge_agent_chunks,
    build_agent_role_map,
    build_action_space_description,
    validate_trajectory_schema,
    _agent_array_mapping_object,
)


def _episodes_for_target(target_transitions: int, max_steps: int) -> int:
    return max(1, math.ceil(target_transitions / max_steps))


def export_random_trajectory(
    env_name: str,
    output_path: Path,
    episodes: int | None,
    target_transitions: int,
    seed: int,
) -> dict[str, object]:
    
    # BenchMARL environment loading
    from benchmarl.environments import Task
    
    if env_name == "simple_spread_v3":
        task = Task.PETTINGZOO_SIMPLE_SPREAD.get_from_yaml()
        task.task = "simple_spread_v3"
        task.max_cycles = 25
    elif env_name == "simple_speaker_listener_v4":
        task = Task.PETTINGZOO_SIMPLE_SPEAKER_LISTENER.get_from_yaml()
        task.task = "simple_speaker_listener_v4"
        task.max_cycles = 25
    else:
        raise ValueError(f"Unknown environment: {env_name}")
        
    env = task.get_env_and_group_map(seed=seed)[0]
    group_map = {
        str(group_name): [str(agent_name) for agent_name in agent_names]
        for group_name, agent_names in task.group_map.items()
    }
    
    episode_count = episodes or _episodes_for_target(
        target_transitions=target_transitions,
        max_steps=task.max_cycles,
    )
    
    rollouts = []
    with torch.no_grad():
        for _ in range(episode_count):
            # policy=None means uniform random actions drawn from action_spec
            rollouts.append(
                env.rollout(
                    max_steps=task.max_cycles,
                    policy=None,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
            )

    # Use existing parsing logic
    chunks = [
        _chunk_from_rollout(
            rollout=rollout,
            group_map=group_map,
            episode_id=episode_id,
            include_hidden_states=False,
        )
        for episode_id, rollout in enumerate(rollouts)
    ]
    
    merged = merge_agent_chunks(chunks=chunks, include_hidden_states=False)
    
    # Build the trajectory dictionary directly without needing an Experiment object
    trajectory = {
        TrajectoryKey.env_name.value: env_name,
        TrajectoryKey.algorithm.value: "random",
        TrajectoryKey.policy_architecture.value: PolicyArchitecture.ff.value,
        TrajectoryKey.config_id.value: "random_baseline",
        TrajectoryKey.seed.value: seed,
        TrajectoryKey.parameter_sharing.value: False,
        TrajectoryKey.training_progress_percent.value: 0,
        TrajectoryKey.observations.value: merged.observations,
        TrajectoryKey.actions_raw.value: merged.actions_raw,
        TrajectoryKey.actions_diagnostic.value: merged.actions_diagnostic,
        TrajectoryKey.rewards.value: merged.rewards,
        TrajectoryKey.timesteps.value: merged.timesteps,
        TrajectoryKey.episode_ids.value: merged.episode_ids,
        TrajectoryKey.dones.value: merged.dones,
        TrajectoryKey.infos.value: merged.infos,
        TrajectoryKey.global_state.value: merged.global_state,
        TrajectoryKey.hidden_states.value: None,
        TrajectoryKey.agent_role_map.value: build_agent_role_map(group_map),
        TrajectoryKey.action_space_description.value: build_action_space_description(
            group_map, env.action_spec
        ),
    }
    
    validate_trajectory_schema(trajectory)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        pickle.dump(trajectory, output_file)
        
    return trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a pure random baseline trajectory using TorchRL default samplers."
    )
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument("--output", type=Path, required=True, help="Output PKL path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--target-transitions", type=int, default=TARGET_DIAGNOSTIC_TRANSITIONS
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectory = export_random_trajectory(
        env_name=args.env,
        output_path=args.output,
        episodes=args.episodes,
        target_transitions=args.target_transitions,
        seed=args.seed,
    )
    observations = _agent_array_mapping_object(trajectory[TrajectoryKey.observations.value])
    sample_count = min(len(values) for values in observations.values())
    print(
        f"Saved random baseline {args.output} with {len(observations)} agents and {sample_count} samples per agent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
