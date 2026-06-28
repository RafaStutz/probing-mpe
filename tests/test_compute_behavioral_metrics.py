from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from probing_mpe.metrics import (
    BehavioralMetricKey,
    compute_behavioral_metrics,
    flatten_wandb_metrics,
    write_behavioral_metrics,
)
from probing_mpe.trajectories import TrajectoryKey


class ComputeBehavioralMetricsTest(unittest.TestCase):
    def test_simple_spread_metrics_from_observations(self) -> None:
        trajectory = _simple_spread_trajectory()

        result = compute_behavioral_metrics(trajectory)
        metrics = result[BehavioralMetricKey.behavioral_metrics.value]

        self.assertEqual(metrics["eval/coverage_success_rate"], 0.5)
        self.assertAlmostEqual(metrics["eval/final_landmark_distance_mean"], 0.325)
        self.assertEqual(metrics["eval/duplicate_coverage_rate"], 0.5)
        self.assertEqual(metrics["eval/collision_rate"], 0.5)
        self.assertEqual(metrics["eval/return_mean"], 3.0)

    def test_speaker_listener_metrics_from_observations_and_actions(self) -> None:
        trajectory = _speaker_listener_trajectory()

        result = compute_behavioral_metrics(trajectory)
        metrics = result[BehavioralMetricKey.behavioral_metrics.value]

        self.assertEqual(metrics["eval/target_success_rate"], 0.5)
        self.assertEqual(metrics["eval/wrong_landmark_rate"], 0.5)
        self.assertAlmostEqual(metrics["eval/final_target_distance_mean"], 0.35)
        self.assertEqual(metrics["eval/speaker_message_target_association"], 1.0)

    def test_unknown_environment_records_omitted_metrics(self) -> None:
        trajectory = _simple_spread_trajectory()
        trajectory[TrajectoryKey.env_name.value] = "unknown_env"

        result = compute_behavioral_metrics(trajectory)

        self.assertEqual(result[BehavioralMetricKey.behavioral_metrics.value]["eval/return_mean"], 3.0)
        self.assertIn("unknown_env", result[BehavioralMetricKey.omitted_metrics.value][0]["reason"])

    def test_write_behavioral_metrics_outputs_json(self) -> None:
        result = compute_behavioral_metrics(_speaker_listener_trajectory())

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "behavioral_metrics_final.json"
            write_behavioral_metrics(result, output_path)

            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn(BehavioralMetricKey.behavioral_metrics.value, loaded)

    def test_flatten_wandb_metrics_returns_only_metric_values(self) -> None:
        result = compute_behavioral_metrics(_simple_spread_trajectory())

        flattened = flatten_wandb_metrics(result)

        self.assertEqual(flattened["eval/coverage_success_rate"], 0.5)
        self.assertNotIn(BehavioralMetricKey.metadata.value, flattened)


def _simple_spread_trajectory() -> dict[str, object]:
    observations = {
        "agent_0": np.array(
            [
                _spread_observation((0.0, 0.0), ((0.5, 0.0), (0.6, 0.0), (0.7, 0.0))),
                _spread_observation((0.0, 0.0), ((0.1, 0.0), (0.5, 0.0), (0.7, 0.0))),
                _spread_observation((0.0, 0.0), ((0.02, 0.0), (1.0, 0.0), (1.2, 0.0))),
                _spread_observation((0.0, 0.0), ((0.05, 0.0), (0.7, 0.0), (0.9, 0.0))),
            ],
            dtype=np.float32,
        ),
        "agent_1": np.array(
            [
                _spread_observation((1.0, 0.0), ((-0.5, 0.0), (-0.4, 0.0), (-0.3, 0.0))),
                _spread_observation((1.0, 0.0), ((-0.9, 0.0), (-0.1, 0.0), (-0.3, 0.0))),
                _spread_observation((0.1, 0.0), ((-0.08, 0.0), (0.9, 0.0), (1.1, 0.0))),
                _spread_observation((0.0, 0.0), ((0.05, 0.0), (0.7, 0.0), (0.9, 0.0))),
            ],
            dtype=np.float32,
        ),
        "agent_2": np.array(
            [
                _spread_observation((2.0, 0.0), ((-1.5, 0.0), (-1.4, 0.0), (-1.3, 0.0))),
                _spread_observation((2.0, 0.0), ((-1.9, 0.0), (-1.1, 0.0), (-0.1, 0.0))),
                _spread_observation((2.0, 0.0), ((-1.98, 0.0), (-1.0, 0.0), (-0.8, 0.0))),
                _spread_observation((2.0, 0.0), ((-1.95, 0.0), (-1.3, 0.0), (-1.1, 0.0))),
            ],
            dtype=np.float32,
        ),
    }
    return _trajectory(
        env_name="simple_spread_v3",
        observations=observations,
        actions={agent: np.zeros(4, dtype=np.int64) for agent in observations},
        rewards={agent: np.array([1.0, 1.0, 2.0, 2.0]) for agent in observations},
        episode_ids={agent: np.array([0, 0, 1, 1], dtype=np.int64) for agent in observations},
        timesteps={agent: np.array([0, 1, 0, 1], dtype=np.int64) for agent in observations},
        role_map={agent: "agent" for agent in observations},
    )


def _speaker_listener_trajectory() -> dict[str, object]:
    observations = {
        "speaker_0": np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "listener_0": np.array(
            [
                _listener_observation(((0.4, 0.0), (0.7, 0.0), (0.9, 0.0))),
                _listener_observation(((0.1, 0.0), (0.4, 0.0), (0.8, 0.0))),
                _listener_observation(((0.2, 0.0), (0.3, 0.0), (0.8, 0.0))),
                _listener_observation(((0.05, 0.0), (0.6, 0.0), (0.9, 0.0))),
            ],
            dtype=np.float32,
        ),
    }
    return _trajectory(
        env_name="simple_speaker_listener_v4",
        observations=observations,
        actions={
            "speaker_0": np.array([0, 0, 1, 1], dtype=np.int64),
            "listener_0": np.array([0, 0, 0, 0], dtype=np.int64),
        },
        rewards={agent: np.array([1.0, 1.0, 2.0, 2.0]) for agent in observations},
        episode_ids={agent: np.array([0, 0, 1, 1], dtype=np.int64) for agent in observations},
        timesteps={agent: np.array([0, 1, 0, 1], dtype=np.int64) for agent in observations},
        role_map={"speaker_0": "speaker", "listener_0": "listener"},
    )


def _trajectory(
    env_name: str,
    observations: dict[str, np.ndarray],
    actions: dict[str, np.ndarray],
    rewards: dict[str, np.ndarray],
    episode_ids: dict[str, np.ndarray],
    timesteps: dict[str, np.ndarray],
    role_map: dict[str, str],
) -> dict[str, object]:
    return {
        TrajectoryKey.env_name.value: env_name,
        TrajectoryKey.algorithm.value: "ippo",
        TrajectoryKey.policy_architecture.value: "ff",
        TrajectoryKey.config_id.value: "ippo_ff",
        TrajectoryKey.seed.value: 0,
        TrajectoryKey.parameter_sharing.value: False,
        TrajectoryKey.training_progress_percent.value: 100,
        TrajectoryKey.observations.value: observations,
        TrajectoryKey.actions_raw.value: actions,
        TrajectoryKey.actions_diagnostic.value: actions,
        TrajectoryKey.rewards.value: rewards,
        TrajectoryKey.timesteps.value: timesteps,
        TrajectoryKey.episode_ids.value: episode_ids,
        TrajectoryKey.dones.value: {
            agent: np.array([False, True, False, True]) for agent in observations
        },
        TrajectoryKey.infos.value: {agent: [{}, {}, {}, {}] for agent in observations},
        TrajectoryKey.global_state.value: None,
        TrajectoryKey.hidden_states.value: None,
        TrajectoryKey.agent_role_map.value: role_map,
        TrajectoryKey.action_space_description.value: {},
    }


def _spread_observation(
    self_position: tuple[float, float],
    landmark_relative_positions: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    return np.array(
        [0.0, 0.0, *self_position, *landmark_relative_positions[0], *landmark_relative_positions[1], *landmark_relative_positions[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )


def _listener_observation(
    landmark_relative_positions: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    return np.array(
        [0.0, 0.0, *landmark_relative_positions[0], *landmark_relative_positions[1], *landmark_relative_positions[2], 0.0, 0.0, 0.0],
        dtype=np.float32,
    )


if __name__ == "__main__":
    unittest.main()
