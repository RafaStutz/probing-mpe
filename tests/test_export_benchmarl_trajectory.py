import unittest

import numpy as np

from probing_mpe.trajectories import (
    AgentChunk,
    GroupName,
    TrajectoryKey,
    build_agent_role_map,
    merge_agent_chunks,
    validate_trajectory_schema,
)


class ExportBenchmarlTrajectoryTests(unittest.TestCase):
    def test_validate_accepts_feed_forward_without_hidden_states(self) -> None:
        trajectory = _base_trajectory(policy_architecture="ff", hidden_states=None)

        validate_trajectory_schema(trajectory)

    def test_validate_rejects_recurrent_trajectory_without_hidden_states(self) -> None:
        trajectory = _base_trajectory(policy_architecture="rnn", hidden_states=None)

        with self.assertRaisesRegex(ValueError, "hidden_states"):
            validate_trajectory_schema(trajectory)

    def test_merge_agent_chunks_preserves_pre_action_hidden_states(self) -> None:
        chunk = AgentChunk(
            observations={
                "agent_0": np.array([[1.0], [2.0]], dtype=np.float32),
                "agent_1": np.array([[3.0], [4.0]], dtype=np.float32),
            },
            actions_raw={
                "agent_0": np.array([0, 1], dtype=np.int64),
                "agent_1": np.array([1, 0], dtype=np.int64),
            },
            actions_diagnostic={
                "agent_0": np.array([0, 1], dtype=np.int64),
                "agent_1": np.array([1, 0], dtype=np.int64),
            },
            rewards={
                "agent_0": np.array([0.0, 1.0], dtype=np.float32),
                "agent_1": np.array([0.0, 1.0], dtype=np.float32),
            },
            timesteps={
                "agent_0": np.array([0, 1], dtype=np.int64),
                "agent_1": np.array([0, 1], dtype=np.int64),
            },
            episode_ids={
                "agent_0": np.array([7, 7], dtype=np.int64),
                "agent_1": np.array([7, 7], dtype=np.int64),
            },
            dones={
                "agent_0": np.array([False, True]),
                "agent_1": np.array([False, True]),
            },
            infos={"agent_0": [{}, {}], "agent_1": [{}, {}]},
            hidden_states={
                "agent_0": np.array([[10.0], [11.0]], dtype=np.float32),
                "agent_1": np.array([[20.0], [21.0]], dtype=np.float32),
            },
            global_state=np.array([[5.0], [6.0]], dtype=np.float32),
        )

        merged = merge_agent_chunks([chunk], include_hidden_states=True)

        np.testing.assert_array_equal(
            merged.hidden_states["agent_0"], np.array([[10.0], [11.0]])
        )
        np.testing.assert_array_equal(
            merged.hidden_states["agent_1"], np.array([[20.0], [21.0]])
        )

    def test_build_agent_role_map_handles_speaker_listener_groups(self) -> None:
        role_map = build_agent_role_map(
            {
                GroupName.speaker.value: ["speaker_0"],
                GroupName.listener.value: ["listener_0"],
            }
        )

        self.assertEqual(
            role_map,
            {
                "speaker_0": GroupName.speaker.value,
                "listener_0": GroupName.listener.value,
            },
        )


def _base_trajectory(policy_architecture: str, hidden_states: object) -> dict[str, object]:
    agent_values = {
        "agent_0": np.array([0, 1], dtype=np.int64),
        "agent_1": np.array([1, 0], dtype=np.int64),
    }
    return {
        TrajectoryKey.env_name.value: "simple_spread_v3",
        TrajectoryKey.algorithm.value: "ippo",
        TrajectoryKey.policy_architecture.value: policy_architecture,
        TrajectoryKey.config_id.value: f"ippo_{policy_architecture}",
        TrajectoryKey.seed.value: 0,
        TrajectoryKey.parameter_sharing.value: False,
        TrajectoryKey.training_progress_percent.value: 100,
        TrajectoryKey.observations.value: {
            "agent_0": np.zeros((2, 3), dtype=np.float32),
            "agent_1": np.zeros((2, 3), dtype=np.float32),
        },
        TrajectoryKey.actions_raw.value: agent_values,
        TrajectoryKey.actions_diagnostic.value: agent_values,
        TrajectoryKey.rewards.value: {
            "agent_0": np.zeros(2, dtype=np.float32),
            "agent_1": np.zeros(2, dtype=np.float32),
        },
        TrajectoryKey.timesteps.value: {
            "agent_0": np.array([0, 1], dtype=np.int64),
            "agent_1": np.array([0, 1], dtype=np.int64),
        },
        TrajectoryKey.episode_ids.value: {
            "agent_0": np.array([0, 0], dtype=np.int64),
            "agent_1": np.array([0, 0], dtype=np.int64),
        },
        TrajectoryKey.dones.value: {
            "agent_0": np.array([False, True]),
            "agent_1": np.array([False, True]),
        },
        TrajectoryKey.infos.value: {"agent_0": [{}, {}], "agent_1": [{}, {}]},
        TrajectoryKey.global_state.value: np.zeros((2, 4), dtype=np.float32),
        TrajectoryKey.hidden_states.value: hidden_states,
        TrajectoryKey.agent_role_map.value: {
            "agent_0": GroupName.agent.value,
            "agent_1": GroupName.agent.value,
        },
        TrajectoryKey.action_space_description.value: {
            "agent_0": {"type": "categorical_discrete"},
            "agent_1": {"type": "categorical_discrete"},
        },
    }


if __name__ == "__main__":
    unittest.main()
