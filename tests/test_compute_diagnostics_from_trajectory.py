from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from probing_mpe.evaluation import (
    DiagnosticName,
    compute_diagnostics_for_trajectory,
    flatten_wandb_metrics,
    trajectory_to_user_data,
    write_diagnostic_outputs,
)
from probing_mpe.trajectories import TrajectoryKey


class FakeUserData:
    def __init__(
        self,
        observations: dict[str, np.ndarray],
        actions: dict[str, np.ndarray],
        timesteps: dict[str, np.ndarray],
        episode_ids: dict[str, np.ndarray],
        hidden_states: dict[str, np.ndarray] | None,
        env_name: str,
        alg_name: str,
        seed: int,
        scenario_name: str,
    ) -> None:
        self.observations = observations
        self.actions = actions
        self.timesteps = timesteps
        self.episode_ids = episode_ids
        self.hidden_states = hidden_states
        self.env_name = env_name
        self.alg_name = alg_name
        self.seed = seed
        self.scenario_name = scenario_name


class FakeResult:
    def __init__(self) -> None:
        self.flags = {
            "history_dependence": True,
            "uses_hidden_teammate_info": False,
            "synchronous_coordination": True,
            "temporal_coordination": False,
        }
        self.raw_row = {
            "run_id": "simple_spread_v3/ippo_rnn/seed7",
            "oar_max": 0.11,
            "oarR_max": 0.12,
            "oar_max_null": 0.01,
            "oarR_max_null": 0.02,
            "har_hidden_max": 0.21,
            "harRcond_hidden_max": 0.22,
            "har_hidden_max_null": 0.03,
            "harRcond_hidden_max_null": 0.04,
            "pif_hidden_max": 0.31,
            "pifRcond_hidden_max": 0.32,
            "pif_hidden_max_null": 0.05,
            "pifRcond_hidden_max_null": 0.06,
            "aa_max": 0.41,
            "aaRcond_max": 0.42,
            "aa_max_null": 0.07,
            "aaRcond_max_null": 0.08,
            "dai_hidden_max": 0.51,
            "daiRcond_hidden_max": math.nan,
            "dai_hidden_max_null": 0.09,
            "daiRcond_hidden_max_null": math.nan,
        }
        self.metrics = self.raw_row


class FakeDiagnosticsModule:
    def __init__(self) -> None:
        self.user_data: FakeUserData | None = None
        self.kwargs: dict[str, object] | None = None

    def UserData(
        self,
        observations: dict[str, np.ndarray],
        actions: dict[str, np.ndarray],
        timesteps: dict[str, np.ndarray],
        episode_ids: dict[str, np.ndarray],
        hidden_states: dict[str, np.ndarray] | None,
        env_name: str,
        alg_name: str,
        seed: int,
        scenario_name: str,
    ) -> FakeUserData:
        self.user_data = FakeUserData(
            observations=observations,
            actions=actions,
            timesteps=timesteps,
            episode_ids=episode_ids,
            hidden_states=hidden_states,
            env_name=env_name,
            alg_name=alg_name,
            seed=seed,
            scenario_name=scenario_name,
        )
        return self.user_data

    def compute_diagnostics(self, data: FakeUserData, **kwargs: object) -> FakeResult:
        self.kwargs = kwargs
        return FakeResult()


class ComputeDiagnosticsFromTrajectoryTest(unittest.TestCase):
    def test_trajectory_to_user_data_uses_diagnostic_actions_and_hidden_states(self) -> None:
        backend = FakeDiagnosticsModule()
        trajectory = _trajectory(policy_architecture="rnn")

        user_data = trajectory_to_user_data(trajectory, backend)

        self.assertIs(user_data, backend.user_data)
        self.assertEqual(user_data.env_name, "simple_spread_v3")
        self.assertEqual(user_data.alg_name, "ippo_rnn")
        self.assertEqual(user_data.seed, 7)
        self.assertEqual(user_data.scenario_name, "simple_spread_v3")
        np.testing.assert_array_equal(
            user_data.actions["agent_0"],
            trajectory[TrajectoryKey.actions_diagnostic.value]["agent_0"],
        )
        self.assertIsNotNone(user_data.hidden_states)

    def test_feed_forward_trajectory_omits_hidden_states(self) -> None:
        backend = FakeDiagnosticsModule()
        trajectory = _trajectory(policy_architecture="ff")

        user_data = trajectory_to_user_data(trajectory, backend)

        self.assertIsNone(user_data.hidden_states)

    def test_compute_diagnostics_maps_raw_normalized_and_null_metrics(self) -> None:
        backend = FakeDiagnosticsModule()
        trajectory = _trajectory(policy_architecture="rnn")

        diagnostics, null_diagnostics = compute_diagnostics_for_trajectory(
            trajectory=trajectory,
            diagnostics_module=backend,
            history_k=3,
            cmi_k=25,
            null_reps=5,
            max_samples=8000,
            posterior_alpha=0.5,
            metrics=(DiagnosticName.oar.value, DiagnosticName.har.value),
            min_effect=0.01,
            parallel_workers=1,
            force_continuous_actions=None,
        )

        self.assertEqual(backend.kwargs["history_k"], 3)
        self.assertEqual(backend.kwargs["null_reps"], 5)
        self.assertEqual(backend.kwargs["metrics"], ("oar", "har"))
        self.assertEqual(diagnostics["diagnostics"]["OAR"]["raw"], 0.11)
        self.assertEqual(diagnostics["diagnostics"]["HAR"]["normalized"], 0.22)
        self.assertTrue(diagnostics["flags"]["history_dependence"])
        self.assertEqual(null_diagnostics["null_diagnostics"]["OAR"]["raw_mean"], 0.01)
        self.assertEqual(
            null_diagnostics["null_diagnostics"]["HAR"]["normalized_mean"],
            0.04,
        )
        self.assertEqual(
            diagnostics["undefined_normalized"][0]["metric"],
            "DAInorm",
        )

    def test_write_diagnostic_outputs_writes_json_with_nan(self) -> None:
        backend = FakeDiagnosticsModule()
        diagnostics, null_diagnostics = compute_diagnostics_for_trajectory(
            trajectory=_trajectory(policy_architecture="rnn"),
            diagnostics_module=backend,
            history_k=3,
            cmi_k=25,
            null_reps=5,
            max_samples=8000,
            posterior_alpha=0.5,
            metrics=tuple(name.value for name in DiagnosticName),
            min_effect=0.01,
            parallel_workers=1,
            force_continuous_actions=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostics_path = Path(temp_dir) / "diagnostics.json"
            null_path = Path(temp_dir) / "diagnostics_null.json"
            write_diagnostic_outputs(diagnostics, null_diagnostics, diagnostics_path, null_path)

            text = diagnostics_path.read_text(encoding="utf-8")
            self.assertIn("NaN", text)
            self.assertTrue(null_path.exists())

    def test_flatten_wandb_metrics_prefers_normalized_final_values(self) -> None:
        backend = FakeDiagnosticsModule()
        diagnostics, null_diagnostics = compute_diagnostics_for_trajectory(
            trajectory=_trajectory(policy_architecture="rnn"),
            diagnostics_module=backend,
            history_k=3,
            cmi_k=25,
            null_reps=5,
            max_samples=8000,
            posterior_alpha=0.5,
            metrics=tuple(name.value for name in DiagnosticName),
            min_effect=0.01,
            parallel_workers=1,
            force_continuous_actions=None,
        )

        metrics = flatten_wandb_metrics(diagnostics, null_diagnostics)

        self.assertEqual(metrics["diagnostics_final/OAR"], 0.11)
        self.assertEqual(metrics["diagnostics_final/OARnorm"], 0.12)
        self.assertEqual(metrics["diagnostics_final/OARnorm_null_mean"], 0.02)
        self.assertTrue(metrics["diagnostics_final/HARnorm_above_null"])


def _trajectory(policy_architecture: str) -> dict[str, object]:
    agent_values = {
        "agent_0": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        "agent_1": np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    }
    actions = {
        "agent_0": np.array([0, 1], dtype=np.int64),
        "agent_1": np.array([1, 0], dtype=np.int64),
    }
    timesteps = {
        "agent_0": np.array([0, 1], dtype=np.int64),
        "agent_1": np.array([0, 1], dtype=np.int64),
    }
    episode_ids = {
        "agent_0": np.array([0, 0], dtype=np.int64),
        "agent_1": np.array([0, 0], dtype=np.int64),
    }
    return {
        TrajectoryKey.env_name.value: "simple_spread_v3",
        TrajectoryKey.algorithm.value: "ippo",
        TrajectoryKey.policy_architecture.value: policy_architecture,
        TrajectoryKey.config_id.value: f"ippo_{policy_architecture}",
        TrajectoryKey.seed.value: 7,
        TrajectoryKey.parameter_sharing.value: False,
        TrajectoryKey.training_progress_percent.value: 100,
        TrajectoryKey.observations.value: agent_values,
        TrajectoryKey.actions_raw.value: actions,
        TrajectoryKey.actions_diagnostic.value: actions,
        TrajectoryKey.rewards.value: agent_values,
        TrajectoryKey.timesteps.value: timesteps,
        TrajectoryKey.episode_ids.value: episode_ids,
        TrajectoryKey.dones.value: {
            "agent_0": np.array([False, True]),
            "agent_1": np.array([False, True]),
        },
        TrajectoryKey.infos.value: {"agent_0": [{}, {}], "agent_1": [{}, {}]},
        TrajectoryKey.global_state.value: None,
        TrajectoryKey.hidden_states.value: (
            {
                "agent_0": np.array([[0.0], [1.0]], dtype=np.float32),
                "agent_1": np.array([[1.0], [0.0]], dtype=np.float32),
            }
            if policy_architecture == "rnn"
            else None
        ),
        TrajectoryKey.agent_role_map.value: {
            "agent_0": "agent",
            "agent_1": "agent",
        },
        TrajectoryKey.action_space_description.value: {},
    }


if __name__ == "__main__":
    unittest.main()
