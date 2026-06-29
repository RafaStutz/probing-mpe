from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from probing_mpe.rendering import normalize_frame, render_checkpoint_video


class RenderVideoTest(unittest.TestCase):
    def test_normalize_frame_transposes_channel_first_frame(self) -> None:
        frame = np.zeros((3, 4, 5), dtype=np.uint8)
        frame[0, :, :] = 10
        frame[1, :, :] = 20
        frame[2, :, :] = 30

        normalized = normalize_frame(frame)

        self.assertEqual(normalized.shape, (4, 5, 3))
        np.testing.assert_array_equal(normalized[0, 0], np.array([10, 20, 30]))

    def test_render_checkpoint_video_writes_captured_frames(self) -> None:
        written_frames: list[np.ndarray] = []
        written_paths: list[Path] = []
        written_fps: list[int] = []

        def write_video(frames: Sequence[np.ndarray], output_path: Path, fps: int) -> None:
            written_frames.extend(frames)
            written_paths.append(output_path)
            written_fps.append(fps)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "policy.gif"
            frame_count = render_checkpoint_video(
                checkpoint_path=Path(temp_dir) / "checkpoint.pt",
                output_path=output_path,
                episodes=1,
                fps=7,
                reload_experiment=_reload_experiment,
                video_writer=write_video,
            )

        self.assertEqual(frame_count, 2)
        self.assertEqual(len(written_frames), 2)
        self.assertEqual(written_frames[0].shape, (3, 4, 3))
        self.assertEqual(written_paths, [output_path])
        self.assertEqual(written_fps, [7])

    def test_render_checkpoint_video_uses_reloadable_checkpoint_for_normalized_final_checkpoint(
        self,
    ) -> None:
        loaded_checkpoints: list[Path] = []

        def reload_experiment(
            checkpoint_path: Path,
            experiment_patch: Mapping[str, object],
        ) -> object:
            loaded_checkpoints.append(checkpoint_path)
            return _reload_experiment(checkpoint_path, experiment_patch)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            normalized_checkpoint = run_dir / "checkpoint_final" / "checkpoint.pt"
            raw_checkpoint = (
                run_dir / "mappo_experiment" / "checkpoints" / "checkpoint_100.pt"
            )
            older_raw_checkpoint = (
                run_dir / "mappo_experiment" / "checkpoints" / "checkpoint_50.pt"
            )
            normalized_checkpoint.parent.mkdir(parents=True)
            raw_checkpoint.parent.mkdir(parents=True)
            normalized_checkpoint.write_bytes(b"checkpoint")
            raw_checkpoint.write_bytes(b"checkpoint")
            older_raw_checkpoint.write_bytes(b"checkpoint")

            render_checkpoint_video(
                checkpoint_path=normalized_checkpoint,
                output_path=run_dir / "policy.gif",
                reload_experiment=reload_experiment,
                video_writer=_ignore_video,
            )

        self.assertEqual(loaded_checkpoints, [raw_checkpoint])

    def test_render_checkpoint_video_fails_when_environment_returns_no_frames(self) -> None:
        def reload_experiment(
            checkpoint_path: Path,
            experiment_patch: Mapping[str, object],
        ) -> object:
            return FakeExperiment(env=FakeEnv(frames=[]))

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "No render frames were captured"):
                render_checkpoint_video(
                    checkpoint_path=Path(temp_dir) / "checkpoint.pt",
                    output_path=Path(temp_dir) / "policy.gif",
                    reload_experiment=reload_experiment,
                )


class FakeExperiment:
    def __init__(self, env: "FakeEnv") -> None:
        self.test_env = env
        self.policy = FakePolicy()
        self.max_steps = 4


class FakePolicy:
    def __call__(self, tensordict: Mapping[str, object]) -> Mapping[str, object]:
        return tensordict


class FakeEnv:
    def __init__(self, frames: Sequence[np.ndarray]) -> None:
        self._frames = list(frames)
        self._step_count = 0
        self.closed = False

    def reset(self) -> Mapping[str, object]:
        self._step_count = 0
        return {}

    def render(self) -> np.ndarray | None:
        if self._step_count >= len(self._frames):
            return None
        return self._frames[self._step_count]

    def step(self, tensordict: Mapping[str, object]) -> Mapping[str, object]:
        self._step_count += 1
        done = self._step_count >= len(self._frames)
        return {"next": {"done": np.array([done]), "truncated": np.array([False])}}

    def close(self) -> None:
        self.closed = True


def _reload_experiment(
    checkpoint_path: Path,
    experiment_patch: Mapping[str, object],
) -> object:
    expected_patch = {
        "evaluation_episodes": 1,
        "render": True,
        "restore_map_location": "cpu",
        "sampling_device": "cpu",
        "train_device": "cpu",
        "buffer_device": "cpu",
    }
    if experiment_patch != expected_patch:
        raise ValueError(f"Unexpected experiment patch: {experiment_patch}")
    frames = [
        np.zeros((3, 3, 4), dtype=np.uint8),
        np.ones((3, 3, 4), dtype=np.uint8),
    ]
    return FakeExperiment(env=FakeEnv(frames=frames))


def _ignore_video(frames: Sequence[np.ndarray], output_path: Path, fps: int) -> None:
    return None


if __name__ == "__main__":
    unittest.main()
