from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from pathlib import Path

import numpy as np

from probing_mpe.experiments.artifacts import reloadable_checkpoint_path


DEFAULT_EPISODES = 1
DEFAULT_FPS = 10
CHANNEL_AXIS = 0
GRAYSCALE_DIMENSIONS = 2
COLOR_DIMENSIONS = 3
FLOAT_FRAME_MAX = 1.0
UINT8_FRAME_MAX = 255


class RenderPatchKey(str, Enum):
    evaluation_episodes = "evaluation_episodes"
    render = "render"
    restore_map_location = "restore_map_location"
    sampling_device = "sampling_device"
    train_device = "train_device"
    buffer_device = "buffer_device"


class DirectoryName(str, Enum):
    checkpoint_final = "checkpoint_final"


class TensorKey(str, Enum):
    next = "next"
    done = "done"
    truncated = "truncated"


class VideoSuffix(str, Enum):
    gif = ".gif"
    mp4 = ".mp4"


ReloadExperiment = Callable[[Path, Mapping[str, object]], object]
VideoWriter = Callable[[Sequence[np.ndarray], Path, int], None]


def normalize_frame(frame: object) -> np.ndarray:
    array = _to_numpy(frame)
    if array.ndim == COLOR_DIMENSIONS and array.shape[CHANNEL_AXIS] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim not in (GRAYSCALE_DIMENSIONS, COLOR_DIMENSIONS):
        raise ValueError(f"Unsupported render frame shape: {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, FLOAT_FRAME_MAX) * UINT8_FRAME_MAX
    return array.astype(np.uint8, copy=False)


def render_checkpoint_video(
    checkpoint_path: Path,
    output_path: Path,
    episodes: int = DEFAULT_EPISODES,
    fps: int = DEFAULT_FPS,
    reload_experiment: ReloadExperiment | None = None,
    video_writer: VideoWriter | None = None,
) -> int:
    experiment = _load_experiment(checkpoint_path, episodes, reload_experiment)
    try:
        frames = collect_render_frames(experiment=experiment, episodes=episodes)
        if not frames:
            raise ValueError("No render frames were captured")
        writer = video_writer or write_video
        writer(frames, output_path, fps)
        return len(frames)
    finally:
        _close_experiment_env(experiment)


def collect_render_frames(experiment: object, episodes: int) -> list[np.ndarray]:
    import torch
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    env = getattr(experiment, "test_env")
    policy = getattr(experiment, "policy")
    max_steps = int(getattr(experiment, "max_steps"))
    frames: list[np.ndarray] = []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(episodes):
            tensordict = env.reset()
            for _ in range(max_steps):
                frame = env.render()
                if frame is not None:
                    frames.append(normalize_frame(frame))
                tensordict = policy(tensordict)
                stepped = env.step(tensordict)
                next_tensordict = _get_value(stepped, TensorKey.next.value)
                tensordict = next_tensordict if next_tensordict is not None else stepped
                if _is_done(tensordict):
                    break
    return frames


def write_video(frames: Sequence[np.ndarray], output_path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == VideoSuffix.gif.value:
        imageio.mimsave(output_path, frames, duration=1 / fps)
        return
    if output_path.suffix.lower() == VideoSuffix.mp4.value:
        imageio.mimsave(output_path, frames, fps=fps)
        return
    raise ValueError(f"Unsupported video output suffix: {output_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a deterministic policy rollout from a BenchMARL checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame_count = render_checkpoint_video(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        episodes=args.episodes,
        fps=args.fps,
    )
    print(f"Saved {args.output} with {frame_count} frames")
    return 0


def _load_experiment(
    checkpoint_path: Path,
    episodes: int,
    reload_experiment: ReloadExperiment | None,
) -> object:
    resolved_checkpoint_path = _reloadable_render_checkpoint_path(checkpoint_path)
    experiment_patch: dict[str, object] = {
        RenderPatchKey.evaluation_episodes.value: episodes,
        RenderPatchKey.render.value: True,
        RenderPatchKey.restore_map_location.value: "cpu",
        RenderPatchKey.sampling_device.value: "cpu",
        RenderPatchKey.train_device.value: "cpu",
        RenderPatchKey.buffer_device.value: "cpu",
    }
    if reload_experiment is not None:
        return reload_experiment(resolved_checkpoint_path, experiment_patch)

    from benchmarl.experiment import Experiment

    return Experiment.reload_from_file(
        str(resolved_checkpoint_path),
        experiment_patch=experiment_patch,
    )


def _reloadable_render_checkpoint_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name != DirectoryName.checkpoint_final.value:
        return checkpoint_path
    return reloadable_checkpoint_path(
        run_dir=checkpoint_path.parent.parent,
        checkpoint_path=checkpoint_path,
    )


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


def _get_value(container: object, key: str) -> object | None:
    if isinstance(container, Mapping):
        return container.get(key)
    get = getattr(container, "get", None)
    if callable(get):
        return get(key)
    return None


def _is_done(tensordict: object) -> bool:
    for key in (TensorKey.done.value, TensorKey.truncated.value):
        value = _get_value(tensordict, key)
        if value is not None and _truthy_array(value):
            return True
    return False


def _truthy_array(value: object) -> bool:
    array = _to_numpy(value)
    return bool(np.asarray(array).any())


def _close_experiment_env(experiment: object) -> None:
    env = getattr(experiment, "test_env", None)
    close = getattr(env, "close", None)
    if callable(close):
        close()
