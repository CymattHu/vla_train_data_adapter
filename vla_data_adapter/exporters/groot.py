"""GR00T 格式导出适配器。

NVIDIA GR00T 用于 humanoid / embodied AI 训练。
参考: https://developer.nvidia.com/isaac/groot
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vla_data_adapter.schema import Episode
from .base import ExportConfig, ModelDatasetAdapter

logger = logging.getLogger(__name__)


@dataclass
class GR00TExportConfig(ExportConfig):
    """GR00T 导出配置。"""
    embodiment: str = "franka"
    modality_keys: list[str] | None = None
    video_format: str = "mp4"


class GR00TAdapter(ModelDatasetAdapter):
    """GR00T 训练格式导出器。

    输出结构：
    output_dir/
      dataset_config.json
      episodes/
        episode_000/
          metadata.json
          video.mp4
          actions.npy
          states.npy
        ...
    """

    def __init__(self, config: GR00TExportConfig):
        super().__init__(config)
        self.config: GR00TExportConfig = config

    def export(self, episodes: list[Episode]) -> Path:
        output_dir = self._ensure_output_dir()

        if self.config.max_episodes:
            episodes = episodes[:self.config.max_episodes]

        logger.info(f"Exporting {len(episodes)} episodes for GR00T at {output_dir}")

        episodes_dir = output_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        for i, episode in enumerate(episodes):
            self._export_single_episode(episode, episodes_dir / f"episode_{i:03d}")

        dataset_config = self._generate_dataset_config(episodes)
        with open(output_dir / "dataset_config.json", "w") as f:
            json.dump(dataset_config, f, indent=2)

        logger.info("GR00T export complete")
        return output_dir

    def _export_single_episode(self, episode: Episode, ep_dir: Path) -> None:
        """导出单个 episode 到 GR00T 格式。"""
        ep_dir.mkdir(parents=True, exist_ok=True)

        actions = []
        states = []
        for frame in episode.frames:
            if frame.action is not None:
                actions.append(frame.action.values)
            if frame.state.joint_pos is not None:
                states.append(frame.state.joint_pos)

        if actions:
            np.save(ep_dir / "actions.npy", np.array(actions))
        if states:
            np.save(ep_dir / "states.npy", np.array(states))

        camera_keys = set()
        for frame in episode.frames:
            camera_keys.update(frame.images.keys())

        for cam_name in sorted(camera_keys):
            frames_data = [
                frame.images[cam_name].data
                for frame in episode.frames
                if cam_name in frame.images
            ]
            if frames_data:
                self._save_video(frames_data, ep_dir / f"{cam_name}.{self.config.video_format}")

        metadata = {
            "episode_id": episode.episode_id,
            "task_instruction": episode.task_instruction,
            "robot_type": episode.robot_type,
            "num_frames": episode.num_frames,
            "success": episode.metadata.success,
            "frequency": episode.frequency,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_video(self, frames: list[np.ndarray], output_path: Path) -> None:
        """保存帧序列为视频。"""
        try:
            import av
        except ImportError:
            logger.warning("pyav not installed, saving frames as npy instead")
            npy_path = output_path.with_suffix(".npy")
            np.save(npy_path, np.array(frames))
            return

        if not frames:
            return

        h, w = frames[0].shape[:2]
        container = av.open(str(output_path), mode="w")
        stream = container.add_stream("h264", rate=10)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"

        for frame_data in frames:
            av_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()

    def _generate_dataset_config(self, episodes: list[Episode]) -> dict:
        """生成 GR00T 数据集配置。"""
        camera_keys = set()
        action_dim = 0
        state_dim = 0

        for ep in episodes:
            for frame in ep.frames:
                camera_keys.update(frame.images.keys())
                if frame.action is not None:
                    action_dim = max(action_dim, len(frame.action.values))
                if frame.state.joint_pos is not None:
                    state_dim = max(state_dim, len(frame.state.joint_pos))
            break

        return {
            "embodiment": self.config.embodiment,
            "num_episodes": len(episodes),
            "total_frames": sum(ep.num_frames for ep in episodes),
            "modalities": {
                "video": sorted(camera_keys),
                "action": {"dim": action_dim},
                "state": {"dim": state_dim},
            },
            "task_instructions": list({ep.task_instruction for ep in episodes}),
        }

    def validate_export(self, output_dir: Path) -> bool:
        if not (output_dir / "dataset_config.json").exists():
            return False
        episodes_dir = output_dir / "episodes"
        if not episodes_dir.exists():
            return False
        return any(episodes_dir.iterdir())
