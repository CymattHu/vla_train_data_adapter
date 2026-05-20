"""LeRobot 格式导出适配器。

LeRobot 是 Hugging Face 的机器人学习框架。
openpi/pi0 的 fine-tuning 流程基于 LeRobot dataset 格式。
参考: https://github.com/huggingface/lerobot
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vla_data_adapter.schema import Episode, Frame
from .base import ExportConfig, ModelDatasetAdapter

logger = logging.getLogger(__name__)


@dataclass
class LeRobotExportConfig(ExportConfig):
    """LeRobot 导出配置。"""
    repo_id: str = "user/my_dataset"
    fps: int = 10
    video_backend: str = "pyav"  # pyav | opencv
    image_format: str = "png"
    use_videos: bool = False


class LeRobotAdapter(ModelDatasetAdapter):
    """LeRobot 数据集格式导出器。

    LeRobot v2 dataset 结构：
    output_dir/
      meta/
        info.json
        episodes.jsonl
        stats.json
        tasks.jsonl
      data/
        chunk-000/
          episode_000000.parquet
          ...
      videos/ (optional)
        chunk-000/
          observation.images.front_episode_000000.mp4
          ...
    """

    def __init__(self, config: LeRobotExportConfig):
        super().__init__(config)
        self.config: LeRobotExportConfig = config

    def export(self, episodes: list[Episode]) -> Path:
        output_dir = self._ensure_output_dir()

        if self.config.max_episodes:
            episodes = episodes[:self.config.max_episodes]

        logger.info(f"Exporting {len(episodes)} episodes to LeRobot format at {output_dir}")

        self._write_meta(output_dir, episodes)
        self._write_data(output_dir, episodes)

        if self.config.use_videos:
            self._write_videos(output_dir, episodes)

        logger.info("LeRobot export complete")
        return output_dir

    def _write_meta(self, output_dir: Path, episodes: list[Episode]) -> None:
        """写入 meta 目录。"""
        meta_dir = output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        camera_keys = set()
        state_dim = 0
        action_dim = 0

        for ep in episodes:
            for frame in ep.frames:
                camera_keys.update(frame.images.keys())
                if frame.state.joint_pos is not None:
                    state_dim = max(state_dim, len(frame.state.joint_pos))
                if frame.action is not None:
                    action_dim = max(action_dim, len(frame.action.values))
                break
            if camera_keys:
                break

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": None,
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
        }

        for cam_key in sorted(camera_keys):
            if self.config.use_videos:
                features[f"observation.images.{cam_key}"] = {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": None,
                    "video_info": {
                        "video.fps": self.config.fps,
                        "video.codec": "av1",
                        "has_audio": False,
                    },
                }
            else:
                features[f"observation.images.{cam_key}"] = {
                    "dtype": "image",
                    "shape": [480, 640, 3],
                    "names": None,
                    "image_format": self.config.image_format,
                }

        info = {
            "codebase_version": "v2.1",
            "robot_type": episodes[0].robot_type if episodes else "unknown",
            "fps": self.config.fps,
            "features": features,
            "total_episodes": len(episodes),
            "total_frames": sum(ep.num_frames for ep in episodes),
            "repo_id": self.config.repo_id,
        }

        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        tasks: dict[str, int] = {}
        with open(meta_dir / "tasks.jsonl", "w") as f:
            for ep in episodes:
                task = ep.task_instruction
                if task not in tasks:
                    tasks[task] = len(tasks)
                    f.write(json.dumps({"task_index": tasks[task], "task": task}) + "\n")

        with open(meta_dir / "episodes.jsonl", "w") as f:
            for i, ep in enumerate(episodes):
                task_index = tasks.get(ep.task_instruction, 0)
                f.write(json.dumps({
                    "episode_index": i,
                    "tasks": [ep.task_instruction],
                    "task_index": task_index,
                    "length": ep.num_frames,
                }) + "\n")

    def _write_data(self, output_dir: Path, episodes: list[Episode]) -> None:
        """写入 parquet 数据文件。"""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for LeRobot export: pip install pyarrow")

        data_dir = output_dir / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)

        images_dir = output_dir / "images"

        for ep_idx, episode in enumerate(episodes):
            rows = self._episode_to_rows(episode, ep_idx, images_dir)
            if not rows:
                continue

            table = pa.table(rows)
            pq.write_table(table, data_dir / f"episode_{ep_idx:06d}.parquet")

    def _episode_to_rows(
        self, episode: Episode, ep_idx: int, images_dir: Path
    ) -> dict[str, list]:
        """将单个 Episode 转换为 parquet 行数据。"""
        rows: dict[str, list] = {
            "timestamp": [],
            "episode_index": [],
            "frame_index": [],
            "index": [],
            "observation.state": [],
            "action": [],
        }

        camera_keys = set()
        for frame in episode.frames:
            camera_keys.update(frame.images.keys())

        for cam_key in sorted(camera_keys):
            rows[f"observation.images.{cam_key}"] = []

        for frame_idx, frame in enumerate(episode.frames):
            rows["timestamp"].append(frame.timestamp)
            rows["episode_index"].append(ep_idx)
            rows["frame_index"].append(frame_idx)
            rows["index"].append(frame_idx)

            state = self._extract_state_vector(frame)
            rows["observation.state"].append(state)

            action = frame.action.values.tolist() if frame.action else []
            rows["action"].append(action)

            for cam_key in sorted(camera_keys):
                if cam_key in frame.images and not self.config.use_videos:
                    img_rel_path = self._save_image(
                        frame.images[cam_key].data,
                        images_dir,
                        cam_key,
                        ep_idx,
                        frame_idx,
                    )
                    rows[f"observation.images.{cam_key}"].append(str(img_rel_path))
                else:
                    rows[f"observation.images.{cam_key}"].append("")

        return rows

    def _extract_state_vector(self, frame: Frame) -> list[float]:
        """从 Frame 中提取统一的 state 向量。"""
        parts: list[np.ndarray] = []
        if frame.state.joint_pos is not None:
            parts.append(frame.state.joint_pos)
        if frame.state.gripper_state is not None:
            parts.append(frame.state.gripper_state)
        if parts:
            return np.concatenate(parts).tolist()
        return []

    def _save_image(
        self,
        image_data: np.ndarray,
        images_dir: Path,
        camera_name: str,
        ep_idx: int,
        frame_idx: int,
    ) -> Path:
        """保存单张图像并返回相对路径。"""
        from PIL import Image

        cam_dir = images_dir / camera_name / f"episode_{ep_idx:06d}"
        cam_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{frame_idx:06d}.{self.config.image_format}"
        filepath = cam_dir / filename
        Image.fromarray(image_data).save(filepath)

        return filepath.relative_to(images_dir.parent)

    def _write_videos(self, output_dir: Path, episodes: list[Episode]) -> None:
        """将图像序列编码为视频文件。"""
        videos_dir = output_dir / "videos" / "chunk-000"
        videos_dir.mkdir(parents=True, exist_ok=True)

        for ep_idx, episode in enumerate(episodes):
            camera_frames: dict[str, list[np.ndarray]] = {}
            for frame in episode.frames:
                for cam_name, img_obs in frame.images.items():
                    if cam_name not in camera_frames:
                        camera_frames[cam_name] = []
                    camera_frames[cam_name].append(img_obs.data)

            for cam_name, frames_data in camera_frames.items():
                video_path = videos_dir / f"observation.images.{cam_name}_episode_{ep_idx:06d}.mp4"
                self._encode_video(frames_data, video_path)

    def _encode_video(self, frames: list[np.ndarray], output_path: Path) -> None:
        """将帧列表编码为 MP4 视频。"""
        try:
            import av
        except ImportError:
            logger.warning("pyav not installed, skipping video encoding")
            return

        if not frames:
            return

        h, w = frames[0].shape[:2]
        container = av.open(str(output_path), mode="w")
        stream = container.add_stream("h264", rate=self.config.fps)
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

    def validate_export(self, output_dir: Path) -> bool:
        """验证 LeRobot 数据集导出是否正确。"""
        meta_dir = output_dir / "meta"
        if not (meta_dir / "info.json").exists():
            return False
        if not (meta_dir / "episodes.jsonl").exists():
            return False

        data_dir = output_dir / "data" / "chunk-000"
        if not data_dir.exists():
            return False

        parquet_files = list(data_dir.glob("*.parquet"))
        return len(parquet_files) > 0
