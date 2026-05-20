"""DROID 数据集适配器。

DROID (Distributed Robot Interaction Dataset) 是大规模多机器人交互数据集。
参考: https://droid-dataset.github.io/
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from vla_data_adapter.schema import (
    Action,
    ActionSpaceType,
    Episode,
    EpisodeMetadata,
    Frame,
    ImageObservation,
    SourceType,
    StateObservation,
)
from .base import DataSourceAdapter, DataSourceConfig

logger = logging.getLogger(__name__)


@dataclass
class DroidConfig(DataSourceConfig):
    """DROID 数据集特定配置。"""
    camera_names: list[str] | None = None
    action_type: ActionSpaceType = ActionSpaceType.EE_DELTA_POSE


class DroidAdapter(DataSourceAdapter):
    """DROID 数据集适配器。

    将 DROID 数据格式转换为 Canonical Episode。
    DROID 数据通常以 HDF5 或 tfrecord 格式存储。
    """

    def __init__(self, config: DroidConfig):
        super().__init__(config)
        self.config: DroidConfig = config

    def get_episode_ids(self) -> list[str]:
        data_dir = self.config.data_dir
        if not data_dir.exists():
            logger.warning(f"DROID data directory not found: {data_dir}")
            return []

        ids = []
        for ep_dir in sorted(data_dir.iterdir()):
            if ep_dir.is_dir():
                ids.append(ep_dir.name)
        return ids

    def load(self) -> Iterator[Episode]:
        episode_ids = self.get_episode_ids()
        if self.config.max_episodes:
            episode_ids = episode_ids[:self.config.max_episodes]

        for ep_id in episode_ids:
            ep = self._load_single_episode(ep_id)
            if ep is not None:
                if self.config.filter_success_only and not ep.metadata.success:
                    continue
                yield ep

    def _load_single_episode(self, episode_id: str) -> Episode | None:
        """加载单个 DROID episode。

        TODO: 实际实现需要根据 DROID 的数据格式解析。
        这里提供骨架实现。
        """
        ep_dir = self.config.data_dir / episode_id

        metadata_file = ep_dir / "metadata.json"
        if not metadata_file.exists():
            logger.warning(f"Metadata not found for episode {episode_id}")
            return None

        try:
            with open(metadata_file) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load metadata for {episode_id}: {e}")
            return None

        frames = self._load_frames(ep_dir, meta)

        return Episode(
            episode_id=episode_id,
            task_instruction=meta.get("language_instruction", ""),
            robot_type=meta.get("robot_type", "unknown"),
            source_type=SourceType.DATASET,
            frames=frames,
            metadata=EpisodeMetadata(
                success=meta.get("success", False),
                task_name=meta.get("task_name", ""),
                scene_id=meta.get("scene_id", ""),
                action_space=None,
            ),
        )

    def _load_frames(self, ep_dir: Path, meta: dict) -> list[Frame]:
        """从 episode 目录加载帧数据。"""
        frames: list[Frame] = []

        actions_file = ep_dir / "actions.npy"
        states_file = ep_dir / "states.npy"

        if not actions_file.exists() or not states_file.exists():
            logger.warning(f"Missing data files in {ep_dir}")
            return frames

        actions = np.load(actions_file)
        states = np.load(states_file)
        num_steps = min(len(actions), len(states))

        frequency = meta.get("control_frequency", 10)

        for i in range(num_steps):
            images = self._load_frame_images(ep_dir, i)

            frame = Frame(
                timestamp=i / frequency,
                images=images,
                state=StateObservation(
                    joint_pos=states[i] if states.ndim > 1 else None,
                ),
                action=Action(
                    values=actions[i],
                    space_type=self.config.action_type,
                ),
                done=(i == num_steps - 1),
            )
            frames.append(frame)

        return frames

    def _load_frame_images(self, ep_dir: Path, frame_idx: int) -> dict[str, ImageObservation]:
        """加载单帧的图像数据。"""
        images: dict[str, ImageObservation] = {}
        camera_names = self.config.camera_names or ["exterior_image_1", "exterior_image_2", "wrist_image"]

        for cam_name in camera_names:
            img_path = ep_dir / "images" / cam_name / f"{frame_idx:06d}.png"
            if img_path.exists():
                from PIL import Image
                img = np.array(Image.open(img_path))
                images[cam_name] = ImageObservation(
                    data=img,
                    camera_name=cam_name,
                )

        return images
