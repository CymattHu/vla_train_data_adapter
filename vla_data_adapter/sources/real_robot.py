"""真实机器人遥操作数据适配器。

用于采集真实机器人数据并转换为 Canonical Episode。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
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
class RealRobotConfig(DataSourceConfig):
    """真实机器人数据采集配置。"""
    robot_type: str = "panda"
    camera_names: list[str] = field(default_factory=lambda: ["wrist", "front"])
    action_type: ActionSpaceType = ActionSpaceType.JOINT_POSITION
    control_frequency: int = 10


class RealRobotTeleopAdapter(DataSourceAdapter):
    """真实机器人遥操作数据适配器。

    假设数据按以下结构存储：
    data_dir/
      episode_000/
        metadata.json
        timestamps.npy
        joint_positions.npy
        actions.npy
        images/
          wrist/
            000000.png
            ...
          front/
            000000.png
            ...
    """

    def __init__(self, config: RealRobotConfig):
        super().__init__(config)
        self.config: RealRobotConfig = config

    def get_episode_ids(self) -> list[str]:
        data_dir = self.config.data_dir
        if not data_dir.exists():
            return []
        return sorted(
            d.name for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        )

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
        ep_dir = self.config.data_dir / episode_id

        metadata_file = ep_dir / "metadata.json"
        if not metadata_file.exists():
            logger.warning(f"No metadata for {episode_id}")
            return None

        try:
            with open(metadata_file) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed loading metadata for {episode_id}: {e}")
            return None

        timestamps = np.load(ep_dir / "timestamps.npy") if (ep_dir / "timestamps.npy").exists() else None
        joint_positions = np.load(ep_dir / "joint_positions.npy") if (ep_dir / "joint_positions.npy").exists() else None
        actions = np.load(ep_dir / "actions.npy") if (ep_dir / "actions.npy").exists() else None

        if actions is None:
            logger.warning(f"No action data for {episode_id}")
            return None

        num_steps = len(actions)
        frames: list[Frame] = []

        for i in range(num_steps):
            ts = float(timestamps[i]) if timestamps is not None else i / self.config.control_frequency

            images: dict[str, ImageObservation] = {}
            for cam_name in self.config.camera_names:
                img_path = ep_dir / "images" / cam_name / f"{i:06d}.png"
                if img_path.exists():
                    from PIL import Image
                    img_data = np.array(Image.open(img_path))
                    images[cam_name] = ImageObservation(data=img_data, camera_name=cam_name)

            state = StateObservation(
                joint_pos=joint_positions[i] if joint_positions is not None else None,
            )

            frame = Frame(
                timestamp=ts,
                images=images,
                state=state,
                action=Action(values=actions[i], space_type=self.config.action_type),
                done=(i == num_steps - 1),
            )
            frames.append(frame)

        return Episode(
            episode_id=episode_id,
            task_instruction=meta.get("task_instruction", ""),
            robot_type=self.config.robot_type,
            source_type=SourceType.REAL,
            frames=frames,
            metadata=EpisodeMetadata(
                success=meta.get("success", False),
                task_name=meta.get("task_name", ""),
                operator_id=meta.get("operator_id", ""),
            ),
        )
