"""LIBERO 数据集适配器。

LIBERO 是一系列仿真基准任务，使用 robosuite 环境。
参考: https://libero-framework.github.io/
"""

from __future__ import annotations

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
class LiberoConfig(DataSourceConfig):
    """LIBERO 数据集特定配置。"""
    suite_name: str = "libero_spatial"
    image_size: tuple[int, int] = (128, 128)


class LiberoAdapter(DataSourceAdapter):
    """LIBERO 数据集适配器。

    将 LIBERO HDF5 demo 文件转换为 Canonical Episode。
    """

    def __init__(self, config: LiberoConfig):
        super().__init__(config)
        self.config: LiberoConfig = config

    def get_episode_ids(self) -> list[str]:
        data_dir = self.config.data_dir
        if not data_dir.exists():
            return []

        ids = []
        for hdf5_file in sorted(data_dir.glob("**/*.hdf5")):
            ids.append(hdf5_file.stem)
        return ids

    def load(self) -> Iterator[Episode]:
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for LIBERO adapter: pip install h5py")

        data_dir = self.config.data_dir
        hdf5_files = sorted(data_dir.glob("**/*.hdf5"))

        if self.config.max_episodes:
            hdf5_files = hdf5_files[:self.config.max_episodes]

        for hdf5_file in hdf5_files:
            episodes = self._load_from_hdf5(hdf5_file)
            for ep in episodes:
                if self.config.filter_success_only and not ep.metadata.success:
                    continue
                yield ep

    def _load_from_hdf5(self, hdf5_path: Path) -> list[Episode]:
        """从单个 HDF5 文件加载所有 demo。"""
        import h5py

        episodes: list[Episode] = []

        try:
            with h5py.File(hdf5_path, "r") as f:
                env_args = {}
                if "data" in f.attrs:
                    import json
                    env_args = json.loads(f.attrs["data"])

                demos = f.get("data", {})
                for demo_key in sorted(demos.keys()):
                    demo = demos[demo_key]
                    ep = self._parse_demo(
                        demo,
                        episode_id=f"{hdf5_path.stem}/{demo_key}",
                        env_args=env_args,
                    )
                    if ep:
                        episodes.append(ep)
        except OSError as e:
            logger.error(f"Failed to open {hdf5_path}: {e}")

        return episodes

    def _parse_demo(self, demo, episode_id: str, env_args: dict) -> Episode | None:
        """解析单个 demo 为 Episode。"""
        try:
            actions = np.array(demo["actions"])
            states = np.array(demo["obs"]["robot0_joint_pos"])

            agentview = None
            if "obs/agentview_rgb" in demo:
                agentview = np.array(demo["obs"]["agentview_rgb"])
            elif "obs/agentview_image" in demo:
                agentview = np.array(demo["obs"]["agentview_image"])

            eye_in_hand = None
            if "obs/robot0_eye_in_hand_rgb" in demo:
                eye_in_hand = np.array(demo["obs"]["robot0_eye_in_hand_rgb"])
            elif "obs/robot0_eye_in_hand_image" in demo:
                eye_in_hand = np.array(demo["obs"]["robot0_eye_in_hand_image"])

        except KeyError as e:
            logger.warning(f"Missing key in demo {episode_id}: {e}")
            return None

        num_steps = len(actions)
        frequency = 20  # LIBERO default control frequency

        frames: list[Frame] = []
        for i in range(num_steps):
            images: dict[str, ImageObservation] = {}
            if agentview is not None:
                images["agentview"] = ImageObservation(
                    data=agentview[i],
                    camera_name="agentview",
                )
            if eye_in_hand is not None:
                images["wrist"] = ImageObservation(
                    data=eye_in_hand[i],
                    camera_name="wrist",
                )

            frame = Frame(
                timestamp=i / frequency,
                images=images,
                state=StateObservation(joint_pos=states[i]),
                action=Action(
                    values=actions[i],
                    space_type=ActionSpaceType.JOINT_POSITION,
                ),
                done=(i == num_steps - 1),
            )
            frames.append(frame)

        task_instruction = env_args.get("language_instruction", "")
        if not task_instruction:
            task_instruction = env_args.get("bddl_file_name", episode_id)

        return Episode(
            episode_id=episode_id,
            task_instruction=task_instruction,
            robot_type="panda",
            source_type=SourceType.SIM,
            frames=frames,
            metadata=EpisodeMetadata(
                success=True,  # LIBERO demos are successful demonstrations
                task_name=env_args.get("bddl_file_name", ""),
                scene_id=self.config.suite_name,
            ),
        )
