"""OpenPI (pi0) 格式导出适配器。

OpenPI fine-tuning 的标准路径是：
  Canonical Episode → LeRobot dataset → openpi data config → fine-tune

此适配器可以直接导出 LeRobot 格式（委托给 LeRobotAdapter），
并额外生成 openpi 所需的 data config 和 norm stats。
参考: https://github.com/Physical-Intelligence/openpi
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vla_data_adapter.schema import Episode
from .base import ExportConfig, ModelDatasetAdapter
from .lerobot import LeRobotAdapter, LeRobotExportConfig

logger = logging.getLogger(__name__)


@dataclass
class OpenPIExportConfig(ExportConfig):
    """OpenPI 导出配置。"""
    repo_id: str = "user/my_dataset"
    action_dim: int = 7
    action_horizon: int = 50
    state_dim: int = 7
    fps: int = 10
    delta_action: bool = True
    norm_stats: dict[str, Any] = field(default_factory=dict)


class OpenPIAdapter(ModelDatasetAdapter):
    """OpenPI/pi0 训练格式导出器。

    输出结构：
    output_dir/
      lerobot_dataset/     (LeRobot 格式数据)
      openpi_config.json   (openpi data mapping 配置)
      norm_stats.json      (归一化统计)
    """

    def __init__(self, config: OpenPIExportConfig):
        super().__init__(config)
        self.config: OpenPIExportConfig = config

    def export(self, episodes: list[Episode]) -> Path:
        output_dir = self._ensure_output_dir()

        if self.config.max_episodes:
            episodes = episodes[:self.config.max_episodes]

        logger.info(f"Exporting {len(episodes)} episodes for OpenPI at {output_dir}")

        lerobot_dir = output_dir / "lerobot_dataset"
        lerobot_config = LeRobotExportConfig(
            output_dir=lerobot_dir,
            repo_id=self.config.repo_id,
            fps=self.config.fps,
        )
        lerobot_adapter = LeRobotAdapter(lerobot_config)
        lerobot_adapter.export(episodes)

        norm_stats = self._compute_norm_stats(episodes)
        with open(output_dir / "norm_stats.json", "w") as f:
            json.dump(norm_stats, f, indent=2)

        openpi_config = self._generate_openpi_config(episodes, norm_stats)
        with open(output_dir / "openpi_config.json", "w") as f:
            json.dump(openpi_config, f, indent=2)

        logger.info("OpenPI export complete")
        return output_dir

    def _compute_norm_stats(self, episodes: list[Episode]) -> dict[str, Any]:
        """计算动作和状态的归一化统计量。"""
        all_actions: list[np.ndarray] = []
        all_states: list[np.ndarray] = []

        for ep in episodes:
            for frame in ep.frames:
                if frame.action is not None:
                    all_actions.append(frame.action.values)
                if frame.state.joint_pos is not None:
                    all_states.append(frame.state.joint_pos)

        stats: dict[str, Any] = {}

        if all_actions:
            actions_arr = np.array(all_actions)
            stats["action"] = {
                "mean": actions_arr.mean(axis=0).tolist(),
                "std": actions_arr.std(axis=0).tolist(),
                "min": actions_arr.min(axis=0).tolist(),
                "max": actions_arr.max(axis=0).tolist(),
            }

        if all_states:
            states_arr = np.array(all_states)
            stats["state"] = {
                "mean": states_arr.mean(axis=0).tolist(),
                "std": states_arr.std(axis=0).tolist(),
                "min": states_arr.min(axis=0).tolist(),
                "max": states_arr.max(axis=0).tolist(),
            }

        return stats

    def _generate_openpi_config(
        self, episodes: list[Episode], norm_stats: dict[str, Any]
    ) -> dict[str, Any]:
        """生成 openpi 训练所需的数据映射配置。"""
        camera_keys = set()
        for ep in episodes:
            for frame in ep.frames:
                camera_keys.update(frame.images.keys())
                break
            if camera_keys:
                break

        config = {
            "dataset": {
                "repo_id": self.config.repo_id,
                "type": "lerobot",
            },
            "model": {
                "action_dim": self.config.action_dim,
                "action_horizon": self.config.action_horizon,
                "state_dim": self.config.state_dim,
            },
            "data_mapping": {
                "state_key": "observation.state",
                "action_key": "action",
                "image_keys": [f"observation.images.{k}" for k in sorted(camera_keys)],
            },
            "normalization": {
                "type": "mean_std" if norm_stats else "none",
                "stats": norm_stats,
            },
            "training": {
                "fps": self.config.fps,
                "delta_action": self.config.delta_action,
            },
        }

        return config

    def validate_export(self, output_dir: Path) -> bool:
        lerobot_dir = output_dir / "lerobot_dataset"
        if not lerobot_dir.exists():
            return False
        if not (output_dir / "openpi_config.json").exists():
            return False
        if not (output_dir / "norm_stats.json").exists():
            return False
        return True
