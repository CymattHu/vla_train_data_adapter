"""Canonical Episode Schema - 统一的机器人 episode 数据格式。

所有 DataSourceAdapter 输出此格式，所有 ModelDatasetAdapter 从此格式导出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class SourceType(str, Enum):
    REAL = "real"
    SIM = "sim"
    DATASET = "dataset"
    VIDEO = "video"


class ActionSpaceType(str, Enum):
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    EE_DELTA_POSE = "ee_delta_pose"
    EE_ABSOLUTE_POSE = "ee_absolute_pose"


class NormalizationType(str, Enum):
    MEAN_STD = "mean_std"
    MINMAX = "minmax"
    NONE = "none"


@dataclass
class ActionSpace:
    type: ActionSpaceType
    dim: int
    horizon: int = 1
    frequency: int = 10
    normalization: NormalizationType = NormalizationType.NONE


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray | None = None


@dataclass
class ImageObservation:
    """单帧中某个相机的图像观测。"""
    data: np.ndarray  # (H, W, C) uint8
    camera_name: str
    intrinsics: CameraIntrinsics | None = None
    extrinsics: np.ndarray | None = None  # 4x4 pose matrix


@dataclass
class StateObservation:
    """本体感觉状态。"""
    joint_pos: np.ndarray | None = None
    joint_vel: np.ndarray | None = None
    ee_pose: np.ndarray | None = None  # 6D or 7D (pos + quat)
    gripper_state: np.ndarray | None = None
    force_torque: np.ndarray | None = None


@dataclass
class Action:
    """单步动作。"""
    values: np.ndarray  # 原始动作向量
    space_type: ActionSpaceType = ActionSpaceType.JOINT_POSITION


@dataclass
class Frame:
    """Episode 中的单个时间步。"""
    timestamp: float
    images: dict[str, ImageObservation] = field(default_factory=dict)
    state: StateObservation = field(default_factory=StateObservation)
    action: Action | None = None
    reward: float = 0.0
    done: bool = False


@dataclass
class EpisodeMetadata:
    """Episode 级别的元数据。"""
    success: bool = False
    task_name: str = ""
    scene_id: str = ""
    operator_id: str = ""
    calibration: dict[str, Any] = field(default_factory=dict)
    camera_intrinsics: dict[str, CameraIntrinsics] = field(default_factory=dict)
    action_space: ActionSpace | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """Canonical Episode - 核心数据单元。

    所有数据源通过 adapter 转换成此格式，
    所有训练框架通过 exporter 从此格式导出。
    """
    episode_id: str
    task_instruction: str
    robot_type: str
    source_type: SourceType
    frames: list[Frame] = field(default_factory=list)
    metadata: EpisodeMetadata = field(default_factory=EpisodeMetadata)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp

    @property
    def frequency(self) -> float:
        if self.duration <= 0:
            return 0.0
        return (self.num_frames - 1) / self.duration

    def validate(self) -> list[str]:
        """校验 episode 数据完整性，返回问题列表。"""
        issues: list[str] = []
        if not self.episode_id:
            issues.append("episode_id is empty")
        if not self.task_instruction:
            issues.append("task_instruction is empty")
        if not self.frames:
            issues.append("episode has no frames")
            return issues

        prev_ts = -float("inf")
        for i, frame in enumerate(self.frames):
            if frame.timestamp < prev_ts:
                issues.append(f"frame {i}: timestamp not monotonic")
            prev_ts = frame.timestamp
            if not frame.images and frame.state.joint_pos is None:
                issues.append(f"frame {i}: no observation data")
        return issues
