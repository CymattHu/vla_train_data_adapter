"""VLA Training Data Adapter - 统一机器人 VLA 训练数据转换框架。

架构：
  DataSourceAdapter (采集) → Canonical Episode (统一) → ModelDatasetAdapter (适配模型)
"""

__version__ = "0.1.0"

from .schema import (
    Action,
    ActionSpace,
    ActionSpaceType,
    Episode,
    EpisodeMetadata,
    Frame,
    ImageObservation,
    NormalizationType,
    SourceType,
    StateObservation,
)
from .pipeline import Pipeline, PipelineConfig

__all__ = [
    "Action",
    "ActionSpace",
    "ActionSpaceType",
    "Episode",
    "EpisodeMetadata",
    "Frame",
    "ImageObservation",
    "NormalizationType",
    "Pipeline",
    "PipelineConfig",
    "SourceType",
    "StateObservation",
]
