"""ModelDatasetAdapter 基类 - 所有模型训练格式导出器的抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vla_data_adapter.schema import Episode


@dataclass
class ExportConfig:
    """导出配置基类。"""
    output_dir: Path
    image_size: tuple[int, int] | None = None
    max_episodes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ModelDatasetAdapter(ABC):
    """模型数据集导出适配器基类。

    将 Canonical Episode 导出为特定训练框架所需的格式。
    """

    def __init__(self, config: ExportConfig):
        self.config = config

    @abstractmethod
    def export(self, episodes: list[Episode]) -> Path:
        """将 episodes 导出为目标格式。

        Args:
            episodes: Canonical Episode 列表

        Returns:
            导出数据的目录路径
        """
        ...

    @abstractmethod
    def validate_export(self, output_dir: Path) -> bool:
        """验证导出结果是否正确。"""
        ...

    def _ensure_output_dir(self) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        return self.config.output_dir
