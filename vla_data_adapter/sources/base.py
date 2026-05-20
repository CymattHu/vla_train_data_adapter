"""DataSourceAdapter 基类 - 所有数据源适配器的抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from vla_data_adapter.schema import Episode


@dataclass
class DataSourceConfig:
    """数据源配置基类。"""
    data_dir: Path
    max_episodes: int | None = None
    filter_success_only: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class DataSourceAdapter(ABC):
    """数据源适配器基类。

    每个数据源（真实机器人遥操作、仿真、现有数据集等）
    实现此接口，将数据转换为 Canonical Episode 格式。
    """

    def __init__(self, config: DataSourceConfig):
        self.config = config

    @abstractmethod
    def load(self) -> Iterator[Episode]:
        """加载数据源中的所有 episodes。

        使用 Iterator 避免一次性加载所有数据到内存。
        """
        ...

    @abstractmethod
    def get_episode_ids(self) -> list[str]:
        """返回数据源中所有可用的 episode ID 列表。"""
        ...

    def load_episode(self, episode_id: str) -> Episode | None:
        """加载特定 ID 的 episode，默认实现遍历所有。"""
        for ep in self.load():
            if ep.episode_id == episode_id:
                return ep
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data_dir={self.config.data_dir})"
