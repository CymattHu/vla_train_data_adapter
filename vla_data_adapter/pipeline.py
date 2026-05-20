"""Pipeline - 串联 source adapter → normalization → exporter 的完整流程。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vla_data_adapter.exporters.base import ModelDatasetAdapter
from vla_data_adapter.normalization import EpisodeNormalizer, NormalizationConfig
from vla_data_adapter.schema import Episode
from vla_data_adapter.sources.base import DataSourceAdapter

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """完整 pipeline 配置。"""
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    validate_before_export: bool = True
    log_statistics: bool = True


class Pipeline:
    """数据转换 Pipeline。

    Source Adapter → Normalization → Quality Check → Model Exporter

    用法：
        pipeline = Pipeline(config)
        pipeline.run(source_adapter, exporter)
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.normalizer = EpisodeNormalizer(self.config.normalization)

    def run(
        self,
        source: DataSourceAdapter,
        exporter: ModelDatasetAdapter,
    ) -> Path:
        """执行完整的数据转换流程。"""
        logger.info(f"Pipeline: loading from {source}")
        episodes = list(source.load())
        logger.info(f"Loaded {len(episodes)} episodes")

        if not episodes:
            raise ValueError("No episodes loaded from source")

        logger.info("Normalizing episodes...")
        episodes = self.normalizer.normalize(episodes)
        logger.info(f"After normalization: {len(episodes)} episodes")

        if self.config.validate_before_export:
            self._validate_episodes(episodes)

        if self.config.log_statistics:
            self._log_statistics(episodes)

        logger.info(f"Exporting to {exporter.__class__.__name__}...")
        output_path = exporter.export(episodes)
        logger.info(f"Export complete: {output_path}")

        return output_path

    def load_and_normalize(self, source: DataSourceAdapter) -> list[Episode]:
        """仅加载和规范化，不导出。适合用于检查数据。"""
        episodes = list(source.load())
        return self.normalizer.normalize(episodes)

    def _validate_episodes(self, episodes: list[Episode]) -> None:
        """验证所有 episodes 的数据完整性。"""
        total_issues = 0
        for ep in episodes:
            issues = ep.validate()
            if issues:
                total_issues += len(issues)
                logger.warning(f"Episode {ep.episode_id}: {issues}")

        if total_issues:
            logger.warning(f"Total validation issues: {total_issues}")
        else:
            logger.info("All episodes passed validation")

    def _log_statistics(self, episodes: list[Episode]) -> None:
        """输出数据集统计信息。"""
        from vla_data_adapter.normalization import DataQualityChecker

        stats = DataQualityChecker.compute_statistics(episodes)
        logger.info(f"Dataset statistics: {stats}")
