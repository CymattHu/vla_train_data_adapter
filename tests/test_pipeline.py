"""测试 Pipeline 端到端流程。"""

import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np

from tests.test_schema import make_dummy_episode
from vla_data_adapter.exporters import LeRobotAdapter, LeRobotExportConfig
from vla_data_adapter.pipeline import Pipeline, PipelineConfig
from vla_data_adapter.schema import Episode
from vla_data_adapter.sources.base import DataSourceAdapter, DataSourceConfig


class DummySource(DataSourceAdapter):
    """用于测试的 dummy 数据源。"""

    def __init__(self, num_episodes: int = 3):
        super().__init__(DataSourceConfig(data_dir=Path("/tmp")))
        self._num_episodes = num_episodes

    def get_episode_ids(self) -> list[str]:
        return [f"ep_{i:03d}" for i in range(self._num_episodes)]

    def load(self) -> Iterator[Episode]:
        for i in range(self._num_episodes):
            ep = make_dummy_episode(num_frames=10)
            ep.episode_id = f"ep_{i:03d}"
            yield ep


def test_pipeline_end_to_end():
    """测试完整 pipeline: source → normalize → export。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"

        source = DummySource(num_episodes=3)
        exporter = LeRobotAdapter(LeRobotExportConfig(
            output_dir=output_dir,
            repo_id="test/test_dataset",
            fps=10,
        ))

        pipeline = Pipeline(PipelineConfig())
        result_path = pipeline.run(source, exporter)

        assert result_path.exists()
        assert (result_path / "meta" / "info.json").exists()
        assert (result_path / "meta" / "episodes.jsonl").exists()
        assert exporter.validate_export(result_path)
