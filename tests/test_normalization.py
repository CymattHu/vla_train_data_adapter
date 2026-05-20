"""测试 Episode 规范化。"""

import numpy as np

from tests.test_schema import make_dummy_episode
from vla_data_adapter.normalization import EpisodeNormalizer, NormalizationConfig


def test_normalize_passthrough():
    """无规范化配置时应原样通过。"""
    config = NormalizationConfig()
    normalizer = EpisodeNormalizer(config)

    ep = make_dummy_episode(num_frames=10)
    result = normalizer.normalize([ep])
    assert len(result) == 1
    assert result[0].num_frames == 10


def test_normalize_resample():
    """测试频率重采样。"""
    config = NormalizationConfig(target_frequency=5.0)
    normalizer = EpisodeNormalizer(config)

    ep = make_dummy_episode(num_frames=11, frequency=10)
    result = normalizer.normalize([ep])
    assert len(result) == 1
    assert result[0].num_frames == 6  # 1s at 5Hz = 6 frames (inclusive)


def test_normalize_remove_failed():
    """测试过滤失败 episode。"""
    config = NormalizationConfig(remove_failed_episodes=True)
    normalizer = EpisodeNormalizer(config)

    ep_success = make_dummy_episode()
    ep_success.metadata.success = True

    ep_fail = make_dummy_episode()
    ep_fail.episode_id = "fail_ep"
    ep_fail.metadata.success = False

    result = normalizer.normalize([ep_success, ep_fail])
    assert len(result) == 1
    assert result[0].episode_id == "test_ep_001"


def test_normalize_action_clip():
    """测试动作裁剪。"""
    config = NormalizationConfig(action_clip=1.0)
    normalizer = EpisodeNormalizer(config)

    ep = make_dummy_episode()
    ep.frames[0].action.values = np.array([5.0, -5.0, 0.5, 0.0, 0.0, 0.0, 0.0])

    result = normalizer.normalize([ep])
    action_vals = result[0].frames[0].action.values
    assert np.all(action_vals <= 1.0)
    assert np.all(action_vals >= -1.0)
