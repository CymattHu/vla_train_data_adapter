"""Episode 规范化层 - 时间对齐、频率统一、数据清洗。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from vla_data_adapter.schema import Episode, Frame

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    """规范化配置。"""
    target_frequency: float | None = None
    remove_failed_episodes: bool = False
    check_timestamp_monotonic: bool = True
    max_timestamp_gap: float | None = None
    image_resize: tuple[int, int] | None = None
    action_clip: float | None = None


class EpisodeNormalizer:
    """Episode 数据规范化处理器。

    负责：
    - 时间戳对齐和重采样
    - 控制频率统一
    - 检查 action/state/image 同步
    - 数据质量过滤
    """

    def __init__(self, config: NormalizationConfig):
        self.config = config

    def normalize(self, episodes: list[Episode]) -> list[Episode]:
        """对一批 episodes 执行规范化。"""
        result: list[Episode] = []

        for ep in episodes:
            if self.config.remove_failed_episodes and not ep.metadata.success:
                logger.debug(f"Skipping failed episode: {ep.episode_id}")
                continue

            normalized = self._normalize_single(ep)
            if normalized is not None:
                result.append(normalized)

        logger.info(f"Normalized {len(result)}/{len(episodes)} episodes")
        return result

    def _normalize_single(self, episode: Episode) -> Episode | None:
        """规范化单个 episode。"""
        issues = episode.validate()
        if issues:
            logger.warning(f"Episode {episode.episode_id} has issues: {issues}")

        if not episode.frames:
            return None

        frames = episode.frames

        if self.config.check_timestamp_monotonic:
            frames = self._fix_timestamps(frames)

        if self.config.target_frequency:
            frames = self._resample_to_frequency(frames, self.config.target_frequency)

        if self.config.image_resize:
            frames = self._resize_images(frames)

        if self.config.action_clip:
            frames = self._clip_actions(frames)

        episode.frames = frames
        return episode

    def _fix_timestamps(self, frames: list[Frame]) -> list[Frame]:
        """修复非单调递增的时间戳。"""
        fixed: list[Frame] = [frames[0]]
        for i in range(1, len(frames)):
            if frames[i].timestamp > fixed[-1].timestamp:
                fixed.append(frames[i])
            else:
                logger.debug(f"Dropped frame with non-monotonic timestamp at index {i}")
        return fixed

    def _resample_to_frequency(self, frames: list[Frame], target_freq: float) -> list[Frame]:
        """将 episode 重采样到目标频率。

        使用最近邻插值选取最接近的帧。
        """
        if len(frames) < 2:
            return frames

        start_time = frames[0].timestamp
        end_time = frames[-1].timestamp
        duration = end_time - start_time

        if duration <= 0:
            return frames

        num_target_frames = int(duration * target_freq) + 1
        target_times = np.linspace(start_time, end_time, num_target_frames)

        source_times = np.array([f.timestamp for f in frames])
        resampled: list[Frame] = []

        for t in target_times:
            idx = int(np.argmin(np.abs(source_times - t)))
            frame = frames[idx]
            frame.timestamp = float(t)
            resampled.append(frame)

        return resampled

    def _resize_images(self, frames: list[Frame]) -> list[Frame]:
        """调整所有图像到统一尺寸。"""
        target_h, target_w = self.config.image_resize  # type: ignore

        for frame in frames:
            for cam_name, img_obs in frame.images.items():
                h, w = img_obs.data.shape[:2]
                if h != target_h or w != target_w:
                    from PIL import Image
                    pil_img = Image.fromarray(img_obs.data)
                    pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
                    img_obs.data = np.array(pil_img)

        return frames

    def _clip_actions(self, frames: list[Frame]) -> list[Frame]:
        """裁剪过大的动作值。"""
        clip_val = self.config.action_clip
        for frame in frames:
            if frame.action is not None and clip_val is not None:
                frame.action.values = np.clip(frame.action.values, -clip_val, clip_val)
        return frames


class DataQualityChecker:
    """数据质量检查器。"""

    @staticmethod
    def check_sync(episode: Episode, max_gap_ratio: float = 0.1) -> dict[str, bool]:
        """检查 episode 内各模态数据的同步性。"""
        results = {
            "has_images": False,
            "has_states": False,
            "has_actions": False,
            "images_complete": True,
            "states_complete": True,
            "actions_complete": True,
        }

        camera_keys = set()
        for frame in episode.frames:
            camera_keys.update(frame.images.keys())

        num_frames = len(episode.frames)
        image_counts: dict[str, int] = {k: 0 for k in camera_keys}
        state_count = 0
        action_count = 0

        for frame in episode.frames:
            for k in camera_keys:
                if k in frame.images:
                    image_counts[k] += 1
            if frame.state.joint_pos is not None:
                state_count += 1
            if frame.action is not None:
                action_count += 1

        results["has_images"] = bool(camera_keys)
        results["has_states"] = state_count > 0
        results["has_actions"] = action_count > 0

        if camera_keys:
            for k, count in image_counts.items():
                if count < num_frames * (1 - max_gap_ratio):
                    results["images_complete"] = False
                    break

        results["states_complete"] = state_count >= num_frames * (1 - max_gap_ratio)
        results["actions_complete"] = action_count >= num_frames * (1 - max_gap_ratio)

        return results

    @staticmethod
    def compute_statistics(episodes: list[Episode]) -> dict:
        """计算数据集统计信息。"""
        stats = {
            "num_episodes": len(episodes),
            "total_frames": 0,
            "avg_frames_per_episode": 0.0,
            "avg_duration": 0.0,
            "success_rate": 0.0,
            "source_types": {},
            "robot_types": {},
        }

        if not episodes:
            return stats

        total_frames = sum(ep.num_frames for ep in episodes)
        total_duration = sum(ep.duration for ep in episodes)
        success_count = sum(1 for ep in episodes if ep.metadata.success)

        stats["total_frames"] = total_frames
        stats["avg_frames_per_episode"] = total_frames / len(episodes)
        stats["avg_duration"] = total_duration / len(episodes)
        stats["success_rate"] = success_count / len(episodes)

        for ep in episodes:
            src = ep.source_type.value
            stats["source_types"][src] = stats["source_types"].get(src, 0) + 1
            stats["robot_types"][ep.robot_type] = stats["robot_types"].get(ep.robot_type, 0) + 1

        return stats
