"""测试 Canonical Episode Schema。"""

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


def make_dummy_episode(num_frames: int = 10, frequency: int = 10) -> Episode:
    """创建用于测试的 dummy episode。"""
    frames = []
    for i in range(num_frames):
        frame = Frame(
            timestamp=i / frequency,
            images={
                "front": ImageObservation(
                    data=np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                    camera_name="front",
                ),
            },
            state=StateObservation(
                joint_pos=np.random.randn(7).astype(np.float32),
                gripper_state=np.array([0.5], dtype=np.float32),
            ),
            action=Action(
                values=np.random.randn(7).astype(np.float32),
                space_type=ActionSpaceType.JOINT_POSITION,
            ),
            done=(i == num_frames - 1),
        )
        frames.append(frame)

    return Episode(
        episode_id="test_ep_001",
        task_instruction="pick up the red block",
        robot_type="panda",
        source_type=SourceType.REAL,
        frames=frames,
        metadata=EpisodeMetadata(
            success=True,
            task_name="pick_place",
        ),
    )


def test_episode_creation():
    ep = make_dummy_episode()
    assert ep.episode_id == "test_ep_001"
    assert ep.num_frames == 10
    assert ep.source_type == SourceType.REAL


def test_episode_duration():
    ep = make_dummy_episode(num_frames=11, frequency=10)
    assert abs(ep.duration - 1.0) < 1e-6


def test_episode_frequency():
    ep = make_dummy_episode(num_frames=11, frequency=10)
    assert abs(ep.frequency - 10.0) < 1e-6


def test_episode_validate_clean():
    ep = make_dummy_episode()
    issues = ep.validate()
    assert len(issues) == 0


def test_episode_validate_empty():
    ep = Episode(
        episode_id="",
        task_instruction="",
        robot_type="test",
        source_type=SourceType.SIM,
    )
    issues = ep.validate()
    assert "episode_id is empty" in issues
    assert "task_instruction is empty" in issues
    assert "episode has no frames" in issues


def test_episode_validate_non_monotonic():
    frames = [
        Frame(timestamp=0.0, state=StateObservation(joint_pos=np.zeros(7))),
        Frame(timestamp=0.2, state=StateObservation(joint_pos=np.zeros(7))),
        Frame(timestamp=0.1, state=StateObservation(joint_pos=np.zeros(7))),
    ]
    ep = Episode(
        episode_id="test",
        task_instruction="test",
        robot_type="test",
        source_type=SourceType.SIM,
        frames=frames,
    )
    issues = ep.validate()
    assert any("not monotonic" in i for i in issues)
