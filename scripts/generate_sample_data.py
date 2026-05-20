"""生成示例数据用于快速测试 pipeline。

用法:
    python scripts/generate_sample_data.py [--output ./data/input] [--episodes 5]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def generate_sample_episodes(output_dir: Path, num_episodes: int = 5, num_frames: int = 50):
    """生成模拟的真实机器人遥操作数据。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in range(num_episodes):
        ep_dir = output_dir / f"episode_{ep_idx:03d}"
        ep_dir.mkdir(exist_ok=True)

        frequency = 10
        num_joints = 7

        timestamps = np.arange(num_frames) / frequency
        np.save(ep_dir / "timestamps.npy", timestamps)

        joint_positions = np.cumsum(np.random.randn(num_frames, num_joints) * 0.01, axis=0)
        np.save(ep_dir / "joint_positions.npy", joint_positions.astype(np.float32))

        actions = np.random.randn(num_frames, num_joints) * 0.1
        np.save(ep_dir / "actions.npy", actions.astype(np.float32))

        images_dir = ep_dir / "images"
        for cam_name in ["front", "wrist"]:
            cam_dir = images_dir / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx in range(num_frames):
                img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                from PIL import Image
                Image.fromarray(img).save(cam_dir / f"{frame_idx:06d}.png")

        metadata = {
            "task_instruction": f"pick up object {ep_idx}",
            "task_name": "pick_place",
            "robot_type": "panda",
            "success": ep_idx % 2 == 0,
            "operator_id": "demo_user",
            "control_frequency": frequency,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"Generated {num_episodes} sample episodes at {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="生成示例数据")
    parser.add_argument("--output", "-o", default="./data/input", help="输出目录")
    parser.add_argument("--episodes", "-n", type=int, default=5, help="episode 数量")
    parser.add_argument("--frames", "-f", type=int, default=50, help="每 episode 帧数")
    args = parser.parse_args()

    generate_sample_episodes(Path(args.output), args.episodes, args.frames)


if __name__ == "__main__":
    main()
