"""MuJoCo 仿真数据生成器。

在 Docker 中无头运行，生成 pick-and-place 的 demonstration 数据，
保存格式与 RealRobotTeleopAdapter 兼容。

用法:
    python -m sim.generate_data --output /data/input --episodes 50
    python -m sim.generate_data --output ./data/input --episodes 20 --task "pick up the red block and place it on the green target"
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sim.envs import PickPlaceEnv
from sim.envs.pick_place import PickPlaceConfig
from sim.policies import ScriptedPickPlacePolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def generate_episodes(
    output_dir: Path,
    num_episodes: int = 50,
    task_instruction: str = "pick up the red block and place it on the green target",
    image_size: tuple[int, int] = (480, 640),
    control_frequency: int = 10,
    max_steps: int = 200,
    camera_names: list[str] | None = None,
    success_only: bool = False,
    seed: int = 42,
):
    """生成仿真 episodes 并保存。"""
    if camera_names is None:
        camera_names = ["front", "wrist"]

    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = PickPlaceConfig(
        image_width=image_size[1],
        image_height=image_size[0],
        control_frequency=control_frequency,
        max_steps=max_steps,
    )
    env = PickPlaceEnv(env_config)
    policy = ScriptedPickPlacePolicy(
        model=env.model,
        data=env.data,
        ik_step_size=0.15,
        ik_damping=1e-3,
    )

    existing_episodes = sorted(
        d.name for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    )
    start_idx = len(existing_episodes)

    generated = 0
    attempted = 0
    np.random.seed(seed)

    logger.info(f"Generating {num_episodes} episodes (starting from idx {start_idx})")
    logger.info(f"Task: {task_instruction}")
    logger.info(f"Cameras: {camera_names}, Size: {image_size}, Freq: {control_frequency}Hz")

    while generated < num_episodes:
        episode_seed = seed + attempted
        attempted += 1

        obs = env.reset(seed=episode_seed)
        policy.reset()

        timestamps: list[float] = []
        joint_positions_list: list[np.ndarray] = []
        actions_list: list[np.ndarray] = []
        images_buffer: dict[str, list[np.ndarray]] = {cam: [] for cam in camera_names}

        done = False
        success = False
        step = 0

        while not done:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            success = info.get("success", False)

            timestamps.append(step / control_frequency)
            joint_positions_list.append(env.get_joint_positions())
            actions_list.append(action[:7])  # 只存关节动作（不含 finger）

            for cam_name in camera_names:
                img = env.render(camera_name=cam_name)
                images_buffer[cam_name].append(img.copy())

            step += 1

        if success_only and not success:
            continue

        ep_idx = start_idx + generated
        ep_id = f"episode_{ep_idx:03d}"
        ep_dir = output_dir / ep_id
        ep_dir.mkdir(parents=True, exist_ok=True)

        np.save(ep_dir / "timestamps.npy", np.array(timestamps, dtype=np.float32))
        np.save(ep_dir / "joint_positions.npy", np.array(joint_positions_list, dtype=np.float32))
        np.save(ep_dir / "actions.npy", np.array(actions_list, dtype=np.float32))

        for cam_name in camera_names:
            cam_dir = ep_dir / "images" / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(images_buffer[cam_name]):
                Image.fromarray(frame).save(cam_dir / f"{i:06d}.png")

        metadata = {
            "episode_id": ep_id,
            "task_instruction": task_instruction,
            "task_name": "pick_place",
            "robot_type": "panda",
            "source_type": "sim",
            "success": success,
            "operator_id": "scripted_policy",
            "control_frequency": control_frequency,
            "num_frames": len(timestamps),
            "duration": timestamps[-1] if timestamps else 0.0,
            "camera_names": camera_names,
            "seed": episode_seed,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        generated += 1
        status = "SUCCESS" if success else "FAIL"
        logger.info(f"  [{generated}/{num_episodes}] {ep_id}: {step} steps, {status}")

    env.close()

    logger.info(f"\nDone! Generated {generated} episodes ({attempted} attempted)")
    logger.info(f"Success rate: {generated}/{attempted} = {generated/max(attempted,1)*100:.1f}%")
    logger.info(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo 仿真数据生成")
    parser.add_argument("--output", "-o", default="/data/input", help="输出目录")
    parser.add_argument("--episodes", "-n", type=int, default=50, help="生成 episode 数量")
    parser.add_argument("--task", "-t", default="pick up the red block and place it on the green target")
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--frequency", type=int, default=10, help="控制频率 (Hz)")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--cameras", nargs="+", default=["front", "wrist"])
    parser.add_argument("--success-only", action="store_true", help="只保留成功 episode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_episodes(
        output_dir=Path(args.output),
        num_episodes=args.episodes,
        task_instruction=args.task,
        image_size=(args.image_height, args.image_width),
        control_frequency=args.frequency,
        max_steps=args.max_steps,
        camera_names=args.cameras,
        success_only=args.success_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
