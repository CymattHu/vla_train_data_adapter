"""可视化 MuJoCo 仿真 — 弹出交互式 3D 窗口观看机器人执行任务。

需要本地安装 mujoco（不能在 Docker 中运行），需要显示器。

用法:
    # 实时观看仿真执行（带 MuJoCo viewer 窗口）
    python sim/visualize.py

    # 回放已生成的 episode 图像
    python sim/visualize.py --replay data/input/episode_000
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def run_live_simulation(max_steps: int = 400, speed: float = 1.0):
    """启动 MuJoCo viewer 实时观看仿真。"""
    from sim.envs import PickPlaceEnv
    from sim.envs.pick_place import PickPlaceConfig
    from sim.policies import ScriptedPickPlacePolicy

    config = PickPlaceConfig(max_steps=max_steps)
    env = PickPlaceEnv(config)
    policy = ScriptedPickPlacePolicy(
        model=env.model,
        data=env.data,
        ik_step_size=0.5,
        ik_damping=1e-4,
    )

    obs = env.reset(seed=42)
    policy.reset()

    dt = 1.0 / config.control_frequency

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("MuJoCo Viewer 已启动")
        print("  - 左键拖拽: 旋转视角")
        print("  - 右键拖拽: 平移")
        print("  - 滚轮: 缩放")
        print("  - 关闭窗口退出")
        print(f"\n任务: pick up the red block and place it on the green target")
        print(f"最大步数: {max_steps}")
        print("")

        step = 0
        done = False

        while viewer.is_running() and not done:
            step_start = time.time()

            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)

            viewer.sync()
            step += 1

            task_done = policy.phase.name == "DONE"
            if task_done:
                done = True

            elapsed = time.time() - step_start
            sleep_time = dt / speed - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        status = "SUCCESS" if info.get("success") else ("DONE" if policy.phase.name == "DONE" else "TIMEOUT")
        print(f"\n仿真结束: {step} steps, {status}")
        print(f"最终阶段: {policy.phase.name}")
        print(f"物体位置: {obs['object_position']}")
        print(f"目标位置: {obs['target_position']}")



def replay_episode(episode_dir: Path, fps: int = 10):
    """用 OpenCV 窗口回放已生成 episode 的图像序列。"""
    try:
        import cv2
    except ImportError:
        print("回放需要 opencv: pip install opencv-python")
        print("尝试使用 matplotlib 替代...")
        _replay_matplotlib(episode_dir, fps)
        return

    front_dir = episode_dir / "images" / "front"
    wrist_dir = episode_dir / "images" / "wrist"

    if not front_dir.exists():
        print(f"找不到图像目录: {front_dir}")
        return

    front_images = sorted(front_dir.glob("*.png"))
    print(f"回放 {episode_dir.name}: {len(front_images)} 帧, {fps} FPS")
    print("按 'q' 退出, 空格暂停")

    paused = False
    for i, img_path in enumerate(front_images):
        front_img = cv2.imread(str(img_path))

        wrist_path = wrist_dir / img_path.name
        if wrist_path.exists():
            wrist_img = cv2.imread(str(wrist_path))
            wrist_img = cv2.resize(wrist_img, (front_img.shape[1] // 3, front_img.shape[0] // 3))
            h, w = wrist_img.shape[:2]
            front_img[10:10+h, front_img.shape[1]-w-10:front_img.shape[1]-10] = wrist_img

        cv2.putText(front_img, f"Frame {i}/{len(front_images)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(f"Episode: {episode_dir.name}", front_img)

        while True:
            key = cv2.waitKey(1 if paused else int(1000 / fps))
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused
            elif not paused:
                break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _replay_matplotlib(episode_dir: Path, fps: int = 10):
    """用 matplotlib 回放（不需要 OpenCV）。"""
    import matplotlib.pyplot as plt
    from PIL import Image

    front_dir = episode_dir / "images" / "front"
    if not front_dir.exists():
        print(f"找不到图像目录: {front_dir}")
        return

    front_images = sorted(front_dir.glob("*.png"))
    print(f"回放 {episode_dir.name}: {len(front_images)} 帧")

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.ion()

    for i, img_path in enumerate(front_images):
        img = Image.open(img_path)
        ax.clear()
        ax.imshow(img)
        ax.set_title(f"{episode_dir.name} - Frame {i}/{len(front_images)}")
        ax.axis("off")
        plt.pause(1.0 / fps)

    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="MuJoCo 仿真可视化")
    parser.add_argument("--replay", type=str, default=None,
                        help="回放已有 episode 的路径 (如 data/input/episode_000)")
    parser.add_argument("--max-steps", type=int, default=300, help="实时仿真最大步数")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度倍率")
    parser.add_argument("--fps", type=int, default=10, help="回放帧率")
    args = parser.parse_args()

    if args.replay:
        replay_episode(Path(args.replay), fps=args.fps)
    else:
        run_live_simulation(max_steps=args.max_steps, speed=args.speed)


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
