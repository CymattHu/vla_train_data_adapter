"""真实机器人遥操作数据采集脚本模板。

根据你的机器人 SDK 修改 RobotInterface 的实现即可。
采集的数据自动按 RealRobotTeleopAdapter 要求的格式保存。

用法:
    python scripts/collect_teleop.py --output ./data/input --task "pick up the red cup"
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================
# 你需要根据实际硬件修改这个类
# ============================================================

@dataclass
class RobotObservation:
    joint_positions: np.ndarray   # (num_joints,)
    ee_pose: np.ndarray | None    # (6,) or (7,) 末端位姿
    gripper_state: float          # 0.0 ~ 1.0
    images: dict[str, np.ndarray] # camera_name → (H, W, 3) uint8


class RobotInterface:
    """机器人硬件接口 - 替换为你的实际实现。

    示例对接：
    - Franka Panda: 用 frankx / polymetis / deoxys
    - UR5/UR10: 用 ur_rtde
    - AgileX / ALOHA: 用对应 SDK
    - ROS: 用 rospy subscriber
    """

    def __init__(self, robot_ip: str = "192.168.1.100"):
        self.robot_ip = robot_ip
        # TODO: 初始化你的机器人连接
        # self.robot = YourRobotSDK(robot_ip)
        # self.cameras = {
        #     "front": Camera(serial="xxx"),
        #     "wrist": Camera(serial="yyy"),
        # }
        print(f"[DEMO] Robot interface initialized (ip={robot_ip})")
        print("[DEMO] Replace this class with your actual robot SDK!")

    def get_observation(self) -> RobotObservation:
        """获取当前观测。替换为你的实际实现。"""
        # TODO: 从机器人读取实际数据
        # joint_pos = self.robot.get_joint_positions()
        # ee_pose = self.robot.get_ee_pose()
        # gripper = self.robot.get_gripper_state()
        # images = {name: cam.capture() for name, cam in self.cameras.items()}

        # DEMO: 返回随机数据
        return RobotObservation(
            joint_positions=np.random.randn(7).astype(np.float32),
            ee_pose=np.random.randn(7).astype(np.float32),
            gripper_state=0.5,
            images={
                "front": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                "wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            },
        )

    def get_action(self) -> np.ndarray:
        """获取当前动作（来自遥操作设备）。替换为你的实际实现。

        遥操作设备示例：
        - SpaceMouse / 3Dconnexion
        - VR 手柄 (Quest, Vive)
        - 主从臂 (ALOHA style)
        - 键盘 (调试用)
        """
        # TODO: 从遥操作设备读取动作
        # action = self.teleop_device.get_action()

        # DEMO: 返回随机动作
        return np.random.randn(7).astype(np.float32) * 0.01

    def is_episode_done(self) -> bool:
        """判断当前 episode 是否结束。

        可以通过按键、时间、或任务完成条件触发。
        """
        # TODO: 实现你的终止条件
        return False

    def reset(self) -> None:
        """重置机器人到初始位姿。"""
        # TODO: self.robot.move_to_home()
        print("[DEMO] Robot reset to home position")


# ============================================================
# 数据采集器 - 一般不需要修改
# ============================================================

class TeleopCollector:
    """遥操作数据采集器，自动保存为 RealRobotTeleopAdapter 兼容格式。"""

    def __init__(
        self,
        robot: RobotInterface,
        output_dir: Path,
        control_frequency: int = 10,
        robot_type: str = "panda",
    ):
        self.robot = robot
        self.output_dir = output_dir
        self.control_frequency = control_frequency
        self.robot_type = robot_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_episode(
        self,
        task_instruction: str,
        max_steps: int = 500,
        operator_id: str = "default",
    ) -> Path:
        """采集单个 episode。"""
        episode_id = self._next_episode_id()
        ep_dir = self.output_dir / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Recording episode: {episode_id}")
        print(f"Task: {task_instruction}")
        print(f"Frequency: {self.control_frequency} Hz")
        print(f"Max steps: {max_steps}")
        print(f"{'='*50}")
        print("Press Ctrl+C to stop recording\n")

        timestamps: list[float] = []
        joint_positions: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        images_buffer: dict[str, list[np.ndarray]] = {}

        dt = 1.0 / self.control_frequency
        start_time = time.time()

        try:
            for step in range(max_steps):
                step_start = time.time()

                obs = self.robot.get_observation()
                action = self.robot.get_action()

                timestamps.append(time.time() - start_time)
                joint_positions.append(obs.joint_positions)
                actions.append(action)

                for cam_name, img in obs.images.items():
                    if cam_name not in images_buffer:
                        images_buffer[cam_name] = []
                    images_buffer[cam_name].append(img)

                if self.robot.is_episode_done():
                    print(f"\nEpisode done at step {step}")
                    break

                elapsed = time.time() - step_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if step % 50 == 0:
                    print(f"  Step {step}/{max_steps} ({timestamps[-1]:.1f}s)")

        except KeyboardInterrupt:
            print("\nRecording stopped by user")

        success = self._ask_success()

        self._save_episode(
            ep_dir=ep_dir,
            episode_id=episode_id,
            task_instruction=task_instruction,
            timestamps=timestamps,
            joint_positions=joint_positions,
            actions=actions,
            images_buffer=images_buffer,
            success=success,
            operator_id=operator_id,
        )

        print(f"Saved {len(timestamps)} frames to {ep_dir}")
        return ep_dir

    def _save_episode(
        self,
        ep_dir: Path,
        episode_id: str,
        task_instruction: str,
        timestamps: list[float],
        joint_positions: list[np.ndarray],
        actions: list[np.ndarray],
        images_buffer: dict[str, list[np.ndarray]],
        success: bool,
        operator_id: str,
    ) -> None:
        """保存采集数据为标准格式。"""
        np.save(ep_dir / "timestamps.npy", np.array(timestamps))
        np.save(ep_dir / "joint_positions.npy", np.array(joint_positions))
        np.save(ep_dir / "actions.npy", np.array(actions))

        from PIL import Image
        for cam_name, frames in images_buffer.items():
            cam_dir = ep_dir / "images" / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(cam_dir / f"{i:06d}.png")

        metadata = {
            "episode_id": episode_id,
            "task_instruction": task_instruction,
            "task_name": task_instruction,
            "robot_type": self.robot_type,
            "success": success,
            "operator_id": operator_id,
            "control_frequency": self.control_frequency,
            "num_frames": len(timestamps),
            "duration": timestamps[-1] if timestamps else 0.0,
            "camera_names": list(images_buffer.keys()),
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _next_episode_id(self) -> str:
        """自动生成下一个 episode ID。"""
        existing = sorted(
            d.name for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        )
        next_idx = len(existing)
        return f"episode_{next_idx:03d}"

    def _ask_success(self) -> bool:
        """采集结束后询问是否成功。"""
        try:
            answer = input("\nEpisode successful? [y/N]: ").strip().lower()
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="遥操作数据采集")
    parser.add_argument("--output", "-o", default="./data/input", help="保存目录")
    parser.add_argument("--task", "-t", required=True, help="任务指令（自然语言）")
    parser.add_argument("--robot-ip", default="192.168.1.100", help="机器人 IP")
    parser.add_argument("--robot-type", default="panda", help="机器人类型")
    parser.add_argument("--frequency", type=int, default=10, help="采集频率 (Hz)")
    parser.add_argument("--max-steps", type=int, default=500, help="最大步数")
    parser.add_argument("--episodes", "-n", type=int, default=1, help="采集几个 episode")
    parser.add_argument("--operator", default="default", help="操作员 ID")
    args = parser.parse_args()

    robot = RobotInterface(robot_ip=args.robot_ip)
    collector = TeleopCollector(
        robot=robot,
        output_dir=Path(args.output),
        control_frequency=args.frequency,
        robot_type=args.robot_type,
    )

    for i in range(args.episodes):
        print(f"\n--- Episode {i+1}/{args.episodes} ---")
        robot.reset()
        time.sleep(1)
        collector.collect_episode(
            task_instruction=args.task,
            max_steps=args.max_steps,
            operator_id=args.operator,
        )

    print(f"\nDone! Collected {args.episodes} episodes at {args.output}")
    print("Next step: vla-adapter convert --source real_robot --input ./data/input --format lerobot --output ./output")


if __name__ == "__main__":
    main()
