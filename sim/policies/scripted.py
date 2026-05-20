"""脚本化专家策略，用于生成 demonstration 数据。

通过状态机实现确定性的抓取-放置动作序列，
使用 MuJoCo Jacobian IK 求解器计算关节动作。
"""

from __future__ import annotations

from enum import Enum, auto

import mujoco
import numpy as np

from .ik_solver import MujocoIKSolver


class Phase(Enum):
    REACH_ABOVE = auto()
    REACH_OBJECT = auto()
    GRASP = auto()
    LIFT = auto()
    MOVE_TO_TARGET = auto()
    PLACE = auto()
    RELEASE = auto()
    DONE = auto()


class ScriptedPickPlacePolicy:
    """基于状态机的 pick-and-place 专家策略。

    阶段：到物体上方 → 下降 → 夹取 → 抬起 → 移动到目标 → 放下 → 松开

    使用 MuJoCo Jacobian-based IK 求解器将末端目标位置转换为关节角度。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        approach_height: float = 0.15,
        grasp_height: float = 0.03,
        lift_height: float = 0.15,
        pos_threshold: float = 0.02,
        noise_scale: float = 0.002,
        ik_damping: float = 1e-4,
        ik_step_size: float = 0.3,
        ik_iterations: int = 50,
    ):
        self.approach_height = approach_height
        self.grasp_height = grasp_height
        self.lift_height = lift_height
        self.pos_threshold = pos_threshold
        self.noise_scale = noise_scale
        self.phase = Phase.REACH_ABOVE
        self._grasp_counter = 0

        self.ik_solver = MujocoIKSolver(
            model=model,
            data=data,
            body_name="hand",
            damping=ik_damping,
            step_size=ik_step_size,
            max_iterations=ik_iterations,
        )

    def reset(self):
        self.phase = Phase.REACH_ABOVE
        self._grasp_counter = 0

    def get_action(self, obs: dict) -> np.ndarray:
        """根据当前观测和阶段返回动作。

        Returns:
            (8,) array: [7 joint targets, 1 gripper (0=close, 255=open)]
        """
        ee_pos = obs["ee_position"]
        obj_pos = obs["object_position"]
        target_pos = obs["target_position"]
        current_joints = obs["joint_positions"]

        goal_pos = self._get_goal_position(ee_pos, obj_pos, target_pos)
        gripper_open = self._get_gripper_target()

        self._update_phase(ee_pos, obj_pos, target_pos)

        delta = goal_pos - ee_pos
        joint_targets = self.ik_solver.compute_joint_delta(delta, current_joints)

        noise = np.random.randn(7) * self.noise_scale
        joint_targets = joint_targets + noise

        gripper_cmd = gripper_open * 255.0
        action = np.concatenate([joint_targets, [gripper_cmd]])

        return action

    def _get_goal_position(
        self, ee_pos: np.ndarray, obj_pos: np.ndarray, target_pos: np.ndarray
    ) -> np.ndarray:
        """根据阶段计算目标末端位置。"""
        if self.phase == Phase.REACH_ABOVE:
            return np.array([obj_pos[0], obj_pos[1], obj_pos[2] + self.approach_height])
        elif self.phase == Phase.REACH_OBJECT:
            return np.array([obj_pos[0], obj_pos[1], obj_pos[2] + self.grasp_height])
        elif self.phase in (Phase.GRASP, Phase.LIFT):
            return np.array([ee_pos[0], ee_pos[1], obj_pos[2] + self.lift_height])
        elif self.phase == Phase.MOVE_TO_TARGET:
            return np.array([target_pos[0], target_pos[1], obj_pos[2] + self.lift_height])
        elif self.phase in (Phase.PLACE, Phase.RELEASE):
            return np.array([target_pos[0], target_pos[1], target_pos[2] + self.grasp_height])
        else:
            return ee_pos

    def _get_gripper_target(self) -> float:
        """0=闭合, 1=张开"""
        if self.phase in (Phase.REACH_ABOVE, Phase.REACH_OBJECT):
            return 1.0
        elif self.phase in (Phase.GRASP, Phase.LIFT, Phase.MOVE_TO_TARGET, Phase.PLACE):
            return 0.0
        else:
            return 1.0

    def _update_phase(
        self, ee_pos: np.ndarray, obj_pos: np.ndarray, target_pos: np.ndarray
    ):
        """根据当前状态转移阶段。"""
        if self.phase == Phase.REACH_ABOVE:
            goal = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + self.approach_height])
            if np.linalg.norm(ee_pos - goal) < self.pos_threshold:
                self.phase = Phase.REACH_OBJECT

        elif self.phase == Phase.REACH_OBJECT:
            goal = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + self.grasp_height])
            if np.linalg.norm(ee_pos - goal) < self.pos_threshold:
                self.phase = Phase.GRASP
                self._grasp_counter = 0

        elif self.phase == Phase.GRASP:
            self._grasp_counter += 1
            if self._grasp_counter > 10:
                self.phase = Phase.LIFT

        elif self.phase == Phase.LIFT:
            if ee_pos[2] > obj_pos[2] + self.lift_height - self.pos_threshold:
                self.phase = Phase.MOVE_TO_TARGET

        elif self.phase == Phase.MOVE_TO_TARGET:
            goal_xy = np.array([target_pos[0], target_pos[1]])
            if np.linalg.norm(ee_pos[:2] - goal_xy) < self.pos_threshold:
                self.phase = Phase.PLACE

        elif self.phase == Phase.PLACE:
            goal = np.array([target_pos[0], target_pos[1], target_pos[2] + self.grasp_height])
            if np.linalg.norm(ee_pos - goal) < self.pos_threshold:
                self.phase = Phase.RELEASE
                self._grasp_counter = 0

        elif self.phase == Phase.RELEASE:
            self._grasp_counter += 1
            if self._grasp_counter > 10:
                self.phase = Phase.DONE
