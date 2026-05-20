"""脚本化专家策略，用于生成 demonstration 数据。

通过状态机实现确定性的抓取-放置动作序列，
使用 MuJoCo Jacobian IK 求解器计算关节动作。
"""

from __future__ import annotations

import logging
from enum import Enum, auto

import mujoco
import numpy as np

from .ik_solver import MujocoIKSolver

logger = logging.getLogger(__name__)


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
        approach_height: float = 0.12,
        grasp_height: float = 0.085,
        lift_height: float = 0.18,
        pos_threshold: float = 0.015,
        noise_scale: float = 0.001,
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
        self._phase_step = 0
        self._step = 0

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
        self._phase_step = 0
        self._step = 0
        self._pick_z = None

    def get_action(self, obs: dict) -> np.ndarray:
        """根据当前观测和阶段返回动作。

        Returns:
            (8,) array: [7 joint targets, 1 gripper (0=close, 255=open)]
        """
        self._step += 1
        ee_pos = obs["ee_position"]
        obj_pos = obs["object_position"]
        target_pos = obs["target_position"]
        current_joints = obs["joint_positions"]

        prev_phase = self.phase
        self._update_phase(ee_pos, obj_pos, target_pos)

        if self.phase != prev_phase:
            logger.info(f"  [Step {self._step}] Phase: {prev_phase.name} → {self.phase.name}")

        goal_pos = self._get_goal_position(ee_pos, obj_pos, target_pos)
        gripper_open = self._get_gripper_target()

        delta = goal_pos - ee_pos
        dist = np.linalg.norm(delta)

        if self._step % 30 == 0:
            logger.debug(
                f"  [Step {self._step}] phase={self.phase.name} "
                f"ee={ee_pos} goal={goal_pos} dist={dist:.4f} "
                f"gripper={'OPEN' if gripper_open else 'CLOSE'}"
            )

        if self.phase == Phase.LIFT:
            max_step = 0.003
        elif self.phase in (Phase.REACH_OBJECT, Phase.PLACE):
            max_step = 0.008
        else:
            max_step = 0.015

        if dist > max_step:
            clipped_goal = ee_pos + (goal_pos - ee_pos) * (max_step / dist)
        else:
            clipped_goal = goal_pos

        if self.phase in (Phase.GRASP, Phase.RELEASE):
            alpha = 0.6
        else:
            alpha = 0.5

        joint_targets = self.ik_solver.solve_position(clipped_goal, current_joints)
        joint_targets = current_joints + alpha * (joint_targets - current_joints)

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
        elif self.phase == Phase.GRASP:
            return np.array([obj_pos[0], obj_pos[1], obj_pos[2] + self.grasp_height])
        elif self.phase == Phase.LIFT:
            lift_z = self._pick_z + self.lift_height + self.grasp_height
            return np.array([ee_pos[0], ee_pos[1], lift_z])
        elif self.phase == Phase.MOVE_TO_TARGET:
            lift_z = self._pick_z + self.lift_height + self.grasp_height
            return np.array([target_pos[0], target_pos[1], lift_z])
        elif self.phase == Phase.PLACE:
            return np.array([target_pos[0], target_pos[1], target_pos[2] + self.grasp_height])
        elif self.phase == Phase.RELEASE:
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
        self._phase_step += 1

        if self.phase == Phase.REACH_ABOVE:
            goal = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + self.approach_height])
            dist = np.linalg.norm(ee_pos - goal)
            if dist < self.pos_threshold and self._phase_step >= 5:
                logger.info(f"    REACH_ABOVE done (dist={dist:.4f}, steps={self._phase_step})")
                self.phase = Phase.REACH_OBJECT
                self._phase_step = 0

        elif self.phase == Phase.REACH_OBJECT:
            goal = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + self.grasp_height])
            dist = np.linalg.norm(ee_pos - goal)
            if dist < self.pos_threshold and self._phase_step >= 10:
                logger.info(f"    REACH_OBJECT done (dist={dist:.4f}, steps={self._phase_step})")
                self.phase = Phase.GRASP
                self._grasp_counter = 0
                self._phase_step = 0

        elif self.phase == Phase.GRASP:
            self._grasp_counter += 1
            if self._grasp_counter > 40:
                self._pick_z = obj_pos[2]
                logger.info(f"    GRASP done (held for {self._grasp_counter} steps, pick_z={self._pick_z:.4f})")
                self.phase = Phase.LIFT
                self._phase_step = 0

        elif self.phase == Phase.LIFT:
            lift_target_z = self._pick_z + self.lift_height + self.grasp_height
            if ee_pos[2] > lift_target_z - self.pos_threshold and self._phase_step >= 5:
                logger.info(f"    LIFT done (z={ee_pos[2]:.4f} > {lift_target_z - self.pos_threshold:.4f})")
                self.phase = Phase.MOVE_TO_TARGET
                self._phase_step = 0

        elif self.phase == Phase.MOVE_TO_TARGET:
            goal_xy = np.array([target_pos[0], target_pos[1]])
            dist_xy = np.linalg.norm(ee_pos[:2] - goal_xy)
            if dist_xy < self.pos_threshold and self._phase_step >= 5:
                logger.info(f"    MOVE_TO_TARGET done (dist_xy={dist_xy:.4f})")
                self.phase = Phase.PLACE
                self._phase_step = 0

        elif self.phase == Phase.PLACE:
            goal = np.array([target_pos[0], target_pos[1], target_pos[2] + self.grasp_height])
            dist = np.linalg.norm(ee_pos - goal)
            if dist < self.pos_threshold and self._phase_step >= 10:
                logger.info(f"    PLACE done (dist={dist:.4f})")
                self.phase = Phase.RELEASE
                self._grasp_counter = 0
                self._phase_step = 0

        elif self.phase == Phase.RELEASE:
            self._grasp_counter += 1
            if self._grasp_counter > 20:
                logger.info(f"    RELEASE done")
                self.phase = Phase.DONE
                self._phase_step = 0
