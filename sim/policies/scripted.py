"""脚本化专家策略，用于生成 demonstration 数据。

通过状态机实现确定性的抓取-放置动作序列，
使用 MuJoCo Jacobian IK 求解器计算关节动作。
纯物理摩擦抓取，无 grasp-lock。
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
    """基于状态机的 pick-and-place 专家策略（纯摩擦抓取）。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        approach_height: float = 0.12,
        grasp_height: float = 0.095,
        lift_height: float = 0.04,
        pos_threshold: float = 0.015,
        noise_scale: float = 0.0005,
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
        self._pick_z = None

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

    HOME_QPOS = np.array([0, -0.785, 0, -2.356, 0, 1.571, -0.785])

    def get_action(self, obs: dict) -> np.ndarray:
        """返回 (8,) 动作: [7 joint targets, 1 gripper (0=close, 255=open)]"""
        self._step += 1
        ee_pos = obs["ee_position"]
        obj_pos = obs["object_position"]
        target_pos = obs["target_position"]
        current_joints = obs["joint_positions"]

        prev_phase = self.phase
        self._update_phase(ee_pos, obj_pos, target_pos)

        if self.phase != prev_phase:
            logger.info(f"  [Step {self._step}] Phase: {prev_phase.name} -> {self.phase.name}")

        gripper_open = self._get_gripper_target()

        if self.phase == Phase.LIFT:
            joint_targets = self.HOME_QPOS.copy()
        elif self.phase == Phase.MOVE_TO_TARGET:
            target_qpos = self.HOME_QPOS.copy()
            j1_angle = np.arctan2(target_pos[1], target_pos[0])
            target_qpos[0] = j1_angle
            joint_targets = target_qpos
        elif self.phase == Phase.PLACE:
            goal_pos = self._get_goal_position(ee_pos, obj_pos, target_pos)
            delta = goal_pos - ee_pos
            dist = np.linalg.norm(delta)
            max_step = self._get_max_cart_step()
            if dist > max_step:
                clipped_goal = ee_pos + delta * (max_step / dist)
            else:
                clipped_goal = goal_pos
            alpha = 0.3
            joint_targets = self.ik_solver.solve_position(clipped_goal, current_joints)
            joint_targets = current_joints + alpha * (joint_targets - current_joints)
        else:
            goal_pos = self._get_goal_position(ee_pos, obj_pos, target_pos)
            delta = goal_pos - ee_pos
            dist = np.linalg.norm(delta)

            if self._step % 50 == 0:
                logger.debug(
                    f"  [Step {self._step}] phase={self.phase.name} "
                    f"dist={dist:.4f} gs={obs.get('gripper_state', 0):.4f} "
                    f"gripper_cmd={'OPEN' if gripper_open else 'CLOSE'}"
                )

            max_step = self._get_max_cart_step()
            if dist > max_step:
                clipped_goal = ee_pos + delta * (max_step / dist)
            else:
                clipped_goal = goal_pos

            alpha = 0.5
            joint_targets = self.ik_solver.solve_position(clipped_goal, current_joints)
            joint_targets = current_joints + alpha * (joint_targets - current_joints)

        noise = np.random.randn(7) * self.noise_scale
        joint_targets = joint_targets + noise

        gripper_cmd = gripper_open * 255.0
        action = np.concatenate([joint_targets, [gripper_cmd]])
        return action

    def _get_max_cart_step(self) -> float:
        """每步最大笛卡尔移动距离（米）。抓取后极慢移动以维持摩擦。"""
        if self.phase == Phase.LIFT:
            return 0.001
        elif self.phase == Phase.MOVE_TO_TARGET:
            return 0.001
        elif self.phase in (Phase.REACH_OBJECT, Phase.PLACE):
            return 0.005
        else:
            return 0.012

    def _get_goal_position(
        self, ee_pos: np.ndarray, obj_pos: np.ndarray, target_pos: np.ndarray
    ) -> np.ndarray:
        if self.phase == Phase.REACH_ABOVE:
            return np.array([obj_pos[0], obj_pos[1], obj_pos[2] + self.approach_height])
        elif self.phase == Phase.REACH_OBJECT:
            return np.array([obj_pos[0], obj_pos[1], obj_pos[2] + self.grasp_height])
        elif self.phase == Phase.GRASP:
            return np.array([obj_pos[0], obj_pos[1], obj_pos[2] + self.grasp_height])
        elif self.phase == Phase.LIFT:
            return np.array([0.25, 0.0, 0.59])
        elif self.phase == Phase.MOVE_TO_TARGET:
            return np.array([target_pos[0], target_pos[1], 0.59])
        elif self.phase in (Phase.PLACE, Phase.RELEASE):
            return np.array([target_pos[0], target_pos[1], 0.57])
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
            if dist < self.pos_threshold and self._phase_step >= 15:
                logger.info(f"    REACH_OBJECT done (dist={dist:.4f}, steps={self._phase_step})")
                self.phase = Phase.GRASP
                self._grasp_counter = 0
                self._phase_step = 0

        elif self.phase == Phase.GRASP:
            self._grasp_counter += 1
            if self._grasp_counter > 80:
                self._pick_z = obj_pos[2]
                logger.info(f"    GRASP done (held {self._grasp_counter} steps, pick_z={self._pick_z:.4f})")
                self.phase = Phase.LIFT
                self._phase_step = 0

        elif self.phase == Phase.LIFT:
            if (ee_pos[2] > 0.58 or self._phase_step >= 200) and self._phase_step >= 30:
                logger.info(f"    LIFT done (z={ee_pos[2]:.4f}, steps={self._phase_step})")
                self.phase = Phase.MOVE_TO_TARGET
                self._phase_step = 0

        elif self.phase == Phase.MOVE_TO_TARGET:
            goal_xy = np.array([target_pos[0], target_pos[1]])
            dist_xy = np.linalg.norm(ee_pos[:2] - goal_xy)
            if (dist_xy < 0.03 or self._phase_step >= 150) and self._phase_step >= 20:
                logger.info(f"    MOVE_TO_TARGET done (dist_xy={dist_xy:.4f}, steps={self._phase_step})")
                self.phase = Phase.PLACE
                self._phase_step = 0

        elif self.phase == Phase.PLACE:
            goal = np.array([target_pos[0], target_pos[1], target_pos[2] + self.grasp_height])
            dist = np.linalg.norm(ee_pos - goal)
            if dist < self.pos_threshold and self._phase_step >= 15:
                logger.info(f"    PLACE done (dist={dist:.4f})")
                self.phase = Phase.RELEASE
                self._grasp_counter = 0
                self._phase_step = 0

        elif self.phase == Phase.RELEASE:
            self._grasp_counter += 1
            if self._grasp_counter > 20:
                logger.info("    RELEASE done")
                self.phase = Phase.DONE
                self._phase_step = 0
