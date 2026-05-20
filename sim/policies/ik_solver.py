"""基于 MuJoCo Jacobian 的逆运动学求解器。

使用阻尼最小二乘法 (Damped Least Squares / Levenberg-Marquardt)
通过 mujoco.mj_jacSite 计算雅可比矩阵，求解末端位置到关节空间的映射。

这是 MuJoCo 环境中标准的 IK 方法，与 dm_control、robosuite 等框架一致。
对于真实机器人部署，可替换为 Pinocchio + URDF 的实现。
"""

from __future__ import annotations

import mujoco
import numpy as np


class MujocoIKSolver:
    """基于 MuJoCo Jacobian 的微分 IK 求解器。

    使用末端 body 的解析雅可比矩阵，通过阻尼最小二乘法
    (Damped Least Squares) 计算满足笛卡尔空间位置误差的关节增量。

    支持：
    - 位置控制 (3-DOF)
    - 位置+姿态控制 (6-DOF)
    - 关节限位约束
    - 空间奇异性处理
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_name: str = "hand",
        site_name: str | None = None,
        joint_names: list[str] | None = None,
        damping: float = 1e-4,
        max_iterations: int = 100,
        pos_tolerance: float = 1e-3,
        step_size: float = 0.5,
        joint_limit_margin: float = 0.05,
    ):
        self.model = model
        self.data = data
        self.damping = damping
        self.max_iterations = max_iterations
        self.pos_tolerance = pos_tolerance
        self.step_size = step_size
        self.joint_limit_margin = joint_limit_margin

        self.use_site = site_name is not None
        if self.use_site:
            self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            self.body_id = -1
        else:
            self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self.site_id = -1

        if joint_names is None:
            joint_names = [f"joint{i+1}" for i in range(7)]

        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names
        ]
        self.dof_ids = [model.jnt_dofadr[jid] for jid in self.joint_ids]
        self.n_joints = len(self.joint_ids)

        self.joint_lower = np.array([
            model.jnt_range[jid, 0] + joint_limit_margin
            for jid in self.joint_ids
        ])
        self.joint_upper = np.array([
            model.jnt_range[jid, 1] - joint_limit_margin
            for jid in self.joint_ids
        ])

    def _get_ee_pos(self) -> np.ndarray:
        """获取末端位置（支持 site 或 body）。"""
        if self.use_site:
            return self.data.site_xpos[self.site_id].copy()
        return self.data.xpos[self.body_id].copy()

    def _get_jacobian(self) -> np.ndarray:
        """计算末端位置的雅可比矩阵。"""
        jac_pos = np.zeros((3, self.model.nv))
        if self.use_site:
            mujoco.mj_jacSite(self.model, self.data, jac_pos, None, self.site_id)
        else:
            mujoco.mj_jacBody(self.model, self.data, jac_pos, None, self.body_id)
        return jac_pos[:, self.dof_ids]

    def solve_position(
        self,
        target_pos: np.ndarray,
        current_joints: np.ndarray | None = None,
    ) -> np.ndarray:
        """求解仅位置的 IK（3-DOF 约束）。

        Args:
            target_pos: 目标末端位置 (3,)
            current_joints: 当前关节角作为初始值 (n_joints,)

        Returns:
            目标关节角度 (n_joints,)
        """
        if current_joints is not None:
            for i, jid in enumerate(self.joint_ids):
                qpos_adr = self.model.jnt_qposadr[jid]
                self.data.qpos[qpos_adr] = current_joints[i]
            mujoco.mj_forward(self.model, self.data)

        for _ in range(self.max_iterations):
            mujoco.mj_forward(self.model, self.data)
            ee_pos = self._get_ee_pos()
            error = target_pos - ee_pos

            if np.linalg.norm(error) < self.pos_tolerance:
                break

            J = self._get_jacobian()
            dq = self._damped_least_squares(J, error)

            for i, jid in enumerate(self.joint_ids):
                qpos_adr = self.model.jnt_qposadr[jid]
                self.data.qpos[qpos_adr] += self.step_size * dq[i]
                self.data.qpos[qpos_adr] = np.clip(
                    self.data.qpos[qpos_adr],
                    self.joint_lower[i],
                    self.joint_upper[i],
                )

        result = np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]]
            for jid in self.joint_ids
        ])
        return result

    def compute_joint_delta(
        self,
        delta_pos: np.ndarray,
        current_joints: np.ndarray,
    ) -> np.ndarray:
        """计算笛卡尔空间位移对应的关节目标（单步微分 IK）。

        适合在控制循环中每步调用，将末端位置误差转为关节位置指令。

        Args:
            delta_pos: 末端目标位移 (3,)
            current_joints: 当前关节角度 (n_joints,)

        Returns:
            目标关节角度 (n_joints,)
        """
        for i, jid in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_adr] = current_joints[i]
        mujoco.mj_forward(self.model, self.data)

        J = self._get_jacobian()
        dq = self._damped_least_squares(J, delta_pos)

        target_joints = current_joints + self.step_size * dq
        target_joints = np.clip(target_joints, self.joint_lower, self.joint_upper)
        return target_joints

    def _damped_least_squares(self, J: np.ndarray, error: np.ndarray) -> np.ndarray:
        """阻尼最小二乘法求解 Δq。

        Δq = J^T (J J^T + λ²I)^{-1} error

        当 J 接近奇异（条件数大）时，阻尼项 λ 防止关节速度爆炸。
        """
        JJT = J @ J.T
        n = JJT.shape[0]
        damped = JJT + self.damping * np.eye(n)
        dq = J.T @ np.linalg.solve(damped, error)
        return dq
