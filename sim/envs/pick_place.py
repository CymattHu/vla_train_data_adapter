"""MuJoCo Pick-and-Place 仿真环境。

使用 mujoco_menagerie 官方 Franka Emika Panda MJCF 模型。
支持无头渲染（EGL/OSMesa），适合在 Docker 中运行。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

_MENAGERIE_DIR = Path(__file__).parent.parent / "assets" / "mujoco_menagerie"
_PANDA_DIR = _MENAGERIE_DIR / "franka_emika_panda"


def _build_scene_xml(config: "PickPlaceConfig") -> str:
    """构建包含 Panda + 桌面任务的场景 XML。

    通过 <include> 引入官方 panda.xml，再添加桌子、物体、相机。
    在 hand body 上添加 ee_site 用于 IK 计算。
    """
    panda_xml_path = _PANDA_DIR / "panda.xml"
    if not panda_xml_path.exists():
        raise FileNotFoundError(
            f"Franka Panda MJCF not found at {panda_xml_path}\n"
            "请运行: ./sim/setup_assets.sh"
        )

    return f"""
<mujoco model="panda_pick_place">
  <include file="{panda_xml_path}"/>

  <option timestep="0.002" integrator="implicitfast"/>

  <statistic center="0.3 0 0.4" extent="0.8"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global offwidth="{config.image_width}" offheight="{config.image_height}"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
    <material name="table_mat" rgba="0.6 0.5 0.4 1"/>
    <material name="obj_mat" rgba="0.8 0.15 0.15 1"/>
    <material name="target_mat" rgba="0.15 0.8 0.15 0.4"/>
  </asset>

  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- Table -->
    <body name="table" pos="0.45 0 0.225">
      <geom type="box" size="0.35 0.45 0.225" material="table_mat" mass="200"/>
    </body>

    <!-- Object to pick (cylinder for stable finger contact) -->
    <body name="object" pos="0.4 0.0 0.475">
      <joint name="obj_free" type="free"/>
      <geom name="object_geom" type="cylinder" size="0.015 0.025"
            material="obj_mat" mass="0.03"
            condim="4" friction="2.0 0.05 0.01"
            solref="0.002 1" solimp="0.99 0.999 0.001"
            priority="1"/>
    </body>

    <!-- Target position -->
    <body name="target" pos="0.4 -0.2 0.47">
      <geom type="cylinder" size="0.04 0.002" material="target_mat"
            contype="0" conaffinity="0"/>
    </body>

    <!-- Cameras -->
    <camera name="front" pos="0.5 -1.0 0.9" xyaxes="1 0 0 0 0.5 0.87"/>
    <camera name="wrist" pos="0.5 0 1.5" xyaxes="1 0 0 0 1 0"
            mode="targetbody" target="hand"/>
    <camera name="side" pos="1.3 0 0.7" xyaxes="0 1 0 -0.4 0 0.92"/>
  </worldbody>

  <!-- EE site 附加到 hand body 上，用于 IK 计算 -->
  <worldbody>
    <body name="ee_site_body" mocap="true">
      <site name="ee_target_vis" size="0.01" rgba="1 0 0 0.3"/>
    </body>
  </worldbody>

  <!-- 在 hand body 上附加 site -->
  <contact>
    <exclude body1="table" body2="link0"/>
  </contact>
</mujoco>
"""


@dataclass
class PickPlaceConfig:
    """环境配置。"""
    image_width: int = 640
    image_height: int = 480
    control_frequency: int = 20
    sim_steps_per_control: int = 20
    object_random_range: float = 0.06
    target_random_range: float = 0.06
    max_steps: int = 200
    success_threshold: float = 0.04


class PickPlaceEnv:
    """基于官方 Franka Panda MJCF 的 Pick-and-Place 环境。

    使用 mujoco_menagerie/franka_emika_panda/panda.xml 官方模型。
    - 7 关节 position actuator + 1 夹爪 actuator（ctrl 0~255）
    - 通过 hand body 位置获取末端执行器位姿
    """

    JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    HOME_QPOS = np.array([0, -0.785, 0, -2.356, 0, 1.571, -0.785])
    GRIPPER_OPEN = 255.0
    GRIPPER_CLOSE = 0.0

    def __init__(self, config: PickPlaceConfig | None = None):
        self.config = config or PickPlaceConfig()

        scene_xml = _build_scene_xml(self.config)

        scene_path = _PANDA_DIR / "_pick_place_scene.xml"
        scene_path.write_text(scene_xml)
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))

        gripper_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        if gripper_act_id >= 0:
            self.model.actuator_gainprm[gripper_act_id, 0] = 0.04
            self.model.actuator_biasprm[gripper_act_id, 1] = -500.0
            self.model.actuator_biasprm[gripper_act_id, 2] = -50.0
            self.model.actuator_forcerange[gripper_act_id] = [-500.0, 500.0]

        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.config.image_height,
            width=self.config.image_width,
        )

        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.JOINT_NAMES
        ]
        self._hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")

        self._step_count = 0
        self._object_init_pos = np.array([0.4, 0.0, 0.475])
        self._target_init_pos = np.array([0.4, -0.2, 0.47])
        self._grasped = False
        self._grasp_offset = np.zeros(3)

    @property
    def num_joints(self) -> int:
        return 7

    @property
    def num_actuators(self) -> int:
        return self.model.nu  # 8: 7 joints + 1 gripper

    def reset(self, seed: int | None = None) -> dict:
        """重置环境。"""
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        for i, jid in enumerate(self._joint_ids):
            self.data.qpos[self.model.jnt_qposadr[jid]] = self.HOME_QPOS[i]

        finger_jnt1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        finger_jnt2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        self.data.qpos[self.model.jnt_qposadr[finger_jnt1]] = 0.04
        self.data.qpos[self.model.jnt_qposadr[finger_jnt2]] = 0.04

        obj_offset = np.random.uniform(
            -self.config.object_random_range,
            self.config.object_random_range,
            size=2,
        )
        obj_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")
        obj_qpos_adr = self.model.jnt_qposadr[obj_jnt_id]
        self.data.qpos[obj_qpos_adr:obj_qpos_adr + 3] = self._object_init_pos + [obj_offset[0], obj_offset[1], 0]
        self.data.qpos[obj_qpos_adr + 3:obj_qpos_adr + 7] = [1, 0, 0, 0]

        target_offset = np.random.uniform(
            -self.config.target_random_range,
            self.config.target_random_range,
            size=2,
        )
        self.model.body_pos[self._target_body_id] = self._target_init_pos + [target_offset[0], target_offset[1], 0]

        self.data.ctrl[:7] = self.HOME_QPOS
        self.data.ctrl[7] = self.GRIPPER_OPEN

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._grasped = False
        self._grasp_offset = np.zeros(3)
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """执行一步动作。

        Args:
            action: (8,) [7 joint targets + 1 gripper (0~255)]
                   或 (9,) [7 joints + gripper_open_flag(0/1) + unused]
        """
        ctrl = np.zeros(self.model.nu)
        ctrl[:7] = action[:7]

        if len(action) >= 8:
            ctrl[7] = action[7]
        else:
            ctrl[7] = self.data.ctrl[7]

        ctrl[:7] = np.clip(ctrl[:7], self.model.actuator_ctrlrange[:7, 0], self.model.actuator_ctrlrange[:7, 1])
        ctrl[7] = np.clip(ctrl[7], 0, 255)
        self.data.ctrl[:] = ctrl

        gripper_closing = ctrl[7] < 10
        if gripper_closing and not self._grasped:
            self._check_grasp()

        if self._grasped:
            self._enforce_grasp()

        if not gripper_closing and self._grasped:
            self._grasped = False

        for _ in range(self.config.sim_steps_per_control):
            if self._grasped:
                self._enforce_grasp()
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward, success = self._compute_reward(obs)
        done = success or self._step_count >= self.config.max_steps

        return obs, reward, done, {"success": success, "step": self._step_count}

    def _check_grasp(self):
        """检测夹爪是否夹住了物体（基于距离和夹爪闭合程度）。"""
        ee_pos = self.get_ee_position()
        obj_pos = self.get_object_position()
        gripper_state = self.get_gripper_state()

        dist_xy = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
        dist_z = abs(ee_pos[2] - obj_pos[2])

        if dist_xy < 0.05 and dist_z < 0.15 and gripper_state < 0.025:
            self._grasped = True
            self._grasp_offset = obj_pos - ee_pos

    def _enforce_grasp(self):
        """通过直接设置物体位置来模拟稳定抓取。"""
        ee_pos = self.get_ee_position()
        target_obj_pos = ee_pos + self._grasp_offset

        obj_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")
        obj_qpos_adr = self.model.jnt_qposadr[obj_jnt_id]
        self.data.qpos[obj_qpos_adr:obj_qpos_adr + 3] = target_obj_pos
        self.data.qvel[self.model.jnt_dofadr[obj_jnt_id]:self.model.jnt_dofadr[obj_jnt_id] + 6] = 0

    def render(self, camera_name: str = "front") -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render()

    def get_joint_positions(self) -> np.ndarray:
        return np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]]
            for jid in self._joint_ids
        ])

    def get_ee_position(self) -> np.ndarray:
        """通过 hand body 位置获取 EE 位置。"""
        return self.data.xpos[self._hand_body_id].copy()

    def get_object_position(self) -> np.ndarray:
        return self.data.xpos[self._obj_body_id].copy()

    def get_target_position(self) -> np.ndarray:
        return self.model.body_pos[self._target_body_id].copy()

    def get_gripper_state(self) -> float:
        """夹爪状态：0=闭合, 0.04=完全张开。"""
        fj1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        return float(self.data.qpos[self.model.jnt_qposadr[fj1]])

    def _get_obs(self) -> dict:
        return {
            "joint_positions": self.get_joint_positions(),
            "ee_position": self.get_ee_position(),
            "object_position": self.get_object_position(),
            "target_position": self.get_target_position(),
            "gripper_state": self.get_gripper_state(),
        }

    def _compute_reward(self, obs: dict) -> tuple[float, bool]:
        obj_pos = obs["object_position"]
        target_pos = obs["target_position"]
        dist = np.linalg.norm(obj_pos[:2] - target_pos[:2])
        height_ok = obj_pos[2] > 0.4
        success = dist < self.config.success_threshold and height_ok
        reward = -dist + (10.0 if success else 0.0)
        return float(reward), bool(success)

    def close(self):
        self.renderer.close()
