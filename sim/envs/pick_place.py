"""MuJoCo Pick-and-Place 仿真环境。

使用 mujoco_menagerie 官方 Franka Emika Panda MJCF 模型。
支持无头渲染（EGL/OSMesa），适合在 Docker 中运行。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

logger = logging.getLogger(__name__)

_MENAGERIE_DIR = Path(__file__).parent.parent / "assets" / "mujoco_menagerie"
_PANDA_DIR = _MENAGERIE_DIR / "franka_emika_panda"


def _build_scene_xml(config: "PickPlaceConfig") -> str:
    """构建包含 Panda + 桌面任务的场景 XML。"""
    panda_xml_path = _PANDA_DIR / "panda.xml"
    if not panda_xml_path.exists():
        raise FileNotFoundError(
            f"Franka Panda MJCF not found at {panda_xml_path}\n"
            "请运行: ./sim/setup_assets.sh"
        )

    return f"""
<mujoco model="panda_pick_place">
  <include file="{panda_xml_path}"/>

  <option timestep="0.002" integrator="implicitfast" noslip_iterations="5" noslip_tolerance="1e-6"/>

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

    <!-- Object to pick: cube 5cm -->
    <body name="object" pos="0.30 0.0 0.475">
      <joint name="obj_free" type="free"/>
      <geom name="object_geom" type="box" size="0.025 0.025 0.025"
            material="obj_mat" mass="0.05"
            condim="6" friction="2.0 0.1 0.01"
            solref="0.001 1.0" solimp="0.99 0.999 0.001"
            priority="1"/>
    </body>

    <!-- Target position -->
    <body name="target" pos="0.30 -0.12 0.47">
      <geom type="cylinder" size="0.04 0.002" material="target_mat"
            contype="0" conaffinity="0"/>
    </body>

    <!-- Cameras -->
    <camera name="front" pos="0.5 -1.0 0.9" xyaxes="1 0 0 0 0.5 0.87"/>
    <camera name="wrist" pos="0.5 0 1.5" xyaxes="1 0 0 0 1 0"
            mode="targetbody" target="hand"/>
    <camera name="side" pos="1.3 0 0.7" xyaxes="0 1 0 -0.4 0 0.92"/>
  </worldbody>

  <worldbody>
    <body name="ee_site_body" mocap="true">
      <site name="ee_target_vis" size="0.01" rgba="1 0 0 0.3"/>
    </body>
  </worldbody>

  <contact>
    <exclude body1="table" body2="link0"/>
    <exclude body1="hand" body2="object"/>
  </contact>
</mujoco>
"""


@dataclass
class PickPlaceConfig:
    """环境配置。"""
    image_width: int = 640
    image_height: int = 480
    control_frequency: int = 20
    sim_steps_per_control: int = 50
    object_random_range: float = 0.03
    target_random_range: float = 0.03
    max_steps: int = 600
    success_threshold: float = 0.05


class PickPlaceEnv:
    """基于官方 Franka Panda MJCF 的 Pick-and-Place 环境。

    使用 mujoco_menagerie/franka_emika_panda/panda.xml 官方模型。
    - 7 关节 position actuator + 1 夹爪 actuator（ctrl 0~255）
    - 通过 hand body 位置获取末端执行器位姿
    - 纯物理摩擦抓取（无 grasp-lock）
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

        self._tune_gripper_actuator()
        self._tune_finger_friction()

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
        self._object_init_pos = np.array([0.30, 0.0, 0.475])
        self._target_init_pos = np.array([0.30, -0.12, 0.47])

    def _tune_gripper_actuator(self):
        """增大夹爪执行器力量。保持 ctrl 范围映射：0=闭合, 255=全开(0.04m)。"""
        act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        if act_id >= 0:
            kp = 800.0
            kv = 50.0
            gain = kp * 0.04 / 255.0
            self.model.actuator_gainprm[act_id, 0] = gain
            self.model.actuator_biasprm[act_id, 0] = 0.0
            self.model.actuator_biasprm[act_id, 1] = -kp
            self.model.actuator_biasprm[act_id, 2] = -kv
            self.model.actuator_forcerange[act_id] = [-1000.0, 1000.0]

    def _tune_finger_friction(self):
        """设置手指碰撞 geom 的高摩擦力。"""
        lf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        rf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        for i in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[i]
            if body_id in (lf_id, rf_id):
                self.model.geom_friction[i] = [2.0, 0.1, 0.01]
                self.model.geom_condim[i] = 6
                self.model.geom_priority[i] = 1
                self.model.geom_solref[i] = [0.001, 1.0]
                self.model.geom_solimp[i] = [0.99, 0.999, 0.001, 0.5, 2.0]

    @property
    def num_joints(self) -> int:
        return 7

    @property
    def num_actuators(self) -> int:
        return self.model.nu

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

        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """执行一步动作（纯物理，无 grasp-lock）。"""
        ctrl = np.zeros(self.model.nu)
        ctrl[:7] = action[:7]

        if len(action) >= 8:
            ctrl[7] = action[7]
        else:
            ctrl[7] = self.data.ctrl[7]

        ctrl[:7] = np.clip(ctrl[:7], self.model.actuator_ctrlrange[:7, 0], self.model.actuator_ctrlrange[:7, 1])
        ctrl[7] = np.clip(ctrl[7], 0, 255)
        self.data.ctrl[:] = ctrl

        for _ in range(self.config.sim_steps_per_control):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward, success = self._compute_reward(obs)
        done = self._step_count >= self.config.max_steps

        return obs, reward, done, {"success": success, "step": self._step_count}

    def render(self, camera_name: str = "front") -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render()

    def get_joint_positions(self) -> np.ndarray:
        return np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]]
            for jid in self._joint_ids
        ])

    def get_ee_position(self) -> np.ndarray:
        return self.data.xpos[self._hand_body_id].copy()

    def get_object_position(self) -> np.ndarray:
        return self.data.xpos[self._obj_body_id].copy()

    def get_target_position(self) -> np.ndarray:
        return self.model.body_pos[self._target_body_id].copy()

    def get_gripper_state(self) -> float:
        """夹爪状态：0=完全闭合, 0.04=完全张开。"""
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
        success = dist < self.config.success_threshold and obj_pos[2] > 0.48
        reward = -dist + (10.0 if success else 0.0)
        return float(reward), bool(success)

    def close(self):
        self.renderer.close()
