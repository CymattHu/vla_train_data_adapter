# MuJoCo 仿真数据生成

使用 MuJoCo 物理仿真环境（Panda 机械臂 Pick-and-Place 任务）自动生成 VLA 训练数据。

## 架构

```
sim/
├── run.sh                  # 一键启动脚本
├── Dockerfile              # 仿真 Docker 镜像（含 MuJoCo + OSMesa 无头渲染）
├── requirements.txt        # Python 依赖
├── generate_data.py        # 数据生成主程序
├── envs/
│   └── pick_place.py       # MuJoCo Pick-and-Place 环境
└── policies/
    ├── ik_solver.py        # Jacobian-based IK 求解器（阻尼最小二乘法）
    └── scripted.py         # 状态机专家策略
```

## 快速开始

### 方式一：使用启动脚本（推荐）

```bash
# 默认模式：生成 50 个成功的 episode
./sim/run.sh

# 快速测试：生成 5 个 episode（约 1 分钟）
./sim/run.sh --quick

# 完整数据集：生成 200 个成功的 episode
./sim/run.sh --full

# 自定义参数（交互式）
./sim/run.sh --custom
```

脚本会自动：
1. 检查并构建 Docker 镜像（首次约 1 分钟）
2. 询问是否清空旧数据
3. 运行仿真生成数据到 `data/input/`
4. 提示下一步操作

### 方式二：使用 Make

```bash
# 先构建镜像
make build-sim

# 生成 50 个 episode（只保留成功的）
make sim

# 快速测试 5 个
make sim-quick

# 自定义参数
make sim-custom ARGS="--episodes 100 --success-only --seed 123"
```

### 方式三：使用 Docker Compose

```bash
# 生成 50 个 episode
docker compose run --rm sim-generate

# 快速测试 5 个
docker compose run --rm sim-generate-quick
```

### 方式四：直接 Docker 命令

```bash
# 构建镜像
docker build -t vla-sim -f sim/Dockerfile .

# 运行（-u 避免权限问题）
docker run --rm -u $(id -u):$(id -g) \
  -v $(pwd)/data/input:/data/input \
  vla-sim \
  --output /data/input \
  --episodes 50 \
  --success-only \
  --max-steps 250
```

## 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output` | `/data/input` | 输出目录 |
| `--episodes` | `50` | 生成 episode 数量 |
| `--max-steps` | `200` | 每 episode 最大仿真步数 |
| `--task` | `pick up the red block...` | 任务自然语言描述 |
| `--success-only` | `false` | 只保留任务成功的 episode |
| `--frequency` | `10` | 控制频率 (Hz) |
| `--cameras` | `front wrist` | 相机列表 |
| `--image-height` | `480` | 图像高度 |
| `--image-width` | `640` | 图像宽度 |
| `--seed` | `42` | 随机种子（可复现） |

## 输出格式

生成的数据与 `RealRobotTeleopAdapter` / `MujocoSimAdapter` 兼容：

```
data/input/
├── episode_000/
│   ├── metadata.json          # 元信息（任务、成功与否、频率等）
│   ├── timestamps.npy         # (N,) float32 时间戳
│   ├── joint_positions.npy    # (N, 7) float32 关节角度
│   ├── actions.npy            # (N, 7) float32 关节动作
│   └── images/
│       ├── front/             # 前置相机
│       │   ├── 000000.png
│       │   ├── 000001.png
│       │   └── ...
│       └── wrist/             # 腕部相机
│           ├── 000000.png
│           └── ...
├── episode_001/
└── ...
```

## 生成后的下一步

```bash
# 1. 检查数据质量
make inspect

# 2. 转换为训练格式
make convert-lerobot      # → data/output/lerobot/
make convert-openpi       # → data/output/openpi/

# 3. 或一键跑通完整流程
make pipeline-quick       # sim → convert → inspect
```

## 仿真环境说明

### 任务

Panda 7-DOF 机械臂从桌面抓取红色方块，放到绿色目标位置。

- 物体和目标位置每 episode 随机偏移
- 成功条件：物体中心距目标 < 5cm
- 控制方式：关节位置控制 (position actuator)

### IK 求解器

使用 MuJoCo 原生 Jacobian API (`mj_jacSite`) + 阻尼最小二乘法：

```
Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ · Δx
```

- 直接使用仿真模型的运动学，零偏差
- 自动处理关节限位
- λ (damping) 防止奇异点附近不稳定

### 专家策略

状态机控制流程：

```
REACH_ABOVE → REACH_OBJECT → GRASP → LIFT → MOVE_TO_TARGET → PLACE → RELEASE → DONE
```

每个阶段通过 IK solver 将末端目标位置转换为关节角度指令。

## 扩展

### 添加新任务

在 `sim/envs/` 下创建新环境类，实现 `reset()` / `step()` / `render()` 接口：

```python
class PushEnv:
    def reset(self, seed=None) -> dict: ...
    def step(self, action) -> tuple[dict, float, bool, dict]: ...
    def render(self, camera_name="front") -> np.ndarray: ...
```

### 添加新策略

在 `sim/policies/` 下创建新策略：

```python
class ScriptedPushPolicy:
    def __init__(self, model, data): ...
    def reset(self): ...
    def get_action(self, obs: dict) -> np.ndarray: ...
```

### 使用 Pinocchio（真实机器人）

对于真实机器人部署，可将 `MujocoIKSolver` 替换为 Pinocchio 版本：

```python
import pinocchio as pin

class PinocchioIKSolver:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

    def compute_joint_delta(self, delta_pos, current_joints):
        pin.computeJointJacobians(self.model, self.data, current_joints)
        J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED)[:3]
        dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(3), delta_pos)
        return current_joints + self.step_size * dq
```
