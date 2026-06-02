# VLA Training Data Adapter

统一机器人 VLA (Vision-Language-Action) 训练数据转换框架。

## 架构

```
Data Source Layer                  Canonical Episode              Export Layer
┌─────────────────┐              ┌──────────────────┐           ┌──────────────────┐
│ Real Robot Teleop│──┐          │                  │     ┌────▶│ LeRobot Format   │
│ Isaac / MuJoCo  │──┤  load()  │  Normalization   │     │     │ (→ openpi/pi0)   │
│ DROID / LIBERO  │──┼─────────▶│  + Validation    │─────┤     ├──────────────────┤
│ Open-X Embodiment│──┤          │  + Quality Check │     ├────▶│ GR00T Format     │
│ Human Video     │──┘          │                  │     │     ├──────────────────┤
└─────────────────┘              └──────────────────┘     └────▶│ OpenPI Format    │
                                                                └──────────────────┘
```

**核心设计原则：**
- Source Adapter 负责"采集" — 每个数据源实现 `load()` 输出统一 Episode
- Canonical Episode 负责"统一" — 内部唯一数据格式，不绑定任何模型
- Model Adapter 负责"适配模型" — 导出为 LeRobot / GR00T / OpenPI 等格式

---

## 快速开始（Docker）

整个流程只需 3 步即可跑通数据转换 pipeline。`make`（不带参数）可随时查看所有可用命令。

### 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0（可选，方便编排）
- Make（可选，简化命令）

### Step 1：构建镜像

项目包含两个镜像：`vla-adapter`（数据转换）和 `vla-sim`（MuJoCo 仿真，含无头渲染依赖）。

```bash
# 一次性构建两个镜像
make build-all

# 或分别构建
make build        # 数据转换镜像 vla-adapter
make build-sim    # MuJoCo 仿真镜像 vla-sim
```

### Step 2：准备数据

有三种方式得到 `data/input/` 下的数据，任选其一：

```bash
# 方式 A（推荐）：用 MuJoCo 仿真生成 pick-and-place demonstration
make sim-quick    # 快速生成 5 个 episode（冒烟测试）
make sim          # 生成 50 个成功 episode
make sim-full     # 生成 200 个成功 episode
make sim-custom ARGS="--episodes 100 --seed 123 --success-only"

# 方式 B：用纯随机示例数据快速验证 pipeline（无需 MuJoCo，本地 Python）
make sample

# 方式 C：放入你自己的数据，遵循下方目录结构即可
```

> 仿真细节（任务、IK 求解器、专家策略、可调参数）见 [`sim/README.md`](sim/README.md)。
> 仿真输出格式与真机遥操作 (`RealRobotTeleopAdapter`) 完全一致，仅 `source_type` 标记为 `sim`，因此可与真机数据无缝混用。

生成后目录结构如下：
```
data/input/
├── episode_000/
│   ├── metadata.json
│   ├── timestamps.npy
│   ├── joint_positions.npy
│   ├── actions.npy
│   └── images/
│       ├── front/
│       │   ├── 000000.png
│       │   └── ...
│       └── wrist/
│           ├── 000000.png
│           └── ...
├── episode_001/
└── ...
```

### Step 3：运行转换

```bash
# 转为 LeRobot 格式（pi0/openpi 依赖此格式）
make convert-lerobot

# 转为 OpenPI 格式（LeRobot + openpi config + norm stats）
make convert-openpi

# 转为 GR00T 格式
make convert-groot

# 一次性转为以上全部格式
make convert-all

# 检查数据集质量
make inspect

# 验证导出结果
make validate
```

等价的 Docker 命令：

```bash
docker run --rm \
  -v $(pwd)/data/input:/data/input \
  -v $(pwd)/data/output:/data/output \
  vla-adapter \
  convert --source mujoco_sim --input /data/input --format lerobot --output /data/output/lerobot
```

### 一键跑通完整流程

```bash
# 仿真生成 → 转 LeRobot → 检查质量
make pipeline          # sim (50 个) + convert-lerobot + inspect
make pipeline-quick    # sim-quick (5 个) + convert-lerobot + inspect
```

### 使用 Docker Compose

```bash
# MuJoCo 仿真生成数据
docker compose run --rm sim-generate          # 50 个 episode
docker compose run --rm sim-generate-quick    # 5 个 episode（快速测试）

# 转 LeRobot 格式
docker compose run --rm convert-lerobot

# 转 OpenPI 格式
docker compose run --rm convert-openpi

# 检查数据
docker compose run --rm inspect
```

### 输出结构

转换完成后，`data/output/` 下会生成对应格式的数据：

```
data/output/
├── lerobot/           # LeRobot v2 格式
│   ├── meta/
│   │   ├── info.json
│   │   ├── episodes.jsonl
│   │   └── tasks.jsonl
│   ├── data/
│   │   └── chunk-000/
│   │       ├── episode_000000.parquet
│   │       └── ...
│   └── images/
├── openpi/            # OpenPI 格式
│   ├── lerobot_dataset/
│   ├── openpi_config.json
│   └── norm_stats.json
└── groot/             # GR00T 格式
    ├── dataset_config.json
    └── episodes/
```

---

## 本地开发

```bash
# 创建虚拟环境并安装
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"

# 运行测试
pytest tests/ -v

# 使用 CLI
vla-adapter convert --source real_robot --input ./data/input --format lerobot --output ./data/output/lerobot
vla-adapter inspect --source real_robot --input ./data/input --verbose
```

---

## Python API 使用

```python
from pathlib import Path
from vla_data_adapter import Pipeline, PipelineConfig
from vla_data_adapter.sources import RealRobotTeleopAdapter, RealRobotConfig
from vla_data_adapter.exporters import LeRobotAdapter, LeRobotExportConfig
from vla_data_adapter.normalization import NormalizationConfig

# 1. 配置数据源
source = RealRobotTeleopAdapter(RealRobotConfig(
    data_dir=Path("./data/input"),
    robot_type="panda",
    control_frequency=10,
))

# 2. 配置导出格式
exporter = LeRobotAdapter(LeRobotExportConfig(
    output_dir=Path("./data/output/lerobot"),
    repo_id="myuser/my_robot_dataset",
    fps=10,
))

# 3. 运行 pipeline（包含规范化和质量检查）
pipeline = Pipeline(PipelineConfig(
    normalization=NormalizationConfig(
        target_frequency=10.0,
        remove_failed_episodes=True,
        image_resize=(256, 256),
    )
))
output_path = pipeline.run(source, exporter)
print(f"Done: {output_path}")
```

---

## 项目结构

```
vla_train_data_adapter/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── scripts/
│   └── generate_sample_data.py    # 随机示例数据生成
├── sim/                           # MuJoCo 仿真数据生成（独立子项目，见 sim/README.md）
│   ├── Dockerfile                 # 仿真镜像（含 MuJoCo + OSMesa 无头渲染）
│   ├── generate_data.py           # 仿真数据生成主程序
│   ├── envs/pick_place.py         # Pick-and-Place 环境
│   └── policies/                  # IK 求解器 + 状态机专家策略
├── tests/
│   ├── test_schema.py
│   ├── test_normalization.py
│   └── test_pipeline.py
└── vla_data_adapter/
    ├── schema/                    # Canonical Episode 数据模型
    │   └── episode.py
    ├── sources/                   # 数据源适配器
    │   ├── base.py                # DataSourceAdapter 基类
    │   ├── real_robot.py          # 真实机器人遥操作
    │   ├── mujoco_sim.py          # MuJoCo 仿真（读取 sim/ 生成的数据）
    │   ├── droid.py               # DROID 数据集
    │   └── libero.py              # LIBERO 数据集
    ├── exporters/                 # 模型格式导出器
    │   ├── base.py                # ModelDatasetAdapter 基类
    │   ├── lerobot.py             # LeRobot v2 格式
    │   ├── openpi.py              # OpenPI/pi0 格式
    │   └── groot.py               # GR00T 格式
    ├── normalization/             # 规范化和质量检查
    │   └── normalizer.py
    ├── pipeline.py                # Pipeline 编排
    └── cli.py                     # CLI 入口
```

---

## 扩展指南

### 添加新数据源

```python
from vla_data_adapter.sources.base import DataSourceAdapter, DataSourceConfig
from vla_data_adapter.schema import Episode, SourceType

class MyRobotConfig(DataSourceConfig):
    my_param: str = "default"

class MyRobotAdapter(DataSourceAdapter):
    def get_episode_ids(self) -> list[str]:
        ...

    def load(self) -> Iterator[Episode]:
        for ep_id in self.get_episode_ids():
            yield Episode(
                episode_id=ep_id,
                task_instruction="pick up the cup",
                robot_type="my_robot",
                source_type=SourceType.REAL,
                frames=[...],
            )
```

### 添加新导出格式

```python
from vla_data_adapter.exporters.base import ModelDatasetAdapter, ExportConfig
from vla_data_adapter.schema import Episode

class MyModelAdapter(ModelDatasetAdapter):
    def export(self, episodes: list[Episode]) -> Path:
        ...

    def validate_export(self, output_dir: Path) -> bool:
        ...
```

---

## 推荐训练路线

| 阶段 | 目标 | 工具 |
|------|------|------|
| 1. 数据采集 | 仿真生成 / 收集遥操作数据 | `make sim` (MuJoCo) / `RealRobotTeleopAdapter` |
| 2. 数据验证 | 检查质量和完整性 | `make inspect` |
| 3. 格式转换 | 导出训练格式 | `make convert-lerobot` |
| 4. BC Baseline | 验证数据可用性 | ACT / Diffusion Policy + LeRobot |
| 5. VLA Fine-tune | 训练 VLA 模型 | openpi/pi0 + LoRA |
| 6. 评估 | sim + real rollout | `sim/envs` + policy server |
