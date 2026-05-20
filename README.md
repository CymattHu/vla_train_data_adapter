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

整个流程只需 3 步即可跑通数据转换 pipeline。

### 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0（可选，方便编排）
- Make（可选，简化命令）

### Step 1：构建镜像

```bash
# 使用 make
make build

# 或直接 docker build
docker build -t vla-adapter .
```

### Step 2：准备数据

可以使用自己的数据，也可以用内置脚本生成示例数据来快速验证：

```bash
# 生成 5 个 episode 的示例数据（需要本地有 python + numpy + Pillow）
make sample

# 或手动运行
python scripts/generate_sample_data.py --output ./data/input --episodes 5
```

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
  convert --source real_robot --input /data/input --format lerobot --output /data/output/lerobot
```

### 使用 Docker Compose

```bash
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
│   └── generate_sample_data.py    # 示例数据生成
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
| 1. 数据采集 | 收集遥操作数据 | `RealRobotTeleopAdapter` |
| 2. 数据验证 | 检查质量和完整性 | `vla-adapter inspect` |
| 3. BC Baseline | 验证数据可用性 | ACT / Diffusion Policy + LeRobot |
| 4. VLA Fine-tune | 训练 VLA 模型 | openpi/pi0 + LoRA |
| 5. 评估 | sim + real rollout | policy server |
