.PHONY: help build build-sim build-all \
	sim sim-quick sim-full sim-custom sample \
	convert-lerobot convert-openpi convert-groot convert-all \
	inspect validate \
	pipeline pipeline-quick \
	test dev-install clean

IMAGE_NAME := vla-adapter
SIM_IMAGE := vla-sim

# 默认目标：显示帮助
help:
	@echo "VLA Training Data Adapter - 常用命令"
	@echo ""
	@echo "构建镜像:"
	@echo "  make build           构建数据转换镜像 (vla-adapter)"
	@echo "  make build-sim       构建 MuJoCo 仿真镜像 (vla-sim)"
	@echo "  make build-all       构建以上全部镜像"
	@echo ""
	@echo "生成数据 (MuJoCo 仿真):"
	@echo "  make sim             生成 50 个成功 episode"
	@echo "  make sim-quick       快速生成 5 个 episode (冒烟测试)"
	@echo "  make sim-full        生成 200 个成功 episode"
	@echo "  make sim-custom ARGS=\"--episodes 100 --seed 7\"   自定义参数"
	@echo "  make sample          生成随机示例数据 (无需 MuJoCo, 本地 Python)"
	@echo ""
	@echo "转换数据 (source=mujoco_sim):"
	@echo "  make convert-lerobot 转为 LeRobot 格式"
	@echo "  make convert-openpi  转为 OpenPI/pi0 格式"
	@echo "  make convert-groot   转为 GR00T 格式"
	@echo "  make convert-all     一次性转为以上全部格式"
	@echo ""
	@echo "检查与一键流程:"
	@echo "  make inspect         检查数据质量"
	@echo "  make validate        验证 LeRobot 导出结果"
	@echo "  make pipeline        sim -> convert-lerobot -> inspect"
	@echo "  make pipeline-quick  sim-quick -> convert-lerobot -> inspect"
	@echo ""
	@echo "开发:"
	@echo "  make dev-install     本地可编辑安装 (pip install -e .[all,dev])"
	@echo "  make test            在容器内跑 pytest"
	@echo "  make clean           清空 data/input 和 data/output"

# ============================================================
# 构建
# ============================================================
build:
	docker build -t $(IMAGE_NAME) .

build-sim:
	docker build -t $(SIM_IMAGE) -f sim/Dockerfile .

build-all: build build-sim

# ============================================================
# 仿真数据生成 (MuJoCo)
# ============================================================

# 生成 50 个成功的 episode（约 5-10 分钟）
sim:
	docker run --rm -u $$(id -u):$$(id -g) -v $(PWD)/data/input:/data/input $(SIM_IMAGE) \
		--output /data/input --episodes 50 --success-only

# 快速测试：生成 5 个 episode
sim-quick:
	docker run --rm -u $$(id -u):$$(id -g) -v $(PWD)/data/input:/data/input $(SIM_IMAGE) \
		--output /data/input --episodes 5 --max-steps 150

# 完整数据集：生成 200 个成功的 episode
sim-full:
	docker run --rm -u $$(id -u):$$(id -g) -v $(PWD)/data/input:/data/input $(SIM_IMAGE) \
		--output /data/input --episodes 200 --success-only

# 自定义参数生成
# 用法: make sim-custom ARGS="--episodes 100 --success-only --seed 123"
sim-custom:
	docker run --rm -u $$(id -u):$$(id -g) -v $(PWD)/data/input:/data/input $(SIM_IMAGE) \
		--output /data/input $(ARGS)

# 不依赖 MuJoCo 的随机示例数据（纯本地 Python，用于快速验证 pipeline）
sample:
	python scripts/generate_sample_data.py --output ./data/input --episodes 5

# ============================================================
# 数据转换
# ============================================================
convert-lerobot:
	docker run --rm \
		-v $(PWD)/data/input:/data/input \
		-v $(PWD)/data/output:/data/output \
		$(IMAGE_NAME) \
		convert --source mujoco_sim --input /data/input --format lerobot --output /data/output/lerobot

convert-openpi:
	docker run --rm \
		-v $(PWD)/data/input:/data/input \
		-v $(PWD)/data/output:/data/output \
		$(IMAGE_NAME) \
		convert --source mujoco_sim --input /data/input --format openpi --output /data/output/openpi

convert-groot:
	docker run --rm \
		-v $(PWD)/data/input:/data/input \
		-v $(PWD)/data/output:/data/output \
		$(IMAGE_NAME) \
		convert --source mujoco_sim --input /data/input --format groot --output /data/output/groot

convert-all: convert-lerobot convert-openpi convert-groot
	@echo "All formats exported to ./data/output/"

# ============================================================
# 检查和验证
# ============================================================
inspect:
	docker run --rm \
		-v $(PWD)/data/input:/data/input \
		$(IMAGE_NAME) \
		inspect --source mujoco_sim --input /data/input --verbose

validate:
	docker run --rm \
		-v $(PWD)/data/output:/data/output \
		$(IMAGE_NAME) \
		validate --format lerobot --output /data/output/lerobot

# ============================================================
# 完整流程：生成 → 转换（一键跑通）
# ============================================================
pipeline: sim convert-lerobot inspect
	@echo "Pipeline complete! Data at ./data/output/lerobot"

pipeline-quick: sim-quick convert-lerobot inspect
	@echo "Quick pipeline complete!"

# ============================================================
# 开发和测试
# ============================================================
dev-install:
	pip install -e ".[all,dev]"

test:
	docker run --rm --entrypoint pytest $(IMAGE_NAME) tests/ -v

clean:
	rm -rf data/input/* data/output/*
