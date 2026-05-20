.PHONY: build build-sim sim sim-quick convert-lerobot convert-openpi convert-groot inspect validate test clean

IMAGE_NAME := vla-adapter
SIM_IMAGE := vla-sim

# ============================================================
# 构建
# ============================================================
build:
	docker build -t $(IMAGE_NAME) .

build-sim:
	docker build -t $(SIM_IMAGE) -f sim/Dockerfile .

build-all: build build-sim

# ============================================================
# 仿真数据生成
# ============================================================

# 生成 50 个成功的 episode（约 5-10 分钟）
sim:
	docker run --rm -u $$(id -u):$$(id -g) -v $(PWD)/data/input:/data/input $(SIM_IMAGE) \
		--output /data/input --episodes 50 --success-only

# 快速测试：生成 5 个 episode
sim-quick:
	docker run --rm -u $$(id -u):$$(id -g) -v $(PWD)/data/input:/data/input $(SIM_IMAGE) \
		--output /data/input --episodes 5 --max-steps 150

# 自定义参数生成
# 用法: make sim-custom ARGS="--episodes 100 --success-only --seed 123"
sim-custom:
	docker run --rm -u $$(id -u):$$(id -g) -v $(PWD)/data/input:/data/input $(SIM_IMAGE) \
		--output /data/input $(ARGS)

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
test:
	docker run --rm --entrypoint pytest $(IMAGE_NAME) tests/ -v

clean:
	rm -rf data/input/* data/output/*
