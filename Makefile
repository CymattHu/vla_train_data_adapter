.PHONY: build run sample convert-lerobot convert-openpi inspect test clean

IMAGE_NAME := vla-adapter

build:
	docker build -t $(IMAGE_NAME) .

# 生成示例数据（本地 python 执行）
sample:
	python scripts/generate_sample_data.py --output ./data/input --episodes 5 --frames 50

# 通用运行入口
run:
	docker run --rm -v $(PWD)/data/input:/data/input -v $(PWD)/data/output:/data/output $(IMAGE_NAME) $(CMD)

# 快速转换示例
convert-lerobot:
	docker run --rm \
		-v $(PWD)/data/input:/data/input \
		-v $(PWD)/data/output:/data/output \
		$(IMAGE_NAME) \
		convert --source real_robot --input /data/input --format lerobot --output /data/output/lerobot

convert-openpi:
	docker run --rm \
		-v $(PWD)/data/input:/data/input \
		-v $(PWD)/data/output:/data/output \
		$(IMAGE_NAME) \
		convert --source real_robot --input /data/input --format openpi --output /data/output/openpi

convert-groot:
	docker run --rm \
		-v $(PWD)/data/input:/data/input \
		-v $(PWD)/data/output:/data/output \
		$(IMAGE_NAME) \
		convert --source real_robot --input /data/input --format groot --output /data/output/groot

inspect:
	docker run --rm \
		-v $(PWD)/data/input:/data/input \
		$(IMAGE_NAME) \
		inspect --source real_robot --input /data/input --verbose

validate:
	docker run --rm \
		-v $(PWD)/data/output:/data/output \
		$(IMAGE_NAME) \
		validate --format lerobot --output /data/output/lerobot

test:
	docker run --rm --entrypoint pytest $(IMAGE_NAME) tests/ -v

clean:
	rm -rf data/output/*
