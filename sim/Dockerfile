FROM python:3.12-slim

WORKDIR /app

# MuJoCo 无头渲染依赖（OSMesa）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libegl1 \
    libosmesa6 \
    libosmesa6-dev \
    libglew-dev \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV XDG_CACHE_HOME=/tmp/.cache

# 下载官方 Franka Panda MJCF 模型
RUN git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git /app/sim/assets/mujoco_menagerie \
    && cd /app/sim/assets/mujoco_menagerie \
    && git sparse-checkout set franka_emika_panda

COPY sim/requirements.txt /app/sim/requirements.txt
RUN pip install --no-cache-dir -r /app/sim/requirements.txt

COPY sim/ /app/sim/
COPY vla_data_adapter/ /app/vla_data_adapter/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e .

VOLUME ["/data/input"]

ENTRYPOINT ["python", "-m", "sim.generate_data"]
CMD ["--output", "/data/input", "--episodes", "50"]
