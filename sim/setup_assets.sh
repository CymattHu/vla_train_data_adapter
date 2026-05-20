#!/bin/bash
# 下载官方 Franka Panda MJCF 模型（本地开发用）
#
# 用法: ./sim/setup_assets.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="$SCRIPT_DIR/assets/mujoco_menagerie"

if [ -d "$ASSETS_DIR/franka_emika_panda" ]; then
    echo "Franka Panda 模型已存在: $ASSETS_DIR/franka_emika_panda"
    exit 0
fi

echo "下载 mujoco_menagerie Franka Panda 模型..."
mkdir -p "$ASSETS_DIR"

git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git "$ASSETS_DIR"

cd "$ASSETS_DIR"
git sparse-checkout set franka_emika_panda

echo "完成! 模型位于: $ASSETS_DIR/franka_emika_panda"
