#!/bin/bash
#
# MuJoCo 仿真数据生成启动脚本
#
# 用法:
#   ./sim/run.sh              # 默认生成 50 个 episode
#   ./sim/run.sh --quick      # 快速测试，生成 5 个
#   ./sim/run.sh --custom     # 自定义参数
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SIM_IMAGE="vla-sim"
DATA_DIR="${PROJECT_DIR}/data/input"
USER_FLAG="-u $(id -u):$(id -g)"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
print_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
print_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ============================================================
# 检查 Docker 镜像是否存在，不存在则构建
# ============================================================
ensure_image() {
    if ! docker image inspect "$SIM_IMAGE" &>/dev/null; then
        print_info "镜像 $SIM_IMAGE 不存在，开始构建..."
        docker build -t "$SIM_IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_DIR"
        print_ok "镜像构建完成"
    else
        print_ok "镜像 $SIM_IMAGE 已存在"
    fi
}

# ============================================================
# 清理旧数据（可选）
# ============================================================
clean_data() {
    if [ -d "$DATA_DIR" ] && [ "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        print_warn "发现已有数据: $DATA_DIR"
        read -p "是否清空旧数据？新数据将追加到已有数据后面。[y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker run --rm -v "${PROJECT_DIR}/data:/data" alpine rm -rf /data/input
            print_info "旧数据已清除"
        fi
    fi
    mkdir -p "$DATA_DIR"
}

# ============================================================
# 运行仿真生成
# ============================================================
run_sim() {
    local args="$@"
    print_info "启动 MuJoCo 仿真数据生成..."
    print_info "参数: $args"
    print_info "输出目录: $DATA_DIR"
    echo ""

    docker run --rm $USER_FLAG \
        -v "$DATA_DIR:/data/input" \
        "$SIM_IMAGE" \
        --output /data/input $args

    echo ""
    local count=$(ls -d "$DATA_DIR"/episode_* 2>/dev/null | wc -l)
    print_ok "完成！共 $count 个 episode 在 $DATA_DIR"
}

# ============================================================
# 主逻辑
# ============================================================
show_help() {
    cat <<EOF
MuJoCo 仿真数据生成工具

用法:
  ./sim/run.sh [模式]

数据生成（Docker 无头模式，无窗口）:
  (无参数)    默认模式：生成 50 个成功的 episode
  --quick     快速测试：生成 5 个 episode
  --full      完整数据集：生成 200 个成功的 episode
  --custom    自定义参数（交互式输入）

可视化（本地运行，弹出窗口）:
  --view      实时观看仿真（MuJoCo 3D 交互窗口）
  --replay    回放已生成的 episode 图像序列
  --replay <path>  回放指定 episode

  --help      显示此帮助

示例:
  ./sim/run.sh --quick          # 先生成数据
  ./sim/run.sh --view           # 看实时仿真
  ./sim/run.sh --replay         # 回放最新 episode

注意：
  数据生成在 Docker 中运行（无需本地安装 MuJoCo），无 GUI 窗口。
  可视化需要本地安装: pip install mujoco opencv-python
EOF
}

case "${1:-default}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --view)
        print_info "启动可视化窗口（本地运行，非 Docker）..."
        cd "$PROJECT_DIR"
        PYTHONPATH="$PROJECT_DIR" python sim/visualize.py --max-steps 300
        exit 0
        ;;
    --replay)
        if [ -z "$2" ]; then
            latest=$(ls -d "$DATA_DIR"/episode_* 2>/dev/null | sort | tail -1)
            if [ -z "$latest" ]; then
                print_warn "没有找到已生成的 episode，请先运行 ./sim/run.sh --quick"
                exit 1
            fi
        else
            latest="$2"
        fi
        print_info "回放 episode: $latest"
        cd "$PROJECT_DIR"
        PYTHONPATH="$PROJECT_DIR" python sim/visualize.py --replay "$latest"
        exit 0
        ;;
    --quick)
        ensure_image
        clean_data
        run_sim --episodes 5 --max-steps 150
        ;;
    --full)
        ensure_image
        clean_data
        run_sim --episodes 200 --success-only --max-steps 300
        ;;
    --custom)
        ensure_image
        clean_data
        echo ""
        read -p "Episode 数量 [50]: " episodes
        episodes=${episodes:-50}
        read -p "每 episode 最大步数 [200]: " max_steps
        max_steps=${max_steps:-200}
        read -p "只保留成功的? [y/N]: " success_only
        read -p "随机种子 [42]: " seed
        seed=${seed:-42}

        extra_args="--episodes $episodes --max-steps $max_steps --seed $seed"
        if [[ $success_only =~ ^[Yy]$ ]]; then
            extra_args="$extra_args --success-only"
        fi
        run_sim $extra_args
        ;;
    default|"")
        ensure_image
        clean_data
        run_sim --episodes 50 --success-only --max-steps 250
        ;;
    *)
        echo "未知选项: $1"
        show_help
        exit 1
        ;;
esac

echo ""
print_info "下一步："
echo "  检查数据:  make inspect"
echo "  转 LeRobot: make convert-lerobot"
echo "  转 OpenPI:  make convert-openpi"
