"""CLI 入口 - 命令行工具用于数据转换和检查。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_convert(args: argparse.Namespace) -> None:
    """执行数据转换。"""
    from vla_data_adapter.normalization import NormalizationConfig
    from vla_data_adapter.pipeline import Pipeline, PipelineConfig

    source = _build_source_adapter(args.source, args.input_dir, args)
    exporter = _build_exporter(args.export_format, args.output_dir, args)

    norm_config = NormalizationConfig(
        target_frequency=args.target_freq,
        remove_failed_episodes=args.success_only,
        image_resize=tuple(args.image_size) if args.image_size else None,
    )

    pipeline_config = PipelineConfig(normalization=norm_config)
    pipeline = Pipeline(pipeline_config)
    output_path = pipeline.run(source, exporter)
    logger.info(f"Done! Output at: {output_path}")


def cmd_inspect(args: argparse.Namespace) -> None:
    """检查数据集信息。"""
    from vla_data_adapter.normalization import DataQualityChecker
    from vla_data_adapter.pipeline import Pipeline, PipelineConfig

    source = _build_source_adapter(args.source, args.input_dir, args)
    pipeline = Pipeline(PipelineConfig())
    episodes = pipeline.load_and_normalize(source)

    stats = DataQualityChecker.compute_statistics(episodes)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.verbose:
        for ep in episodes[:5]:
            sync = DataQualityChecker.check_sync(ep)
            print(f"\n  Episode {ep.episode_id}: {ep.num_frames} frames, "
                  f"duration={ep.duration:.2f}s, sync={sync}")


def cmd_validate(args: argparse.Namespace) -> None:
    """验证已导出的数据集。"""
    exporter = _build_exporter(args.export_format, args.output_dir, args)
    output_dir = Path(args.output_dir)

    if exporter.validate_export(output_dir):
        print("Validation PASSED")
    else:
        print("Validation FAILED")
        sys.exit(1)


def _build_source_adapter(source_type: str, input_dir: str, args: argparse.Namespace):
    """根据类型构建数据源适配器。"""
    from vla_data_adapter.sources import (
        DroidAdapter,
        DroidConfig,
        LiberoAdapter,
        LiberoConfig,
        MujocoSimAdapter,
        MujocoSimConfig,
        RealRobotConfig,
        RealRobotTeleopAdapter,
    )

    data_dir = Path(input_dir)
    max_episodes = getattr(args, "max_episodes", None)

    adapters = {
        "droid": lambda: DroidAdapter(DroidConfig(data_dir=data_dir, max_episodes=max_episodes)),
        "libero": lambda: LiberoAdapter(LiberoConfig(data_dir=data_dir, max_episodes=max_episodes)),
        "real_robot": lambda: RealRobotTeleopAdapter(RealRobotConfig(data_dir=data_dir, max_episodes=max_episodes)),
        "mujoco_sim": lambda: MujocoSimAdapter(MujocoSimConfig(data_dir=data_dir, max_episodes=max_episodes)),
    }

    if source_type not in adapters:
        raise ValueError(f"Unknown source type: {source_type}. Available: {list(adapters.keys())}")

    return adapters[source_type]()


def _build_exporter(export_format: str, output_dir: str, args: argparse.Namespace):
    """根据格式构建导出适配器。"""
    from vla_data_adapter.exporters import (
        GR00TAdapter,
        GR00TExportConfig,
        LeRobotAdapter,
        LeRobotExportConfig,
        OpenPIAdapter,
        OpenPIExportConfig,
    )

    out_path = Path(output_dir)

    exporters = {
        "lerobot": lambda: LeRobotAdapter(LeRobotExportConfig(output_dir=out_path)),
        "openpi": lambda: OpenPIAdapter(OpenPIExportConfig(output_dir=out_path)),
        "groot": lambda: GR00TAdapter(GR00TExportConfig(output_dir=out_path)),
    }

    if export_format not in exporters:
        raise ValueError(f"Unknown export format: {export_format}. Available: {list(exporters.keys())}")

    return exporters[export_format]()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLA Training Data Adapter - 统一机器人数据转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 将 DROID 数据转换为 LeRobot 格式
  vla-adapter convert --source droid --input ./droid_data --format lerobot --output ./output

  # 检查数据集信息
  vla-adapter inspect --source real_robot --input ./teleop_data

  # 验证导出结果
  vla-adapter validate --format lerobot --output ./output
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # convert
    p_convert = subparsers.add_parser("convert", help="转换数据格式")
    p_convert.add_argument("--source", "-s", required=True, choices=["droid", "libero", "real_robot", "mujoco_sim"])
    p_convert.add_argument("--input", "-i", required=True, dest="input_dir")
    p_convert.add_argument("--format", "-f", required=True, dest="export_format", choices=["lerobot", "openpi", "groot"])
    p_convert.add_argument("--output", "-o", required=True, dest="output_dir")
    p_convert.add_argument("--max-episodes", type=int, default=None)
    p_convert.add_argument("--target-freq", type=float, default=None)
    p_convert.add_argument("--success-only", action="store_true")
    p_convert.add_argument("--image-size", type=int, nargs=2, default=None)
    p_convert.set_defaults(func=cmd_convert)

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="检查数据集信息")
    p_inspect.add_argument("--source", "-s", required=True, choices=["droid", "libero", "real_robot", "mujoco_sim"])
    p_inspect.add_argument("--input", "-i", required=True, dest="input_dir")
    p_inspect.add_argument("--max-episodes", type=int, default=None)
    p_inspect.add_argument("--verbose", "-v", action="store_true")
    p_inspect.set_defaults(func=cmd_inspect)

    # validate
    p_validate = subparsers.add_parser("validate", help="验证导出结果")
    p_validate.add_argument("--format", "-f", required=True, dest="export_format", choices=["lerobot", "openpi", "groot"])
    p_validate.add_argument("--output", "-o", required=True, dest="output_dir")
    p_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
