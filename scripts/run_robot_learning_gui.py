"""Entry point for the Robot Learning Hand Pipeline.

Keys in the GUI: c = calibrate, r = reset, ESC = quit. A session report PNG
is written to --report-dir on exit.
"""
import argparse, sys
from pathlib import Path

THIS         = Path(__file__).resolve()
PROJECT_ROOT = THIS.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import RobotLearningHandPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Robot Learning Hand Pipeline")
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--score-thr",   type=float, default=0.08)
    parser.add_argument("--depth-model",
                        default=str(PROJECT_ROOT/"checkpoints"/"depth_anything_v2"))
    parser.add_argument("--da2-encoder", default="vitl",
                        choices=["vits","vitb","vitl"])
    parser.add_argument("--infer-scale", type=float, default=0.55)
    parser.add_argument("--report-dir",
                        default=str(PROJECT_ROOT),
                        help="directory to save session_report_*.png")
    args = parser.parse_args()

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Depth model  : {args.depth_model}")
    print(f"DA2 encoder  : {args.da2_encoder}")
    print(f"Device       : {args.device}")
    print(f"Report dir   : {args.report_dir}")

    pipeline = RobotLearningHandPipeline(
        device      = args.device,
        score_thr   = args.score_thr,
        depth_model = args.depth_model,
        da2_encoder = args.da2_encoder,
        infer_scale = args.infer_scale,
        report_dir  = args.report_dir,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
