"""Depth Anything V2 + Depth Fusion (EXT 1).

Runs DA2 inference and converts the raw output to absolute metres. Supports both
relative checkpoints (disparity output, need calibration) and metric checkpoints
(already in metres).
"""

from pathlib import Path
import sys
import cv2
import numpy as np
import torch

TARGET_M   = 0.40
DA2_H = DA2_W = 518

SETUP_INSTRUCTIONS = """
Depth Anything V2 setup required.

Step 1 — Clone the repo into your checkpoint directory:
  cd {model_path}
  git clone https://github.com/DepthAnything/Depth-Anything-V2

Step 2 — Install DA2 dependencies (Python 3.10 compatible):
  pip install torch torchvision timm

Step 3 — Download a checkpoint and place it in {model_path}/ as
  depth_anything_v2_{encoder}.pth

  vits (fast, ~100MB):
    https://huggingface.co/depth-anything/Depth-Anything-V2-Small
  vitb (~400MB):
    https://huggingface.co/depth-anything/Depth-Anything-V2-Base
  vitl (best, ~1.3GB):
    https://huggingface.co/depth-anything/Depth-Anything-V2-Large

Step 4 — Run:
  python3 scripts/run_robot_learning_gui.py --device cuda --da2-encoder {encoder}
"""


def _load_da2(model_path: Path, device: str, encoder: str):
    # Import DepthAnythingV2 from the local repo clone and load the checkpoint
    # manually. Avoids the pip package (which requires Python 3.12).
    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    if encoder not in model_configs:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose from: vits, vitb, vitl")

    ckpt      = model_path / f"depth_anything_v2_{encoder}.pth"
    repo_path = model_path / "Depth-Anything-V2"

    instructions = SETUP_INSTRUCTIONS.format(
        model_path=model_path,
        encoder=encoder,
    )

    if not repo_path.exists():
        raise RuntimeError(f"DA2 repo not found at {repo_path}\n{instructions}")

    if not ckpt.exists():
        raise RuntimeError(f"DA2 checkpoint not found: {ckpt}\n{instructions}")

    repo_str = str(repo_path)
    inserted = False
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
        inserted = True
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        cfg   = model_configs[encoder]
        model = DepthAnythingV2(**cfg)
        state = torch.load(str(ckpt), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        print(f"DA2 loaded: encoder={encoder} ckpt={ckpt.name}")
        return model
    except ImportError as e:
        raise RuntimeError(
            f"DA2 import failed from {repo_path}: {e}\n"
            "Make sure the full repo is cloned.\n" + instructions) from e
    except Exception as e:
        raise RuntimeError(f"DA2 load failed: {e}\n{instructions}") from e
    finally:
        if inserted and repo_str in sys.path:
            sys.path.remove(repo_str)


class DepthFusion:
    # Converts raw DA2 output to absolute metres.
    # Relative mode: raw is disparity, so depth = 1/raw then scaled by calibration trim.
    # Metric mode:   raw is already metres, optionally trimmed by calibration.

    def __init__(self, min_depth: float = 0.1, max_depth: float = 3.0,
                 is_relative: bool = False):
        self.min_depth   = min_depth
        self.max_depth   = max_depth
        self.is_relative = is_relative
        self.scale_trim  = None

    def fuse(self, raw: np.ndarray) -> np.ndarray:
        if self.is_relative:
            depth = 1.0 / np.clip(raw, 1e-6, None)
        else:
            depth = raw.copy()
        if self.scale_trim is not None:
            depth = depth * self.scale_trim
        return np.clip(depth, self.min_depth, self.max_depth).astype(np.float32)

    def calibrate(self, raw_da2: np.ndarray, palm_kp2d: np.ndarray) -> bool:
        # Sample raw DA2 at the palm and compute a scale factor so the fused
        # output reads TARGET_M (0.40 m) at that point.
        H, W  = raw_da2.shape
        cx    = int(np.clip(palm_kp2d[:, 0].mean(), 0, W - 1))
        cy    = int(np.clip(palm_kp2d[:, 1].mean(), 0, H - 1))
        r     = 25
        patch = raw_da2[max(0, cy-r):min(H, cy+r),
                        max(0, cx-r):min(W, cx+r)]
        if patch.size == 0:
            print("Calibration failed: empty patch at palm.")
            return False

        raw_val = float(np.median(patch))
        if raw_val < 1e-6:
            print("Calibration failed: zero raw value at palm.")
            return False

        if self.is_relative:
            depth_at_palm = 1.0 / raw_val
        else:
            depth_at_palm = raw_val

        self.scale_trim = TARGET_M / depth_at_palm
        mode = "relative" if self.is_relative else "metric"
        print(f"Calibrated ({mode}): palm depth {depth_at_palm:.3f} m "
              f"-> target {TARGET_M:.2f} m, trim={self.scale_trim:.4f}")
        return True

    def reset(self):
        self.scale_trim = None


class DepthEstimator:
    def __init__(self, model_path: str, device: str = "cpu",
                 encoder: str = "vitl",
                 min_depth: float = 0.1, max_depth: float = 3.0):
        self.device  = device
        self.path    = Path(model_path)
        self.encoder = encoder
        self._prev   = None

        self.model = _load_da2(self.path, self.device, encoder)

        ckpt_name   = f"depth_anything_v2_{encoder}"
        is_relative = not any(k in ckpt_name for k in ("metric", "indoor", "outdoor"))
        self.fusion = DepthFusion(min_depth=min_depth, max_depth=max_depth,
                                  is_relative=is_relative)
        mode = "relative (press 'c' to calibrate)" if is_relative else "metric (absolute metres)"
        print(f"Depth estimator: encoder={encoder}, mode={mode}")

    @property
    def calib_scale(self):
        return self.fusion.scale_trim

    @calib_scale.setter
    def calib_scale(self, v):
        self.fusion.scale_trim = v

    @property
    def enabled(self):
        return True

    def _da2_infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Forward pass: normalise to ImageNet stats, resize to 518x518 (DA2
        # requires input dims divisible by 14), run model, resize back.
        H, W = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp  = cv2.resize(rgb, (DA2_W, DA2_H)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std  = np.array([0.229, 0.224, 0.225], np.float32)
        inp  = (inp - mean) / std
        x    = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        if isinstance(out, dict):
            raw = out.get("metric_depth") or list(out.values())[0]
        else:
            raw = out
        raw = raw.squeeze().cpu().numpy().astype(np.float32)
        return cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)

    def estimate(self, frame_bgr: np.ndarray):
        # Returns (smoothed_depth_m, raw_da2_output). Pass raw_da2_output
        # into calibrate(), use smoothed_depth_m for display and sampling.
        raw_output = self._da2_infer(frame_bgr)
        metres     = self.fusion.fuse(raw_output)

        if self._prev is None:
            self._prev = metres.copy()
        else:
            self._prev = (0.8 * self._prev + 0.2 * metres).astype(np.float32)

        return self._prev.copy(), raw_output

    def calibrate(self, raw_da2: np.ndarray, palm_kp2d: np.ndarray) -> bool:
        return self.fusion.calibrate(raw_da2, palm_kp2d)

    def sample(self, depth_map: np.ndarray, xy, patch: int = 10) -> float:
        H, W = depth_map.shape
        x = int(np.clip(round(float(xy[0])), 0, W - 1))
        y = int(np.clip(round(float(xy[1])), 0, H - 1))
        return float(np.median(
            depth_map[max(0, y-patch):min(H, y+patch+1),
                      max(0, x-patch):min(W, x+patch+1)]))

    def depth_at_hand(self, depth_map: np.ndarray,
                      kp2d: np.ndarray, sc: np.ndarray,
                      joint_thr: float = 0.12):
        # Median depth at the palm centroid (wrist + four MCP joints).
        palm_ids = [20, 7, 11, 15, 19]
        valid    = [j for j in palm_ids if sc[j] > joint_thr]
        if not valid:
            return None
        cx = float(np.mean(kp2d[valid, 0]))
        cy = float(np.mean(kp2d[valid, 1]))
        return self.sample(depth_map, (cx, cy), patch=12)

    @staticmethod
    def colorize(depth: np.ndarray) -> np.ndarray:
        d = depth - depth.min()
        d = (d / max(depth.max() - depth.min(), 1e-6) * 255).astype(np.uint8)
        return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)
