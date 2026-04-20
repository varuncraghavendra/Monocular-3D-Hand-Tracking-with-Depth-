# Robot Learning Hand Pipeline

Real-time 3D hand pose estimation for robot learning, built on **MMPose InterNet** and **Depth Anything V2**. Runs on a single monocular webcam and outputs 21 joints per hand in absolute metric camera space, together with semantic gestures (`GRASP`, `OPEN PALM`, `PINCH`).

Submitted as the final project for **PRCV, Spring 2026** — *Varun Raghavendra, MS Robotics, Northeastern University Boston.*

---

![Alt text](Untitled design.gif)

## Overview

The pipeline combines a strong 3D-hand baseline (InterNet, ECCV 2020) with three extensions that make it usable for robot teleoperation from a single RGB webcam:

| Stage | Module | Role |
|---|---|---|
| **Baseline** | InterNet (ResNet-50 + 3D heatmap head) | 21 keypoints per hand, handedness, root depth |
| **EXT 1** | Depth Anything V2 + depth fusion | Per-pixel metric depth, scale-aligned to the palm |
| **EXT 2** | 1 Euro filter | Per-joint adaptive low-pass smoothing |
| **EXT 3** | Gesture abstraction | Digit distances + palm normals → `GRASP` / `OPEN PALM` / `PINCH` |

On top of that, the pipeline writes a full **session report PNG** on exit (gesture dwell, depth timelines, pinch history, FPS).

---

## Architecture

```
Webcam (RGB, 30 fps)
        │
        ├──► Depth Anything V2  ──► DepthFusion  ──► metric depth map (m)
        │                                                     │
        ▼                                                     │
MMPose preprocessing  →  ResNet-50 backbone  →  InterNet head │
(256×256 tensor)          (2048-d features)     3D heatmap    │
                                                root depth Z  │
        │                                       handedness    │
        ▼                                                     │
Hand3DHeatmap codec  →  21 × (x, y, z) per hand  ◄────────────┘
        │                     (camera space)
        ▼
1 Euro filter (per joint, per axis)  →  jitter-free keypoints
        │
        ▼
Gesture abstraction  →  { GRASP, OPEN PALM, PINCH }  + meta (pinch_norm,
                                                            extension,
                                                            palm normal)
```

---

## File Map

| File | Role |
|---|---|
| `src/camera.py` | Threaded OpenCV capture |
| `src/pose_backends.py` | MMPose InterNet backend (baseline) |
| `src/depth_estimator.py` | Depth Anything V2 loader + DepthFusion (EXT 1) |
| `src/one_euro_filter.py` | 1 Euro adaptive low-pass filter (EXT 2) |
| `src/gesture_abstraction.py` | Gesture classifier (EXT 3) |
| `src/pipeline.py` | End-to-end runtime + live GUI + session report |
| `scripts/run_robot_learning_gui.py` | Entry point / CLI |

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision mmpose opencv-python matplotlib timm
```

Tested on Python 3.10 with CUDA 11.8 and on CPU-only machines.

### 2. Depth Anything V2 checkpoint

Clone the DA2 repo inside `checkpoints/depth_anything_v2/` and drop a checkpoint next to it:

```bash
mkdir -p checkpoints/depth_anything_v2
cd checkpoints/depth_anything_v2
git clone https://github.com/DepthAnything/Depth-Anything-V2
```

Then download one of the checkpoints from the Hugging Face mirrors below and place it at `checkpoints/depth_anything_v2/depth_anything_v2_<encoder>.pth`.

| Encoder | File | Size | Source |
|---|---|---|---|
| `vits` | `depth_anything_v2_vits.pth` | ~100 MB | [Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) |
| `vitb` | `depth_anything_v2_vitb.pth` | ~400 MB | [Depth-Anything-V2-Base](https://huggingface.co/depth-anything/Depth-Anything-V2-Base) |
| `vitl` | `depth_anything_v2_vitl.pth` | ~1.3 GB | [Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) |

Relative vs metric: any file without `metric` / `indoor` / `outdoor` in the name produces disparity and requires the 3-second palm calibration described below. Metric checkpoints (e.g. `depth_anything_v2_metric_indoor_vitl.pth`) output absolute metres directly; calibration is then optional.

### 3. InterNet checkpoint

Place MMPose's InterNet weights at `checkpoints/res50.pth`, with the matching config at
`configs/hand_3d_keypoint/internet/interhand3d/internet_res50_4xb16-20e_interhand3d-256x256.py`.

---

## Run

```bash
# Default: CPU, vitl encoder
python scripts/run_robot_learning_gui.py

# Faster: small encoder on CPU
python scripts/run_robot_learning_gui.py --da2-encoder vits

# GPU: large encoder with full precision
python scripts/run_robot_learning_gui.py \
    --device cuda:0 \
    --da2-encoder vitl \
    --depth-model checkpoints/depth_anything_v2
```

### Keyboard shortcuts

| Key | Action |
|---|---|
| `c` | Start 3-second depth calibration (hold open palm at 40 cm) |
| `r` | Reset depth calibration |
| `ESC` | Quit and save session report |

### Depth calibration

1. Press `c`.
2. Hold your open palm **exactly 40 cm** from the camera.
3. Wait for the 3-second countdown.
4. Depth readings switch from `m*` (uncalibrated) to `m` (calibrated).

---

## Dataset

Trained on **InterHand2.6M** (ECCV 2020) — the first large-scale real-captured dataset with accurate 3D interacting hand poses: 2.6 M labelled frames, 80–140 cameras per capture, 21 keypoints per hand. The exact 21-joint layout from that dataset is used throughout this project.

---

## Applications

Teleoperated robotic manipulation, imitation learning from human demonstrations, telesurgery, VR / AR input, and accessible gesture control.

---

## Acknowledgements

- Moon et al., *InterHand2.6M* (ECCV 2020)
- Yang et al., *Depth Anything V2* (NeurIPS 2024)
- Casiez et al., *1 Euro Filter* (CHI 2012)
- MMPose, OpenMMLab
