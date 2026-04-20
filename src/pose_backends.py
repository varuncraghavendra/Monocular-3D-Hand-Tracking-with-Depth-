"""MMPose InterNet backend (baseline).

Runs InterNet (ResNet-50 + 3D heatmap head, ECCV 2020) on each frame,
deduplicates overlapping detections, and assigns hands to left/right slots
with EMA smoothing on centre / size / depth.
"""

from pathlib import Path
import os
import cv2
import numpy as np
from mmpose.apis import init_model, inference_topdown


def _find_mmpose_root() -> Path:
    # Locate the mmpose repo root. Honours MMPOSE_ROOT env var, else walks up
    # from this file looking for configs/hand_3d_keypoint/.
    env = os.environ.get("MMPOSE_ROOT")
    if env:
        root = Path(env).resolve()
        if not (root / "configs").exists():
            raise RuntimeError(
                f"MMPOSE_ROOT={root} has no configs/ directory. "
                "Point MMPOSE_ROOT at your mmpose repo root.")
        return root

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "hand_3d_keypoint").exists():
            return parent

    raise RuntimeError(
        "Could not find mmpose root. Set MMPOSE_ROOT=/path/to/mmpose "
        "and rerun.")


class MMPoseHandBackend:
    DEPTH_W  = 0.4
    MAX_MISS = 12

    def __init__(self, device: str = "cpu", score_thr: float = 0.12,
                 infer_scale: float = 0.6):
        repo   = _find_mmpose_root()
        config = str(repo / "configs/hand_3d_keypoint/internet/interhand3d"
                           "/internet_res50_4xb16-20e_interhand3d-256x256.py")
        ckpt   = str(repo / "checkpoints/res50.pth")

        # MMPose resolves _base_ config imports relative to cwd, so temporarily
        # chdir into the repo root during init_model.
        _orig_cwd = os.getcwd()
        try:
            os.chdir(str(repo))
            self.model = init_model(config, ckpt, device=device)
        finally:
            os.chdir(_orig_cwd)

        self.score_thr   = score_thr
        self.infer_scale = infer_scale

        self._centers = [None, None]
        self._sizes   = [None, None]
        self._depths  = [None, None]
        self._missing = [0, 0]

    def infer(self, frame_bgr: np.ndarray, depth_info: dict | None = None):
        # Full inference step: run model, dedup, attach palm depth if calibrated
        # DA2 is available, then match detections to left/right slots and update EMA.
        raw = self._run(frame_bgr)
        raw = _dedup(raw)

        if depth_info is not None:
            dm  = depth_info["depth_map"]
            est = depth_info["depth_est"]
            for h in raw:
                h["palm_depth"] = est.depth_at_hand(
                    dm, h["keypoints"][:, :2], h["scores"])
        else:
            for h in raw:
                h["palm_depth"] = None

        out = self._match(raw)

        for slot, hand in enumerate(out):
            if hand is not None:
                self._missing[slot] = 0
                self._ema(slot, hand)
            else:
                self._missing[slot] += 1
                if self._missing[slot] >= self.MAX_MISS:
                    self._centers[slot] = self._sizes[slot] = None
                    self._depths[slot]  = None

        return [h for h in out if h is not None]

    def _run(self, frame_bgr: np.ndarray):
        scaled  = cv2.resize(frame_bgr, None,
                             fx=self.infer_scale, fy=self.infer_scale)
        results = inference_topdown(self.model, scaled)
        return self._extract(results)

    def _extract(self, results):
        # Parse MMPose results into a uniform list of hand dicts. InterNet may
        # return 42 keypoints (both hands) or 21 (one hand) per result.
        hands  = []
        sc_min = self.score_thr * 0.5
        for res in results:
            pred = getattr(res, "pred_instances", None)
            if pred is None:
                continue
            kpa = np.asarray(getattr(pred, "keypoints",       []))
            sca = np.asarray(getattr(pred, "keypoint_scores", []))
            if kpa.ndim < 3 or kpa.shape[0] == 0:
                continue
            kp = kpa[0].astype(np.float32)
            sc = sca[0].astype(np.float32)
            if sc.max() > 1.0:
                sc /= 255.0
            kp[:, 0] /= self.infer_scale
            kp[:, 1] /= self.infer_scale
            n = kp.shape[0]
            if n == 42:
                for start, side in [(0, "right"), (21, "left")]:
                    hkp = kp[start:start+21].copy()
                    hsc = sc[start:start+21].copy()
                    if np.mean(hsc) >= sc_min:
                        hands.append(_mkhand(hkp, hsc, side))
            elif n == 21 and np.mean(sc) >= sc_min:
                hands.append(_mkhand(kp, sc, "unknown"))
        return hands

    def _match(self, raw: list) -> list:
        # Assign detections to slot 0 (right) / slot 1 (left). First honour any
        # hand_side label from the model, then fill remaining slots by best score.
        out, used = [None, None], set()
        for j, h in enumerate(raw):
            slot = {"right": 0, "left": 1}.get(h["hand_side"])
            if slot is not None and out[slot] is None:
                out[slot] = h
                used.add(j)
        for slot in range(2):
            if out[slot] is not None:
                continue
            best_s, best_j = -1e9, None
            for j, h in enumerate(raw):
                if j in used:
                    continue
                s = self._slot_score(h, slot)
                if s > best_s:
                    best_s, best_j = s, j
            if (best_j is not None and
                    raw[best_j]["mean_score"] >= self.score_thr):
                out[slot] = raw[best_j]
                used.add(best_j)
        return out

    def _slot_score(self, hand: dict, slot: int) -> float:
        # Score = confidence penalised by distance from slot's last known centre
        # and, if available, by depth discontinuity from the slot's last depth.
        conf = hand["mean_score"]
        if self._centers[slot] is None:
            return conf
        sz   = self._sizes[slot] or 150.0
        dist = np.linalg.norm(
            hand["keypoints"][:, :2].mean(0) - self._centers[slot])
        score = conf - 0.5 * (dist / sz)
        if (self._depths[slot] is not None and
                hand.get("palm_depth") is not None):
            diff  = abs(hand["palm_depth"] - self._depths[slot])
            score -= self.DEPTH_W * max(0.0, diff - 0.15)
        return score

    def _ema(self, slot: int, hand: dict, alpha: float = 0.35):
        # Update per-slot running estimates of centre, size, and palm depth.
        kp = hand["keypoints"][:, :2]
        nc = kp.mean(0)
        d  = kp.max(0) - kp.min(0)
        ns = float(max(d[0], d[1], 100.0))
        if self._centers[slot] is None:
            self._centers[slot] = nc
            self._sizes[slot]   = ns
        else:
            self._centers[slot] = alpha*nc + (1-alpha)*self._centers[slot]
            self._sizes[slot]   = alpha*ns + (1-alpha)*self._sizes[slot]
        pd = hand.get("palm_depth")
        if pd is not None:
            if self._depths[slot] is None:
                self._depths[slot] = pd
            else:
                self._depths[slot] = alpha*pd + (1-alpha)*self._depths[slot]


def _mkhand(kp: np.ndarray, sc: np.ndarray, side: str) -> dict:
    if kp.shape[1] == 2:
        kp = np.concatenate([kp, np.zeros((21, 1), np.float32)], axis=1)
    return {"keypoints": kp[:, :3].copy(), "scores": sc.copy(),
            "mean_score": float(np.mean(sc)), "hand_side": side,
            "palm_depth": None}


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter == 0:
        return 0.0
    return inter / ((ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter + 1e-9)


def _dedup(hands: list, thr: float = 0.5) -> list:
    # NMS over hand bboxes: keep the higher-scoring detection when two overlap.
    if len(hands) <= 1:
        return hands
    boxes = [(*h["keypoints"][:, :2].min(0), *h["keypoints"][:, :2].max(0))
             for h in hands]
    keep, sup = [], set()
    for i in range(len(hands)):
        if i in sup:
            continue
        keep.append(i)
        for j in range(i+1, len(hands)):
            if j in sup:
                continue
            if _iou(boxes[i], boxes[j]) > thr:
                if hands[j]["mean_score"] > hands[keep[-1]]["mean_score"]:
                    keep[-1] = j
                sup.add(j)
    return [hands[k] for k in keep]
