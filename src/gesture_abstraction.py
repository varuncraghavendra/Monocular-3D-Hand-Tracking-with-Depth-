"""Gesture Abstraction Layer (EXT 3).

Classifies 21-keypoint hand poses into three semantic states: GRASP, OPEN PALM,
PINCH. Uses Euclidean digit distances, a combined 2D+3D finger extension ratio,
and the palm orientation normal.
"""

from __future__ import annotations
from collections import deque
import numpy as np

DEBUG = False

# InterHand2.6M keypoint layout (21 per hand)
WRIST = 20
FINGER_CHAINS_IDX = [
    [20,  3,  2,  1,  0],
    [20,  7,  6,  5,  4],
    [20, 11, 10,  9,  8],
    [20, 15, 14, 13, 12],
    [20, 19, 18, 17, 16],
]
FINGER_TIPS = [0,  4,  8, 12, 16]
FINGER_MCPS = [3,  7, 11, 15, 19]

_PALM_IDX_MCP  = 7
_PALM_PINK_MCP = 19


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-9)


def _palm_normal(kp: np.ndarray) -> np.ndarray:
    # Palm-plane normal via cross product of two palm vectors (wrist->index MCP,
    # wrist->pinky MCP). Sign depends on handedness.
    w  = kp[WRIST,          :3]
    vi = kp[_PALM_IDX_MCP,  :3] - w
    vp = kp[_PALM_PINK_MCP, :3] - w
    return _unit(np.cross(vi, vp))


class GestureAbstractor:
    PINCH_THR        = 0.55
    CURL_THR         = 1.2
    OPEN_EACH_THR    = 1.3
    OPEN_MEAN_THR    = 1.5
    OPEN_MIN_FINGERS = 2
    MIN_OPEN_FRAMES  = 2

    def __init__(self, history: int = 5):
        self.history      = deque(maxlen=history)
        self._current     = "GRASP"
        self._open_streak = 0

    def _extension_ratios(self, kp: np.ndarray, sc: np.ndarray,
                          score_thr: float) -> list[float | None]:
        # For each non-thumb finger, compute max(2D pixel ratio, 3D ratio) of
        # tip->wrist over MCP->wrist. Takes whichever axis carries the extension
        # so both lateral and forward-facing openings are detected.
        wrist_2d = kp[WRIST, :2]
        wrist_3d = kp[WRIST, :3]
        ratios   = []

        for fi in range(1, 5):
            tip_i = FINGER_TIPS[fi]
            mcp_i = FINGER_MCPS[fi]

            if sc[tip_i] < score_thr or sc[mcp_i] < score_thr:
                ratios.append(None)
                continue

            d_tip_2d = np.linalg.norm(kp[tip_i, :2] - wrist_2d)
            d_mcp_2d = np.linalg.norm(kp[mcp_i, :2] - wrist_2d)
            ratio_2d = (d_tip_2d / d_mcp_2d) if d_mcp_2d > 1e-3 else 0.0

            has_z = kp.shape[1] >= 3 and np.any(kp[:, 2] != 0)
            if has_z:
                d_tip_3d = np.linalg.norm(kp[tip_i, :3] - wrist_3d)
                d_mcp_3d = np.linalg.norm(kp[mcp_i, :3] - wrist_3d)
                ratio_3d = (d_tip_3d / d_mcp_3d) if d_mcp_3d > 1e-3 else 0.0
            else:
                ratio_3d = 0.0

            ratios.append(float(max(ratio_2d, ratio_3d)))

        return ratios

    def _digit_distances(self, kp: np.ndarray, sc: np.ndarray,
                         score_thr: float) -> dict:
        # Thumb-tip to each fingertip distance, normalised by palm width.
        palm_w = np.linalg.norm(
            kp[_PALM_IDX_MCP, :2] - kp[_PALM_PINK_MCP, :2])
        palm_w = max(palm_w, 1e-3)
        thumb  = kp[FINGER_TIPS[0], :2]
        names  = ["thumb_index", "thumb_middle", "thumb_ring", "thumb_pinky"]
        dists  = {}
        for i, name in enumerate(names):
            tip_i = FINGER_TIPS[i + 1]
            if sc[FINGER_TIPS[0]] < score_thr or sc[tip_i] < score_thr:
                dists[name] = None
            else:
                dists[name] = float(
                    np.linalg.norm(kp[tip_i, :2] - thumb) / palm_w)
        return dists

    def _pinch_norm(self, kp: np.ndarray, sc: np.ndarray,
                    score_thr: float) -> float | None:
        if sc[FINGER_TIPS[0]] < score_thr or sc[FINGER_TIPS[1]] < score_thr:
            return None
        palm_w = np.linalg.norm(
            kp[_PALM_IDX_MCP, :2] - kp[_PALM_PINK_MCP, :2])
        if palm_w < 1e-3:
            return None
        return float(np.linalg.norm(
            kp[FINGER_TIPS[0], :2] - kp[FINGER_TIPS[1], :2]) / palm_w)

    def classify(self, keypoints: np.ndarray, scores: np.ndarray,
                 score_thr: float = 0.12) -> tuple[str, dict]:
        # State machine priority: PINCH > OPEN PALM > GRASP. Uses per-frame
        # streaks and majority-vote history to reduce label flicker.
        kp = np.asarray(keypoints, dtype=np.float32)
        sc = np.asarray(scores,    dtype=np.float32)

        _empty_meta = {
            "digit_distances": {}, "extension_ratios": [],
            "palm_normal": np.zeros(3), "pinch_norm": None,
            "n_extended": 0, "n_curled": 0,
            "mean_ratio": 0.0, "confidence": 1.0,
        }

        if kp.shape[0] < 21 or np.mean(sc) < score_thr:
            self.history.clear()
            self._current     = "GRASP"
            self._open_streak = 0
            return "GRASP", _empty_meta

        ext_ratios  = self._extension_ratios(kp, sc, score_thr)
        digit_dists = self._digit_distances(kp, sc, score_thr)
        palm_normal = _palm_normal(kp)
        pinch_n     = self._pinch_norm(kp, sc, score_thr)

        valid_ratios = [r for r in ext_ratios if r is not None]
        mean_ratio   = float(np.mean(valid_ratios)) if valid_ratios else 0.0
        n_extended   = sum(1 for r in valid_ratios if r > self.OPEN_EACH_THR)
        n_curled     = sum(1 for r in valid_ratios if r < self.CURL_THR)

        if DEBUG:
            names = ["idx", "mid", "rng", "pnk"]
            ratio_str = "  ".join(
                f"{n}:{r:.2f}" if r is not None else f"{n}:---"
                for n, r in zip(names, ext_ratios))
            pinch_str = f"{pinch_n:.2f}" if pinch_n is not None else "---"
            print(f"gesture  {ratio_str}  mean:{mean_ratio:.2f}  "
                  f"ext:{n_extended}/4  curl:{n_curled}/4  pinch:{pinch_str}")

        is_pinch = (pinch_n is not None and pinch_n < self.PINCH_THR)
        is_open = (
            len(valid_ratios) >= 2 and
            mean_ratio   >= self.OPEN_MEAN_THR and
            n_extended   >= self.OPEN_MIN_FINGERS
        )

        if is_pinch:
            raw = "PINCH"
            self._open_streak = 0
        elif is_open:
            self._open_streak += 1
            raw = ("OPEN PALM"
                   if self._open_streak >= self.MIN_OPEN_FRAMES
                   else self._current)
        else:
            self._open_streak = 0
            raw = "GRASP"

        # Hysteresis: stay in OPEN PALM until a curl or pinch forces an exit.
        if self._current == "OPEN PALM" and not is_pinch and n_curled == 0:
            raw = "OPEN PALM"

        self.history.append(raw)
        vals  = list(self.history)
        label = max(set(vals), key=vals.count)
        conf  = vals.count(label) / len(vals)
        self._current = label

        return label, {
            "digit_distances":  digit_dists,
            "extension_ratios": ext_ratios,
            "palm_normal":      palm_normal,
            "pinch_norm":       pinch_n,
            "n_extended":       n_extended,
            "n_curled":         n_curled,
            "mean_ratio":       mean_ratio,
            "confidence":       float(conf),
        }
