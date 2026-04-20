"""Real-time 3D hand tracking pipeline for robot learning.

Wires all stages together: threaded camera -> DA2 depth -> MMPose InterNet ->
1 Euro filter -> gesture abstraction, with a live OpenCV overlay, a Matplotlib
3D preview, and a session report saved on exit.
"""

import os, time, datetime
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa

from .camera              import ThreadedCamera
from .depth_estimator     import DepthEstimator
from .gesture_abstraction import GestureAbstractor, FINGER_CHAINS_IDX, WRIST, FINGER_TIPS
from .one_euro_filter     import OneEuroFilter
from .pose_backends       import MMPoseHandBackend

_BGR = [(203,192,255), (50,100,255), (50,205,50), (0,165,255), (147,112,219)]
_RGB = [(r/255, g/255, b/255) for (b, g, r) in _BGR]
_PALM_BGR = (220, 220, 220)

_GESTURE_COL = {
    "GRASP":     (80,  80, 220),
    "OPEN PALM": (60, 200,  60),
    "PINCH":     (0,  200, 200),
}
_GESTURE_COL_MPL = {
    "GRASP":     "#5050DC",
    "OPEN PALM": "#3CC83C",
    "PINCH":     "#00C8C8",
}

_MEAN_THR  = 0.18
_JOINT_THR = 0.12
_FRAC_THR  = 0.55

CAM_W, CAM_H   = 960, 720
PLOT_W, PLOT_H = 640, 720
CALIB_SECS     = 3

DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 2.5


class SessionStats:
    """Per-frame accumulator for session metrics; saves a PNG report on exit."""

    GESTURES = ["GRASP", "OPEN PALM", "PINCH"]

    def __init__(self):
        self.t_start      = time.time()
        self.total_frames = 0
        self.det_frames   = 0

        self.fps_log = []

        self.gesture_log  = [[], []]
        self.depth_log    = [[], []]
        self.pinch_log    = [[], []]
        self.ext_log      = [[], []]
        self.normal_z_log = [[], []]

    def update(self, fps: float, hands: list, metas: list, depth_vals: list):
        # Called once per frame. hands/metas/depth_vals are same-order lists
        # indexed by hand slot (0=right, 1=left).
        t = time.time() - self.t_start
        self.total_frames += 1
        self.fps_log.append(fps)

        if hands:
            self.det_frames += 1

        for slot, (hand, meta, dv) in enumerate(
                zip(hands, metas, depth_vals)):
            if slot >= 2:
                break
            label = meta.get("_label", "GRASP")
            self.gesture_log[slot].append((t, label))

            if dv is not None:
                self.depth_log[slot].append((t, dv))

            pn = meta.get("pinch_norm")
            if pn is not None:
                self.pinch_log[slot].append((t, pn))

            er = meta.get("mean_ratio")
            if er is not None and er > 0:
                self.ext_log[slot].append((t, er))

            nz = meta.get("palm_normal")
            if nz is not None and not np.allclose(nz, 0):
                self.normal_z_log[slot].append((t, abs(float(nz[2]))))

    @staticmethod
    def _gesture_dwell(log):
        if not log:
            return {g: 0.0 for g in SessionStats.GESTURES}, 0
        dwell = defaultdict(float)
        transitions = 0
        prev_label  = log[0][1]
        prev_t      = log[0][0]
        for t, label in log[1:]:
            dwell[prev_label] += t - prev_t
            if label != prev_label:
                transitions += 1
            prev_label = label
            prev_t     = t
        return dict(dwell), transitions

    @staticmethod
    def _depth_stats(log):
        if not log:
            return None
        vals = [v for _, v in log]
        return {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
        }

    def save_report(self, out_dir: str = "."):
        # Build a single PNG with session summary, FPS timeline, per-hand
        # gesture timelines, depth timelines, and pinch-distance timelines.
        duration  = time.time() - self.t_start
        det_rate  = self.det_frames / max(self.total_frames, 1) * 100
        mean_fps  = float(np.mean(self.fps_log)) if self.fps_log else 0

        slot_names = ["Right hand", "Left hand"]

        fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
        fig.suptitle(
            f"Robot Learning Hand Pipeline — Session Report\n"
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            color="white", fontsize=14, fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(3, 4, figure=fig,
                               hspace=0.55, wspace=0.38,
                               left=0.06, right=0.97,
                               top=0.91, bottom=0.06)

        _ax_style = dict(facecolor="#0f0f1a")

        ax_sum = fig.add_subplot(gs[0, 0:2], **_ax_style)
        ax_sum.axis("off")
        summary_lines = [
            f"Session duration   : {duration:.1f} s  ({duration/60:.1f} min)",
            f"Total frames       : {self.total_frames}",
            f"Frames with hands  : {self.det_frames}  ({det_rate:.1f}%)",
            f"Average FPS        : {mean_fps:.1f}",
            "",
        ]
        for si in range(2):
            dwell, trans = self._dwell_and_trans(si)
            ds = self._depth_stats(self.depth_log[si])
            summary_lines.append(
                f"{'Right' if si==0 else 'Left'} hand — transitions: {trans}")
            for g in self.GESTURES:
                pct = dwell.get(g, 0) / max(duration, 1) * 100
                summary_lines.append(f"  {g:<10}: {dwell.get(g,0):5.1f}s  ({pct:.0f}%)")
            if ds:
                summary_lines.append(
                    f"  Depth  mean={ds['mean']:.2f}m  "
                    f"std={ds['std']:.2f}m  "
                    f"[{ds['min']:.2f}–{ds['max']:.2f}m]")
            summary_lines.append("")

        ax_sum.text(0.02, 0.97, "\n".join(summary_lines),
                    transform=ax_sum.transAxes,
                    va="top", ha="left",
                    fontsize=7.5, family="monospace", color="#e0e0e0",
                    linespacing=1.5)
        ax_sum.set_title("Session Summary", color="white", fontsize=9, pad=4)

        ax_fps = fig.add_subplot(gs[0, 2:4], **_ax_style)
        if self.fps_log:
            ax_fps.plot(self.fps_log, color="#f0c040", linewidth=0.6, alpha=0.8)
            ax_fps.axhline(mean_fps, color="#f0c040", linestyle="--",
                           linewidth=0.8, alpha=0.5, label=f"mean {mean_fps:.1f}")
            ax_fps.legend(fontsize=7, facecolor="#1a1a2e",
                          labelcolor="white", framealpha=0.5)
        ax_fps.set_title("FPS over time", color="white", fontsize=9, pad=4)
        ax_fps.set_xlabel("frame", color="#aaa", fontsize=7)
        ax_fps.set_ylabel("fps",   color="#aaa", fontsize=7)
        ax_fps.tick_params(colors="#aaa", labelsize=6)
        for spine in ax_fps.spines.values(): spine.set_edgecolor("#333")

        for si in range(2):
            col_start = si * 2
            ax_g = fig.add_subplot(gs[1, col_start:col_start+2], **_ax_style)
            self._plot_gesture_timeline(ax_g, self.gesture_log[si],
                                        slot_names[si] + " — Gesture timeline")

        ax_d = fig.add_subplot(gs[2, 0:2], **_ax_style)
        for si, sname in enumerate(slot_names):
            if self.depth_log[si]:
                ts   = [t for t, _ in self.depth_log[si]]
                vals = [v for _, v in self.depth_log[si]]
                ax_d.plot(ts, vals, linewidth=0.7,
                          color=["#c090ff", "#50c8ff"][si],
                          label=sname, alpha=0.85)
        ax_d.set_title("Palm depth over time (m)", color="white", fontsize=9, pad=4)
        ax_d.set_xlabel("time (s)", color="#aaa", fontsize=7)
        ax_d.set_ylabel("depth (m)", color="#aaa", fontsize=7)
        ax_d.legend(fontsize=7, facecolor="#1a1a2e",
                    labelcolor="white", framealpha=0.5)
        ax_d.tick_params(colors="#aaa", labelsize=6)
        for spine in ax_d.spines.values(): spine.set_edgecolor("#333")

        ax_p = fig.add_subplot(gs[2, 2:4], **_ax_style)
        for si, sname in enumerate(slot_names):
            if self.pinch_log[si]:
                ts   = [t for t, _ in self.pinch_log[si]]
                vals = [v for _, v in self.pinch_log[si]]
                ax_p.plot(ts, vals, linewidth=0.7,
                          color=["#c090ff", "#50c8ff"][si],
                          label=sname, alpha=0.85)
        ax_p.axhline(0.40, color="#00C8C8", linestyle="--",
                     linewidth=0.8, alpha=0.6, label="pinch thr 0.40")
        ax_p.set_title("Thumb->index distance (normalised)", color="white",
                       fontsize=9, pad=4)
        ax_p.set_xlabel("time (s)", color="#aaa", fontsize=7)
        ax_p.set_ylabel("norm dist", color="#aaa", fontsize=7)
        ax_p.legend(fontsize=7, facecolor="#1a1a2e",
                    labelcolor="white", framealpha=0.5)
        ax_p.tick_params(colors="#aaa", labelsize=6)
        for spine in ax_p.spines.values(): spine.set_edgecolor("#333")

        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out    = Path(out_dir) / f"session_report_{ts_str}.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Session report saved to: {out}")
        self._print_terminal_summary(duration, det_rate, mean_fps)
        return str(out)

    def _dwell_and_trans(self, slot):
        log = self.gesture_log[slot]
        if not log:
            return {g: 0.0 for g in self.GESTURES}, 0
        dwell = defaultdict(float)
        trans = 0
        prev_t, prev_l = log[0]
        for t, l in log[1:]:
            dwell[prev_l] += t - prev_t
            if l != prev_l: trans += 1
            prev_t, prev_l = t, l
        return dict(dwell), trans

    @staticmethod
    def _plot_gesture_timeline(ax, log, title):
        _col = _GESTURE_COL_MPL
        gestures = SessionStats.GESTURES
        if not log:
            ax.text(0.5, 0.5, "no data", color="#888",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            prev_t, prev_l = log[0]
            for t, l in log[1:]:
                ax.barh(0, t - prev_t, left=prev_t, height=0.5,
                        color=_col.get(prev_l, "#888"), alpha=0.85)
                prev_t, prev_l = t, l
            from matplotlib.patches import Patch
            patches = [Patch(color=_col[g], label=g) for g in gestures]
            ax.legend(handles=patches, fontsize=6, facecolor="#1a1a2e",
                      labelcolor="white", framealpha=0.5, loc="upper right")
        ax.set_yticks([])
        ax.set_title(title, color="white", fontsize=8, pad=3)
        ax.set_xlabel("time (s)", color="#aaa", fontsize=7)
        ax.tick_params(colors="#aaa", labelsize=6)
        for spine in ax.spines.values(): spine.set_edgecolor("#333")
        ax.set_facecolor("#0f0f1a")

    def _print_terminal_summary(self, duration, det_rate, mean_fps):
        print("")
        print("Session summary")
        print("---------------")
        print(f"Duration      : {duration:.1f}s ({duration/60:.1f} min)")
        print(f"Frames total  : {self.total_frames}")
        print(f"Hand detected : {self.det_frames} frames ({det_rate:.1f}%)")
        print(f"Average FPS   : {mean_fps:.1f}")
        for si, sname in enumerate(["Right", "Left"]):
            dwell, trans = self._dwell_and_trans(si)
            ds = self._depth_stats(self.depth_log[si])
            print(f"")
            print(f"{sname} hand (transitions: {trans})")
            for g in self.GESTURES:
                pct = dwell.get(g, 0) / max(duration, 1) * 100
                bar = "#" * int(pct / 5)
                print(f"  {g:<10} {dwell.get(g,0):5.1f}s  {pct:4.0f}%  {bar}")
            if ds:
                print(f"  depth mean={ds['mean']:.3f}m "
                      f"std={ds['std']:.3f}m "
                      f"range=[{ds['min']:.3f}, {ds['max']:.3f}] m")


class RobotLearningHandPipeline:
    """Main pipeline. Camera overlay (left) + live 3D pose (right).

    Keys: c = calibrate depth, r = reset calibration, ESC = quit.
    """

    def __init__(self, device="cpu", score_thr=0.12,
                 depth_model="", da2_encoder="vitl", infer_scale=0.6,
                 report_dir="."):
        self.cam   = ThreadedCamera(width=CAM_W, height=CAM_H)
        self.depth = DepthEstimator(depth_model or "", device=device,
                                    encoder=da2_encoder)
        self.pose  = MMPoseHandBackend(device=device, score_thr=score_thr,
                                       infer_scale=infer_scale)
        self.score_thr  = score_thr
        self.report_dir = report_dir

        self.filters  = [OneEuroFilter(freq=25, min_cutoff=1.2, beta=0.05),
                         OneEuroFilter(freq=25, min_cutoff=1.2, beta=0.05)]
        self.gestures = [GestureAbstractor(history=5),
                         GestureAbstractor(history=5)]

        self._fig = plt.figure(figsize=(PLOT_W/100, PLOT_H/100),
                               dpi=100, facecolor="white")
        self._ax  = self._fig.add_subplot(111, projection="3d")

        self.last_t       = time.time()
        self.fps          = 0.0
        self.calib_active = False
        self.calib_start  = None
        self.calib_done   = False

        self.stats = SessionStats()

    def _tick(self):
        # Exponential moving average FPS counter.
        now = time.time(); dt = now - self.last_t; self.last_t = now
        if dt > 1e-6:
            self.fps = (1/dt) if self.fps == 0 else (0.8*self.fps + 0.2/dt)
        return self.fps

    def _visible(self, sc):
        return (float(np.mean(sc)) >= _MEAN_THR and
                float(np.mean(sc > _JOINT_THR)) >= _FRAC_THR)

    def _draw_skeleton(self, frame, kp2d, sc):
        # Draw five finger chains in distinct colours plus a grey palm polyline.
        for fi, chain in enumerate(FINGER_CHAINS_IDX):
            col = _BGR[fi]
            pts = [(int(round(kp2d[j,0])), int(round(kp2d[j,1])))
                   for j in chain if sc[j] > _JOINT_THR]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(frame, a, b, col, 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, 3, col, -1, cv2.LINE_AA)
        for a, b in zip([20,7,11,15], [7,11,15,19]):
            if sc[a] > _JOINT_THR and sc[b] > _JOINT_THR:
                cv2.line(frame,
                         tuple(np.round(kp2d[a]).astype(int)),
                         tuple(np.round(kp2d[b]).astype(int)),
                         _PALM_BGR, 1, cv2.LINE_AA)

    def _draw_depth_display(self, frame, slot, kp2d, sc, depth_map):
        # Anchored depth readout above each palm with a colour-graded bar.
        # Suffix 'm*' = uncalibrated, 'm' = calibrated.
        palm = [20, 7, 11, 15, 19]
        v    = [j for j in palm if sc[j] > _JOINT_THR]
        if not v:
            return None

        cx = float(np.mean(kp2d[v, 0]))
        cy = float(np.mean(kp2d[v, 1]))
        d  = self.depth.sample(depth_map, (cx, cy), patch=10)

        t = np.clip((d - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M), 0, 1)
        if t < 0.5:
            r = int(t * 2 * 255)
            g = 200
        else:
            r = 200
            g = int((1 - (t - 0.5) * 2) * 200)
        depth_col = (0, g, r)

        H, W    = frame.shape[:2]
        side_s  = "R" if slot == 0 else "L"
        suf     = "m" if self.calib_done else "m*"
        txt     = f"{side_s}  {d:.2f}{suf}"

        bw, bh  = 180, 38
        bx = int(np.clip(cx - bw//2, 4, W - bw - 4))
        by = int(np.clip(cy - 90,     4, H - bh - 4))

        ov = frame.copy()
        cv2.rectangle(ov, (bx, by), (bx+bw, by+bh), (15, 15, 15), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, txt, (bx+6, by+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, depth_col, 2, cv2.LINE_AA)

        bar_x, bar_y = bx+6, by+24
        bar_w, bar_h = bw-12, 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h),
                      (40, 40, 40), -1)
        fill = int(np.clip(t, 0, 1) * bar_w)
        if fill > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x+fill, bar_y+bar_h), depth_col, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h),
                      (100, 100, 100), 1)

        cv2.putText(frame, f"{DEPTH_MIN_M:.1f}", (bar_x, bar_y+bar_h+9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, (120,120,120), 1)
        cv2.putText(frame, f"{DEPTH_MAX_M:.1f}m", (bar_x+bar_w-22, bar_y+bar_h+9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, (120,120,120), 1)

        return d

    def _draw_gesture_label(self, frame, slot, kp2d, sc, label, meta):
        if sc[WRIST] > _JOINT_THR:
            ax_, ay_ = kp2d[WRIST]
        else:
            vis = kp2d[sc > _JOINT_THR]
            if not len(vis): return
            ax_, ay_ = vis.mean(0)

        H, W    = frame.shape[:2]
        col     = _GESTURE_COL.get(label, (200,200,200))
        side_s  = "R" if slot == 0 else "L"
        conf    = int(meta.get("confidence", 0) * 100)
        n_ext   = meta.get("n_extended", 0)
        pinch_n = meta.get("pinch_norm")
        pinch_s = f"{pinch_n:.2f}" if pinch_n is not None else "--"

        line1 = f"{side_s}: {label} {conf}%"
        line2 = f"ext:{n_ext}/4  pinch:{pinch_s}"

        fx = int(np.clip(ax_ - 70, 2, W - 240))
        fy = int(np.clip(ay_ + 24, 2, H - 44))

        ov = frame.copy()
        cv2.rectangle(ov, (fx, fy), (fx+232, fy+38), (10,10,10), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, line1, (fx+4, fy+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
        cv2.putText(frame, line2, (fx+4, fy+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1, cv2.LINE_AA)

    def _draw_digit_distances(self, frame, slot, kp2d, sc, meta):
        dists     = meta.get("digit_distances", {})
        thumb_tip = kp2d[FINGER_TIPS[0]]
        if sc[FINGER_TIPS[0]] < _JOINT_THR:
            return
        names     = ["thumb_index","thumb_middle","thumb_ring","thumb_pinky"]
        tip_ids   = [FINGER_TIPS[1], FINGER_TIPS[2], FINGER_TIPS[3], FINGER_TIPS[4]]
        line_cols = [(50,100,255),(50,205,50),(0,165,255),(147,112,219)]
        for name, tid, lcol in zip(names, tip_ids, line_cols):
            if sc[tid] < _JOINT_THR: continue
            d = dists.get(name)
            if d is None: continue
            p1 = tuple(np.round(thumb_tip).astype(int))
            p2 = tuple(np.round(kp2d[tid]).astype(int))
            cv2.line(frame, p1, p2, lcol, 1, cv2.LINE_AA)
            mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            cv2.putText(frame, f"{d:.2f}", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, lcol, 1, cv2.LINE_AA)

    def _draw_palm_orientation(self, frame, slot, kp2d, sc, palm_normal):
        palm = [20,7,11,15,19]
        v    = [j for j in palm if sc[j] > _JOINT_THR]
        if not v or np.allclose(palm_normal, 0): return
        cx = int(np.mean(kp2d[v,0])); cy = int(np.mean(kp2d[v,1]))
        nx, ny = float(palm_normal[0]), float(-palm_normal[1])
        ex, ey = int(cx+nx*45), int(cy+ny*45)
        col = _GESTURE_COL.get(self.gestures[slot]._current, (200,200,200))
        cv2.arrowedLine(frame, (cx,cy), (ex,ey), col, 2,
                        cv2.LINE_AA, tipLength=0.35)
        cv2.putText(frame, "n", (ex+3,ey-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1, cv2.LINE_AA)

    def _draw_depth_inset(self, frame, depth):
        col   = self.depth.colorize(depth)
        inset = cv2.resize(col, (140, 105))
        h     = frame.shape[0]
        frame[h-115:h-10, 10:150] = inset
        cv2.rectangle(frame,(10,h-115),(150,h-10),(140,140,140),1)
        label = "DA2 CAL" if self.calib_done else "DA2 UNCAL"
        cv2.putText(frame, label, (14,h-101),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (80,220,80) if self.calib_done else (180,180,180), 1)

    def _draw_session_overlay(self, frame):
        # Running stats in the bottom-left corner above the depth inset.
        elapsed  = time.time() - self.stats.t_start
        det_rate = self.stats.det_frames / max(self.stats.total_frames, 1) * 100
        H, W     = frame.shape[:2]

        lines = [
            f"T:{elapsed:.0f}s  det:{det_rate:.0f}%",
        ]
        for si in range(2):
            dwell, _ = self.stats._dwell_and_trans(si)
            if sum(dwell.values()) < 0.5:
                continue
            dom = max(dwell, key=dwell.get) if dwell else "-"
            lines.append(f"{'R' if si==0 else 'L'}: {dom[:9]}")

        x0, y0 = 160, H - 10 - len(lines)*14
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (x0, y0 + i*14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160,160,160), 1)

    def _draw_legend(self, frame):
        H, W = frame.shape[:2]
        items = [("GRASP", _GESTURE_COL["GRASP"]),
                 ("OPEN PALM", _GESTURE_COL["OPEN PALM"]),
                 ("PINCH", _GESTURE_COL["PINCH"])]
        x0, y0 = W-130, 12
        for i, (name, col) in enumerate(items):
            y = y0 + i*18
            cv2.rectangle(frame,(x0,y),(x0+12,y+10),col,-1)
            cv2.putText(frame,name,(x0+16,y+9),
                        cv2.FONT_HERSHEY_SIMPLEX,0.36,(200,200,200),1)

    def _draw_calib_ui(self, frame, hands):
        # Show a 3-second countdown overlay; return palm keypoints when done.
        elapsed   = time.time() - self.calib_start
        remaining = max(0.0, CALIB_SECS - elapsed)
        H, W      = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(W,H),(0,0,0),-1)
        cv2.addWeighted(ov,0.38,frame,0.62,0,frame)
        cv2.putText(frame,"Calibration: hold open palm at 40 cm",
                    (W//2-300,H//2-18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,220,60),2,cv2.LINE_AA)
        cv2.putText(frame,f"Capturing in {remaining:.1f}s",
                    (W//2-130,H//2+18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.60,(255,255,255),2,cv2.LINE_AA)
        bw = int((elapsed/CALIB_SECS)*(W-80))
        cv2.rectangle(frame,(40,H//2+42),(W-40,H//2+58),(50,50,50),-1)
        cv2.rectangle(frame,(40,H//2+42),(40+bw,H//2+58),(60,200,60),-1)
        if elapsed >= CALIB_SECS and hands:
            return hands[0]["keypoints"][:,:2]
        return None

    def _draw_hud(self, frame, n):
        H, W = frame.shape[:2]
        cal  = "CAL" if self.calib_done else "UNCAL"
        cv2.putText(frame,
                    f"FPS:{self.fps:.1f}  Hands:{n}  DA2/{cal}",
                    (12,H-8),cv2.FONT_HERSHEY_SIMPLEX,0.44,(160,160,160),1)
        cv2.putText(frame,"c=calibrate  r=reset  ESC=quit",
                    (W-260,H-8),cv2.FONT_HERSHEY_SIMPLEX,0.36,(120,120,120),1)

    def _render_3d(self, hands_data, title):
        # Render the current hand(s) as a 3D scatter + lines onto a Matplotlib
        # canvas, then convert to a BGR numpy image for OpenCV display.
        ax = self._ax; ax.cla()
        ax.set_facecolor("white")
        for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
            pane.fill=False; pane.set_edgecolor("#cccccc")
        ax.grid(True,color="#e0e0e0",linewidth=0.3)
        ax.tick_params(labelsize=5)
        ax.set_title(title,fontsize=9,pad=2)

        all_pts=[]
        for kp,sc in hands_data:
            has_z = kp.shape[1]>=3 and np.any(kp[:,2]!=0)
            px=kp[:,0].copy(); py=kp[:,2].copy() if has_z else np.zeros(21,np.float32)
            pz=-kp[:,1].copy()
            all_pts.append(np.stack([px,py,pz],axis=1))

            for fi,chain in enumerate(FINGER_CHAINS_IDX):
                col=_RGB[fi]
                good=[j for j in chain if sc[j]>_JOINT_THR]
                if len(good)<2: continue
                ax.plot([px[j] for j in good],[py[j] for j in good],[pz[j] for j in good],
                        color=col,linewidth=1.8,solid_capstyle="round")
                ax.scatter([px[j] for j in good],[py[j] for j in good],[pz[j] for j in good],
                           color=col,s=12,zorder=5,depthshade=False)
            pb=[j for j in [20,7,11,15,19] if sc[j]>_JOINT_THR]
            if len(pb)>=2:
                ax.plot([px[j] for j in pb],[py[j] for j in pb],[pz[j] for j in pb],
                        color=(0.75,0.75,0.75),linewidth=1.0)

        if all_pts:
            pts=np.concatenate(all_pts,axis=0)
            for dim,setter in enumerate([ax.set_xlim,ax.set_ylim,ax.set_zlim]):
                lo,hi=pts[:,dim].min(),pts[:,dim].max()
                pad=max((hi-lo)*0.28,40.0)
                setter(lo-pad,hi+pad)

        ax.set_xlabel("X",fontsize=7,labelpad=1)
        ax.set_ylabel("Z depth",fontsize=7,labelpad=1)
        ax.set_zlabel("Y",fontsize=7,labelpad=1)
        self._fig.canvas.draw()
        buf=np.frombuffer(self._fig.canvas.tostring_rgb(),dtype=np.uint8)
        buf=buf.reshape(self._fig.canvas.get_width_height()[::-1]+(3,))
        return cv2.cvtColor(cv2.resize(buf,(PLOT_W,PLOT_H)),cv2.COLOR_RGB2BGR)

    def run(self):
        # Main loop: grab frame -> DA2 depth -> InterNet -> per-hand 1 Euro
        # smoothing + gesture classification -> overlays -> 3D preview.
        blank = self._render_3d([], "Prediction (0)")

        while True:
            fps_now = self._tick()

            frame = self.cam.read()
            if frame is None:
                continue
            if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
                frame = cv2.resize(frame, (CAM_W, CAM_H))

            vis = frame.copy()

            depth_map, raw_depth = self.depth.estimate(frame)
            self._draw_depth_inset(vis, depth_map)

            depth_info = {"depth_map": depth_map, "depth_est": self.depth} \
                         if self.calib_done else None
            hands = self.pose.infer(frame, depth_info=depth_info)

            if self.calib_active:
                ckp = self._draw_calib_ui(vis, hands)
                if ckp is not None:
                    if self.depth.calibrate(raw_depth, ckp):
                        self.calib_done = True
                    self.calib_active = False

            hands_3d   = []
            metas_out  = []
            depths_out = []

            if hands:
                for slot, hand in enumerate(hands):
                    kp = hand["keypoints"].copy()
                    sc = hand["scores"].copy()

                    kp[:, :2] = self.filters[slot](kp[:, :2],
                                                   freq=max(fps_now, 10.0))

                    label, meta = self.gestures[slot].classify(
                        kp, sc, score_thr=self.score_thr)
                    meta["_label"] = label

                    dv = None
                    if self._visible(sc):
                        self._draw_skeleton(vis, kp[:, :2], sc)
                        dv = self._draw_depth_display(vis, slot,
                                                      kp[:, :2], sc, depth_map)
                        self._draw_gesture_label(vis, slot, kp[:, :2], sc,
                                                 label, meta)
                        self._draw_digit_distances(vis, slot, kp[:, :2],
                                                   sc, meta)
                        self._draw_palm_orientation(vis, slot, kp[:, :2], sc,
                                                    meta.get("palm_normal",
                                                             np.zeros(3)))

                    metas_out.append(meta)
                    depths_out.append(dv)
                    hands_3d.append((kp, sc))

                for slot in range(2):
                    if slot >= len(hands):
                        self.filters[slot].reset()
                        self.gestures[slot].history.clear()
                        self.gestures[slot]._current     = "GRASP"
                        self.gestures[slot]._open_streak = 0
            else:
                for f in self.filters: f.reset()
                for g in self.gestures:
                    g.history.clear(); g._current="GRASP"; g._open_streak=0
                if not self.calib_active:
                    cv2.putText(vis,"No hand detected",(12,44),
                                cv2.FONT_HERSHEY_SIMPLEX,0.75,(140,140,140),2)

            self.stats.update(fps_now, hands, metas_out, depths_out)
            self._draw_session_overlay(vis)

            self._draw_legend(vis)
            self._draw_hud(vis, len(hands))

            n    = len(hands_3d)
            plot = self._render_3d(hands_3d,f"Prediction ({n})") if n else blank

            canvas = np.ones((CAM_H,CAM_W+PLOT_W,3),dtype=np.uint8)*235
            canvas[:,:CAM_W]                                  = vis
            canvas[:plot.shape[0],CAM_W:CAM_W+plot.shape[1]] = plot

            cv2.imshow("Robot Learning Hand Pipeline", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('c') and not self.calib_active:
                self.calib_active = True
                self.calib_start  = time.time()
            elif key == ord('r'):
                self.depth.calib_scale = None
                self.calib_done = self.calib_active = False
                self.pose._depths = [None, None]
                print("Depth calibration reset.")

        self.stats.save_report(out_dir=self.report_dir)

        plt.close(self._fig)
        self.cam.release()
        cv2.destroyAllWindows()
