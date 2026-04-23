# Varun Raghavendra
# PRCV Spring 2026
# Adaptive 1 Euro low-pass filter for smoothing noisy keypoint streams (Casiez et al., CHI 2012)

import math
import numpy as np


class LowPassFilter:

    def __init__(self):
        self.prev: np.ndarray | None = None

    def apply(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        # Applies a single-pole IIR filter y[n] = a*x[n] + (1-a)*y[n-1], broadcastable over any array shape.
        x     = np.asarray(x,     dtype=np.float32)
        alpha = np.asarray(alpha, dtype=np.float32)
        if self.prev is None:
            self.prev = x.copy()
            return x.copy()
        self.prev = alpha * x + (1.0 - alpha) * self.prev
        return self.prev.copy()

    def reset(self):
        # Clears the filter state so the next call is treated as the first sample.
        self.prev = None


class OneEuroFilter:

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.2,
                 beta: float = 0.05, d_cutoff: float = 1.0):
        self.freq       = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta       = float(beta)
        self.d_cutoff   = float(d_cutoff)
        self.xf         = LowPassFilter()
        self.dxf        = LowPassFilter()
        self.prev: np.ndarray | None = None

    def _alpha(self, cutoff) -> np.ndarray:
        # Converts a cutoff frequency in Hz to the equivalent IIR alpha coefficient at the current sample rate.
        te  = 1.0 / max(self.freq, 1e-6)
        tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff, dtype=np.float32))
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: np.ndarray, freq: float | None = None) -> np.ndarray:
        # Filters x with an adaptive cutoff f_c = min_cutoff + beta * |velocity|.
        # Pass freq to update the assumed sampling rate on the fly.
        if freq is not None and freq > 1e-6:
            self.freq = float(freq)

        x = np.asarray(x, dtype=np.float32)

        if self.prev is None:
            self.prev = x.copy()
            self.xf.apply(x, self._alpha(self.min_cutoff))
            return x.copy()

        dx        = (x - self.prev) * self.freq
        self.prev = x.copy()
        edx       = self.dxf.apply(dx, self._alpha(self.d_cutoff))

        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        return self.xf.apply(x, self._alpha(cutoff))

    def reset(self):
        # Resets both internal low-pass filters and clears the previous sample.
        self.xf.reset()
        self.dxf.reset()
        self.prev = None
