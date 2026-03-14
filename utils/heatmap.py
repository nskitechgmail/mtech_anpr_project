"""
utils/heatmap.py — Traffic density heatmap overlay (Sprint 3).

Accumulates vehicle bounding-box centre positions across frames and
renders a colour-coded OpenCV heat-map overlay on top of the video frame.
Supports per-hour statistics and snapshot saving.
"""

from __future__ import annotations
import cv2
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger("Heatmap")


class TrafficHeatmap:
    """
    Incremental traffic density heatmap.

    Parameters
    ----------
    width, height : int
        Frame dimensions (used to size the accumulator grid).
    decay         : float
        Per-frame decay factor (0–1). Lower = faster fade.
    output_dir    : str
        Directory where snapshot images are saved.
    """

    def __init__(
        self,
        width:      int   = 1280,
        height:     int   = 720,
        decay:      float = 0.995,
        output_dir: str   = "outputs/heatmaps",
    ):
        self.width      = width
        self.height     = height
        self.decay      = decay
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Float accumulator grid (same resolution as frame)
        self._density   = np.zeros((height, width), dtype=np.float32)
        self._frame_count = 0

        # Hourly vehicle counts {hour_int: count}
        self._hourly_counts: dict[int, int] = defaultdict(int)

        # Gaussian spread kernel (radius ~30 px)
        self._kernel = self._make_gaussian_kernel(radius=30)

    # ── Public API ──────────────────────────────────────────────────

    def update(self, detections: list):
        """
        Add vehicle positions from the current frame to the accumulator.

        Parameters
        ----------
        detections : list[VehicleDetection]
            Output from PlateRecogniser.process_frame().
        """
        self._frame_count += 1
        hour = datetime.now().hour

        # Decay existing density
        self._density *= self.decay

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            self._add_gaussian(cx, cy)
            self._hourly_counts[hour] += 1

    def render(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """
        Overlay the heatmap onto *frame* and return the blended result.
        The original frame is NOT modified.
        """
        h, w = frame.shape[:2]

        # Resize density grid if frame size changed
        density = self._density
        if (h, w) != (self.height, self.width):
            density = cv2.resize(density, (w, h))

        # Normalise to 0–255
        d_norm = density.copy()
        max_val = d_norm.max()
        if max_val > 0:
            d_norm = (d_norm / max_val * 255).astype(np.uint8)
        else:
            d_norm = d_norm.astype(np.uint8)

        # Apply JET colour map
        heatmap_colour = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)

        # Blend onto frame
        result = cv2.addWeighted(frame, 1.0 - alpha, heatmap_colour, alpha, 0)
        return result

    def save_snapshot(self, tag: str = "") -> Path:
        """Save current density grid as a standalone PNG."""
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"heatmap_{ts}{'_' + tag if tag else ''}.png"
        path  = self.output_dir / fname

        d_norm = self._density.copy()
        max_val = d_norm.max()
        if max_val > 0:
            d_norm = (d_norm / max_val * 255).astype(np.uint8)
        else:
            d_norm = d_norm.astype(np.uint8)

        heatmap_colour = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(path), heatmap_colour)
        log.info(f"Heatmap snapshot saved → {path}")
        return path

    def get_stats(self) -> dict:
        """Return serialisable stats for REST API."""
        return {
            "frame_count"  : self._frame_count,
            "peak_density" : float(self._density.max()),
            "hourly_counts": dict(self._hourly_counts),
        }

    def reset(self):
        self._density[:] = 0
        self._hourly_counts.clear()
        self._frame_count = 0

    # ── Internals ───────────────────────────────────────────────────

    def _add_gaussian(self, cx: int, cy: int):
        """Splat a 2-D Gaussian centred at (cx, cy) into the density grid."""
        kr = self._kernel.shape[0] // 2  # half-radius
        x1 = cx - kr
        y1 = cy - kr
        x2 = cx + kr + 1
        y2 = cy + kr + 1

        # Clip to grid bounds and compute corresponding kernel slice
        gx1 = max(0, -x1)
        gy1 = max(0, -y1)
        gx2 = self._kernel.shape[1] - max(0, x2 - self.width)
        gy2 = self._kernel.shape[0] - max(0, y2 - self.height)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.width,  x2)
        y2 = min(self.height, y2)

        if x2 <= x1 or y2 <= y1 or gx2 <= gx1 or gy2 <= gy1:
            return

        self._density[y1:y2, x1:x2] += self._kernel[gy1:gy2, gx1:gx2]

    @staticmethod
    def _make_gaussian_kernel(radius: int = 30) -> np.ndarray:
        size = 2 * radius + 1
        k    = np.zeros((size, size), dtype=np.float32)
        cx   = cy = radius
        sigma = radius / 2.5
        for y in range(size):
            for x in range(size):
                dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                k[y, x] = np.exp(-dist_sq / (2 * sigma ** 2))
        return k / k.sum()   # normalise to unit mass
