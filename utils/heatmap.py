"""
utils/heatmap.py — Traffic density heatmap per junction per hour.

Accumulates vehicle detection centroids in a spatial grid and renders
an OpenCV-colourmap overlay on top of a road background image or blank canvas.
Supports hourly snapshots and multi-camera junction comparison.
"""
from __future__ import annotations
import cv2, json, logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

log = logging.getLogger("Heatmap")

# Colourmap: COLORMAP_JET gives blue(cold) → red(hot)
_CMAP = cv2.COLORMAP_JET
_DECAY = 0.98          # per-frame exponential decay so old detections fade
_BLUR_SIGMA = 25       # Gaussian blur radius for smooth heat blobs


class TrafficHeatmap:
    """
    Maintains a floating-point density map the same size as the video frame.

    Usage::
        hm = TrafficHeatmap(width=1280, height=720)
        hm.update(detections)          # call once per frame
        overlay = hm.render(frame)     # BGR overlay image
        hm.save_snapshot("out/")       # write hour-stamped PNG + JSON
    """

    def __init__(self, width: int = 1280, height: int = 720,
                 decay: float = _DECAY, output_dir: str = "outputs/heatmaps"):
        self.w      = width
        self.h      = height
        self.decay  = decay
        self.outdir = Path(output_dir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self._density  = np.zeros((height, width), dtype=np.float32)
        self._counts   = defaultdict(int)     # {hour_str: total_vehicles}
        self._frame_no = 0
        self._last_hour = None

    def update(self, detections: list) -> None:
        """
        Add vehicle centroids from the current frame to the density map.
        *detections* is a list of VehicleDetection objects.
        """
        self._frame_no += 1
        # Decay existing heat
        self._density *= self.decay

        hour_key = datetime.now().strftime("%Y-%m-%d %H:00")
        if self._last_hour and self._last_hour != hour_key:
            # New hour started — save snapshot of the previous hour
            log.info(f"Heatmap: saving hourly snapshot for {self._last_hour}")
            self.save_snapshot(tag=self._last_hour.replace(":", "-").replace(" ", "_"))
        self._last_hour = hour_key

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cx = max(0, min(self.w - 1, cx))
            cy = max(0, min(self.h - 1, cy))
            # Add Gaussian blob at centroid
            self._add_blob(cx, cy, radius=40)
            self._counts[hour_key] += 1

    def _add_blob(self, cx: int, cy: int, radius: int = 40) -> None:
        """Add a soft circular hotspot at (cx, cy)."""
        # Create a small patch
        size   = radius * 2 + 1
        patch  = np.zeros((size, size), dtype=np.float32)
        cv2.circle(patch, (radius, radius), radius, 1.0, -1)
        patch  = cv2.GaussianBlur(patch, (0, 0), _BLUR_SIGMA)

        x1 = cx - radius;  y1 = cy - radius
        x2 = cx + radius + 1; y2 = cy + radius + 1

        # Clip to frame bounds
        px1 = max(0, -x1);  py1 = max(0, -y1)
        px2 = size - max(0, x2 - self.w)
        py2 = size - max(0, y2 - self.h)
        x1  = max(0, x1);  y1 = max(0, y1)
        x2  = min(self.w, x2); y2 = min(self.h, y2)

        if x2 > x1 and y2 > y1:
            self._density[y1:y2, x1:x2] += patch[py1:py2, px1:px2]

    def render(self, frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """
        Render heatmap overlay on top of *frame*.
        Returns a new BGR image (same size as frame).
        """
        # Normalise to 0–255
        d = self._density.copy()
        max_val = d.max()
        if max_val > 0:
            d = (d / max_val * 255).astype(np.uint8)
        else:
            d = d.astype(np.uint8)

        # Apply colourmap
        coloured = cv2.applyColorMap(d, _CMAP)

        # Blend over the frame
        resized = cv2.resize(coloured, (frame.shape[1], frame.shape[0]))
        overlay = cv2.addWeighted(frame, 1 - alpha, resized, alpha, 0)

        # Draw legend
        self._draw_legend(overlay)
        return overlay

    def _draw_legend(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        # Gradient bar (20 px wide, 100 px tall)
        bar_h, bar_w = 100, 20
        bx, by = w - 60, h - 140
        gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
        gradient  = np.tile(gradient, (1, bar_w))
        coloured  = cv2.applyColorMap(gradient, _CMAP)
        img[by:by+bar_h, bx:bx+bar_w] = coloured
        cv2.putText(img, "High", (bx + bar_w + 4, by + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv2.putText(img, "Low",  (bx + bar_w + 4, by + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv2.putText(img, "Density", (bx - 4, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    def save_snapshot(self, tag: str = "") -> Path:
        """Save current density map as a PNG and summary JSON."""
        ts   = tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        png  = self.outdir / f"heatmap_{ts}.png"
        json_path = self.outdir / f"heatmap_{ts}.json"

        # Render on blank dark background
        bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img = self.render(bg, alpha=0.85)
        cv2.imwrite(str(png), img)

        summary = {
            "snapshot_time" : ts,
            "frame_count"   : self._frame_no,
            "hourly_counts" : dict(self._counts),
            "peak_hour"     : max(self._counts, key=self._counts.get)
                              if self._counts else None,
        }
        json_path.write_text(json.dumps(summary, indent=2))
        log.info(f"Heatmap snapshot saved: {png}")
        return png

    def get_stats(self) -> dict:
        """Return a dict of heatmap stats for the API."""
        return {
            "frame_count"   : self._frame_no,
            "hourly_counts" : dict(self._counts),
            "peak_hour"     : max(self._counts, key=self._counts.get)
                              if self._counts else None,
            "current_max_density": float(self._density.max()),
        }
