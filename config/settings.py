"""
config/settings.py — Centralised runtime configuration for the ANPR system.
All pipeline modules read from this single Settings object.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    # ── Input source ──────────────────────────────────────────────
    source: str = "0"           # "0" = webcam, path, or rtsp://...
    camera_id: str = "CCTV-001"

    # ── Model / inference ─────────────────────────────────────────
    use_genai: bool = True       # enable Real-ESRGAN enhancement
    conf_thresh: float = 0.40   # YOLO detection confidence
    iou_thresh: float = 0.45    # YOLO NMS IoU
    device: str = "auto"        # auto | cpu | cuda | mps
    esrgan_scale: int = 4       # Real-ESRGAN upscale factor

    # ── Safety compliance thresholds ──────────────────────────────
    helmet_conf: float = 0.55   # min confidence to flag no-helmet
    seatbelt_conf: float = 0.55 # min confidence to flag no-seatbelt
    violation_frames: int = 3   # consecutive frames before confirming violation

    # ── Plate settings ────────────────────────────────────────────
    plate_min_area: int = 500   # minimum plate pixel area to process

    # ── Processing ────────────────────────────────────────────────
    fps_target: int = 30        # target processing FPS

    # ── Privacy ───────────────────────────────────────────────────
    anonymise_faces: bool = True

    # ── Heatmap (Sprint 3) ────────────────────────────────────────
    enable_heatmap: bool = False

    # ── Output ────────────────────────────────────────────────────
    output_dir: str = "outputs"
    save_violations: bool = True

    def __post_init__(self):
        # Convert numeric string source to int (webcam index)
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)

        # Resolve device
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"

        # Create output directories
        for sub in ("reports", "violations", "heatmaps"):
            Path(self.output_dir, sub).mkdir(parents=True, exist_ok=True)
