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
    esrgan_tile: int = 0        # tile size (0 = auto; set 256 for low VRAM)
    detector_model: str = "yolov9c"   # yolov9c | yolov9e | yolov8n fallback
    ocr_languages: list = field(default_factory=lambda: ["en"])

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

    # ── Alerts (Sprint 3) ─────────────────────────────────────────
    enable_alerts: bool = False
    alert_repeat_threshold: int = 3   # violations before alert fires
    alert_smtp_host: str = ""
    alert_smtp_port: int = 587
    alert_smtp_user: str = ""
    alert_smtp_pass: str = ""
    alert_email_from: str = ""
    alert_email_to: str = ""
    alert_twilio_sid: str = ""
    alert_twilio_token: str = ""
    alert_sms_to: str = ""
    alert_sms_from: str = ""

    # ── UI / Display ──────────────────────────────────────────────
    show_plate_crop: bool = True   # show inset plate thumbnail in GUI/annotator

    # ── Output ────────────────────────────────────────────────────
    output_dir: str = "outputs"
    save_violations: bool = True

    def __post_init__(self):
        # Convert numeric string source to int (webcam index)
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)

        # Load alert credentials from environment if not set
        def _env(attr: str, env_key: str):
            if not getattr(self, attr):
                val = os.environ.get(env_key, "")
                if val:
                    setattr(self, attr, val)

        _env("alert_smtp_host",   "ANPR_ALERT_SMTP_HOST")
        _env("alert_smtp_user",   "ANPR_ALERT_SMTP_USER")
        _env("alert_smtp_pass",   "ANPR_ALERT_SMTP_PASS")
        _env("alert_email_from",  "ANPR_ALERT_EMAIL_FROM")
        _env("alert_email_to",    "ANPR_ALERT_EMAIL_TO")
        _env("alert_twilio_sid",  "ANPR_TWILIO_SID")
        _env("alert_twilio_token","ANPR_TWILIO_TOKEN")
        _env("alert_sms_to",      "ANPR_ALERT_SMS_TO")
        _env("alert_sms_from",    "ANPR_ALERT_SMS_FROM")
        smtp_port_env = os.environ.get("ANPR_ALERT_SMTP_PORT", "")
        if smtp_port_env.isdigit():
            self.alert_smtp_port = int(smtp_port_env)

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
