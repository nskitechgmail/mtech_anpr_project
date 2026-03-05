"""
core/pipeline.py — Main ANPR pipeline orchestrator.

Manages:
  • Video capture loop (webcam / file / RTSP)
  • Frame-by-frame processing
  • Violation tracking with temporal smoothing
  • Output: annotated video, CSV/JSON reports, saved violation images
  • Headless server mode (no GUI)
"""

from __future__ import annotations
import cv2, time, logging, json, csv, os, threading
from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import numpy as np

from models.model_manager   import ModelManager
from core.plate_recogniser  import PlateRecogniser, VehicleDetection
from utils.annotator        import FrameAnnotator
from utils.report_writer    import ReportWriter
from utils.anonymiser       import FaceAnonymiser

log = logging.getLogger("Pipeline")


@dataclass
class FrameStats:
    frame_num:    int   = 0
    fps:          float = 0.0
    vehicles:     int   = 0
    violations:   int   = 0
    plates_read:  int   = 0


class ViolationTracker:
    """
    Temporal smoothing: a violation is confirmed only after appearing
    in N consecutive frames (cfg.violation_frames) to reduce false positives.
    """

    def __init__(self, n_frames: int = 3):
        self.n = n_frames
        self._counts: dict[str, deque] = defaultdict(lambda: deque(maxlen=n_frames))
        self._reported: set[str] = set()

    def update(self, plate_text: str, violation: str) -> bool:
        """
        Returns True if this violation should be reported now
        (first frame it is confirmed after N consecutive detections).
        """
        key = f"{plate_text}:{violation}"
        self._counts[key].append(1)
        if len(self._counts[key]) == self.n \
                and sum(self._counts[key]) == self.n \
                and key not in self._reported:
            self._reported.add(key)
            return True
        return False

    def reset(self, plate_text: str):
        """Reset tracker for a plate (e.g. vehicle left scene)."""
        keys = [k for k in self._counts if k.startswith(plate_text)]
        for k in keys:
            self._counts[k].clear()


class ANPRPipeline:
    """
    Top-level pipeline object.

    Usage (headless):
        pipeline = ANPRPipeline(settings)
        pipeline.run_headless()

    Usage (GUI-driven):
        pipeline = ANPRPipeline(settings)
        pipeline.start()
        frame, stats, detections = pipeline.get_latest()
        pipeline.stop()
    """

    def __init__(self, settings):
        self.cfg       = settings
        self.models    = ModelManager(settings)
        self.recogniser= None
        self.annotator = FrameAnnotator(settings)
        self.reporter  = ReportWriter(settings)
        self.anonymiser= FaceAnonymiser()
        self.tracker   = ViolationTracker(settings.violation_frames)

        # Shared state (GUI ↔ pipeline thread)
        self._running       = False
        self._latest_frame  = None
        self._latest_stats  = FrameStats()
        self._latest_dets   = []
        self._lock          = threading.Lock()
        self._thread        = None

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self):
        """Start pipeline in background thread (for GUI mode)."""
        self.models.load_all()
        self.recogniser = PlateRecogniser(self.models, self.cfg)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Pipeline thread started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self.reporter.close()
        log.info("Pipeline stopped.")

    def get_latest(self) -> tuple:
        """Return (annotated_frame, stats, detections) — thread-safe."""
        with self._lock:
            return (
                self._latest_frame,
                self._latest_stats,
                list(self._latest_dets),
            )

    def process_single_image(self, image_path: str) -> tuple[np.ndarray, list]:
        """Process a single image (no video loop). Returns (annotated, dets)."""
        self.models.load_all()
        self.recogniser = PlateRecogniser(self.models, self.cfg)
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        detections = self.recogniser.process_frame(frame)
        annotated  = self.annotator.draw(frame, detections, FrameStats(vehicles=len(detections)))
        return annotated, detections

    def run_headless(self):
        """Blocking headless processing loop with console output."""
        self.models.load_all()
        self.recogniser = PlateRecogniser(self.models, self.cfg)
        cap = self._open_capture()
        if not cap or not cap.isOpened():
            log.error("Could not open video source.")
            return

        log.info("Headless processing started. Press Ctrl+C to stop.")
        self._running = True
        stats = FrameStats()
        try:
            self._capture_loop(cap, stats, headless=True)
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        finally:
            cap.release()
            self.reporter.close()
            self._print_summary(stats)

    # ── Internal loop ───────────────────────────────────────────────────────

    def _loop(self):
        """Background thread for GUI mode."""
        cap = self._open_capture()
        if not cap or not cap.isOpened():
            log.error("Could not open video source.")
            return
        stats = FrameStats()
        self._capture_loop(cap, stats, headless=False)
        cap.release()

    def _capture_loop(self, cap, stats: FrameStats, headless: bool):
        fps_timer  = time.time()
        fps_frames = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                if isinstance(self.cfg.source, str) and self.cfg.source.endswith(
                    (".mp4", ".avi", ".mov", ".mkv")
                ):
                    # Loop video file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            stats.frame_num += 1

            # Skip frames to maintain target FPS (for high-res CCTV streams)
            if stats.frame_num % max(1, int(cap.get(cv2.CAP_PROP_FPS) /
                                           max(1, self.cfg.fps_target))) != 0:
                continue

            # ── Process frame ─────────────────────────────────────────────
            detections = self.recogniser.process_frame(frame)

            # ── Face anonymisation ────────────────────────────────────────
            if self.cfg.anonymise_faces:
                frame = self.anonymiser.blur_faces(frame)

            # ── Violation tracking + saving ───────────────────────────────
            for det in detections:
                stats.vehicles   += 1
                if det.plate:
                    stats.plates_read += 1
                if det.has_violation():
                    stats.violations  += 1
                    plate_txt = det.plate.text if det.plate else "UNKNOWN"
                    confirmed = self.tracker.update(plate_txt, det.violation)
                    if confirmed:
                        self._handle_violation(frame, det)

            # ── Annotate ──────────────────────────────────────────────────
            fps_frames += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                stats.fps = fps_frames / (now - fps_timer)
                fps_frames = 0
                fps_timer = now

            annotated = self.annotator.draw(frame, detections, stats)

            if headless:
                # Console output
                self._print_frame_info(stats, detections)
                # Optional: write annotated video
            else:
                # Share with GUI
                with self._lock:
                    self._latest_frame = annotated
                    self._latest_stats = FrameStats(
                        frame_num  = stats.frame_num,
                        fps        = stats.fps,
                        vehicles   = len(detections),
                        violations = sum(1 for d in detections if d.has_violation()),
                        plates_read= sum(1 for d in detections if d.plate),
                    )
                    self._latest_dets = detections

    def _handle_violation(self, frame: np.ndarray, det: VehicleDetection):
        """Save violation image, log to report, emit alert."""
        plate_txt = det.plate.normalised() if det.plate else "UNKNOWN"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{self.cfg.camera_id}_{det.violation.replace(' ','_')}_{plate_txt.replace(' ','')}_{ts}.jpg"
        save_path = Path(self.cfg.output_dir) / "violations" / fname

        if self.cfg.save_violations:
            cv2.imwrite(str(save_path), frame)

        record = {
            "timestamp"    : datetime.now().isoformat(),
            "camera_id"    : self.cfg.camera_id,
            "plate"        : plate_txt,
            "plate_raw"    : det.plate.raw_text if det.plate else "",
            "plate_conf"   : round(det.plate.confidence * 100, 1) if det.plate else 0,
            "plate_valid"  : det.plate.valid_format if det.plate else False,
            "plate_enhanced": det.plate.enhanced if det.plate else False,
            "vehicle_class": det.vehicle_class,
            "violation"    : det.violation,
            "helmet"       : det.helmet,
            "helmet_conf"  : round(det.helmet_conf * 100, 1),
            "seatbelt"     : det.seatbelt,
            "seatbelt_conf": round(det.seatbelt_conf * 100, 1),
            "image_saved"  : str(save_path) if self.cfg.save_violations else "",
        }
        self.reporter.write(record)
        log.warning(
            f"⚠  VIOLATION | {det.violation:<18} | "
            f"Plate: {plate_txt:<14} | "
            f"Vehicle: {det.vehicle_class}"
        )

    def _open_capture(self) -> cv2.VideoCapture:
        src = self.cfg.source
        log.info(f"Opening source: {src}")
        cap = cv2.VideoCapture(src)

        # CCTV-friendly capture settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # minimal buffer lag
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if isinstance(src, str) and src.startswith("rtsp"):
            # Force FFMPEG backend for RTSP
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            log.error(f"Failed to open: {src}")
        return cap

    def _print_frame_info(self, stats: FrameStats, dets: list):
        viol = [d for d in dets if d.has_violation()]
        plates = [d.plate.text for d in dets if d.plate]
        print(
            f"\r[Frame {stats.frame_num:06d}] "
            f"FPS:{stats.fps:5.1f} | "
            f"Vehicles:{len(dets)} | "
            f"Plates:{plates} | "
            f"Violations:{len(viol)}",
            end="", flush=True,
        )

    def _print_summary(self, stats: FrameStats):
        print("\n" + "=" * 60)
        print("  PROCESSING SUMMARY")
        print("=" * 60)
        print(f"  Total frames processed : {stats.frame_num}")
        print(f"  Total vehicles detected: {stats.vehicles}")
        print(f"  Total plates read      : {stats.plates_read}")
        print(f"  Total violations       : {stats.violations}")
        print(f"  Report saved to        : {self.cfg.output_dir}/reports/")
        print("=" * 60)
