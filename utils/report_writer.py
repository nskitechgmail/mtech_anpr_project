"""
utils/report_writer.py — Writes CSV and JSON violation reports.
utils/anonymiser.py    — Face blurring for privacy compliance.
utils/frame_utils.py   — Common frame utility functions.
"""

# ════════════════════════════════════════════════════════════════════════════
# report_writer.py
# ════════════════════════════════════════════════════════════════════════════

import csv, json, os, logging, threading
from datetime import datetime
from pathlib import Path

log = logging.getLogger("ReportWriter")

class ReportWriter:
    """Thread-safe CSV + JSON report writer."""

    FIELDS = [
        "timestamp","camera_id","frame",
        "plate","plate_raw","plate_conf","plate_valid","plate_enhanced",
        "vehicle_class","violation",
        "helmet","helmet_conf","seatbelt","seatbelt_conf",
        "image_saved",
    ]

    def __init__(self, settings):
        self.cfg   = settings
        self._lock = threading.Lock()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = Path(settings.output_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path  = reports_dir / f"violations_{ts}.csv"
        self._json_path = reports_dir / f"violations_{ts}.json"
        self._json_buf  = []
        self._frame_no  = 0

        # Open CSV
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.FIELDS,
                                          extrasaction="ignore")
        self._csv_writer.writeheader()
        log.info(f"Report file: {self._csv_path}")

    def write(self, record: dict):
        with self._lock:
            self._frame_no += 1
            record["frame"] = self._frame_no
            # CSV
            self._csv_writer.writerow(record)
            self._csv_file.flush()
            # JSON buffer
            self._json_buf.append(record)
            if len(self._json_buf) % 10 == 0:
                self._flush_json()

    def close(self):
        with self._lock:
            self._flush_json()
            self._csv_file.close()
            log.info(f"Report saved: {self._csv_path}")

    def _flush_json(self):
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(self._json_buf, f, indent=2, default=str)


# ════════════════════════════════════════════════════════════════════════════
# anonymiser.py
# ════════════════════════════════════════════════════════════════════════════

import cv2
import numpy as np

class FaceAnonymiser:
    """
    Detects and blurs faces using MediaPipe (primary) or
    OpenCV Haar cascade (fallback) for GDPR / DPDP compliance.
    """

    def __init__(self):
        self._mp_detector = None
        self._haar        = None
        self._init_detector()

    def _init_detector(self):
        try:
            import mediapipe as mp
            self._mp_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            log.info("FaceAnonymiser: MediaPipe ready")
        except ImportError:
            # Haar cascade fallback
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade):
                self._haar = cv2.CascadeClassifier(cascade)
                log.info("FaceAnonymiser: Haar cascade ready")
            else:
                log.info("FaceAnonymiser: no face detector available")

    def blur_faces(self, frame: np.ndarray, blur_ksize: int = 51) -> np.ndarray:
        if frame is None:
            return frame
        faces = self._detect_faces(frame)
        for (x,y,w,h) in faces:
            # Expand ROI slightly
            pad = int(max(w,h) * 0.15)
            x1 = max(0, x-pad); y1 = max(0, y-pad)
            x2 = min(frame.shape[1], x+w+pad)
            y2 = min(frame.shape[0], y+h+pad)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                k = blur_ksize | 1   # must be odd
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k,k), 0)
        return frame

    def _detect_faces(self, frame: np.ndarray) -> list:
        if self._mp_detector:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._mp_detector.process(rgb)
            faces = []
            if results.detections:
                h,w = frame.shape[:2]
                for d in results.detections:
                    bb = d.location_data.relative_bounding_box
                    x = int(bb.xmin * w); y = int(bb.ymin * h)
                    fw = int(bb.width * w); fh = int(bb.height * h)
                    faces.append((x,y,fw,fh))
            return faces
        elif self._haar is not None:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return list(self._haar.detectMultiScale(grey,1.1,4,minSize=(30,30)))
        return []


# ════════════════════════════════════════════════════════════════════════════
# frame_utils.py
# ════════════════════════════════════════════════════════════════════════════

def resize_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize preserving aspect ratio, pad with black."""
    h, w = frame.shape[:2]
    scale = min(target_w/w, target_h/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(frame, (nw,nh))
    out = np.zeros((target_h,target_w,3),dtype=np.uint8)
    y_off = (target_h-nh)//2
    x_off = (target_w-nw)//2
    out[y_off:y_off+nh, x_off:x_off+nw] = resized
    return out

def pil_to_cv(pil_image) -> np.ndarray:
    import numpy as np
    from PIL import Image
    arr = np.array(pil_image.convert("RGB"))
    return arr[:,:,::-1]   # RGB → BGR

def cv_to_pil(cv_image: np.ndarray):
    from PIL import Image
    return Image.fromarray(cv_image[:,:,::-1])
