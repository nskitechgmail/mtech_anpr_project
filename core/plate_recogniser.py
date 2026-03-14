"""
core/plate_recogniser.py — End-to-end licence plate recognition pipeline.

Stage 1 : YOLO vehicle detection
Stage 2 : Plate localisation (contour + aspect-ratio filter)
Stage 3 : GenAI enhancement via Real-ESRGAN (or PIL fallback)
Stage 4 : CLAHE + Otsu pre-processing → EasyOCR
Stage 5 : Indian plate format validation + post-processing correction
"""

from __future__ import annotations
import cv2, re, time, logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("PlateRecogniser")

# ── Indian licence plate regex ─────────────────────────────────────────────
# Covers: MH12AB1234 / MH 12 AB 1234 / MH-12-AB-1234 / MH12A1234 etc.
_PLATE_RE = re.compile(
    r"^[A-Z]{2}[\s\-]?\d{1,2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{1,4}$",
    re.IGNORECASE,
)

# COCO class IDs that correspond to vehicles
_VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Common OCR correction map (visually similar characters on plates)
_OCR_FIXES = {
    "O": "0", "I": "1", "Z": "2", "S": "5", "B": "8",
}


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class PlateResult:
    text:          str
    confidence:    float
    bbox:          tuple          # (x1,y1,x2,y2) in original frame coords
    plate_crop:    Optional[np.ndarray] = None   # enhanced plate image (BGR)
    enhanced:      bool = False
    valid_format:  bool = False
    raw_text:      str  = ""      # OCR output before post-processing

    def __post_init__(self):
        self.valid_format = bool(_PLATE_RE.match(self.text))

    def normalised(self) -> str:
        """Return plate text in canonical upper-case spaced form."""
        t = re.sub(r"[\s\-]+", " ", self.text.upper().strip())
        return t


@dataclass
class VehicleDetection:
    vehicle_class: str
    bbox:          tuple          # (x1,y1,x2,y2)
    confidence:    float
    plate:         Optional[PlateResult] = None
    helmet:        Optional[bool] = None
    helmet_conf:   float = 0.0
    seatbelt:      Optional[bool] = None
    seatbelt_conf: float = 0.0
    violation:     str = "Compliant"
    timestamp:     float = field(default_factory=time.time)

    def has_violation(self) -> bool:
        return self.violation != "Compliant"


# ── Plate Recogniser ───────────────────────────────────────────────────────

class PlateRecogniser:
    """
    Orchestrates the full plate detection + recognition pipeline.

    Parameters
    ----------
    models   : ModelManager  — pre-loaded model registry
    settings : Settings      — runtime configuration
    """

    def __init__(self, models, settings):
        self.models = models
        self.cfg    = settings

    # ── Public API ────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> list[VehicleDetection]:
        """Process one frame end-to-end. Returns VehicleDetection list."""
        detections: list[VehicleDetection] = []

        vehicles = self._detect_vehicles(frame)

        for vdet in vehicles:
            x1, y1, x2, y2 = vdet["bbox"]
            vcls            = vdet["class"]
            vconf           = vdet["conf"]

            roi = self._safe_crop(frame, x1, y1, x2, y2, pad=10)
            if roi is None:
                continue

            plate_result = self._detect_and_read_plate(frame, roi, x1, y1)

            helmet_ok, helmet_conf, belt_ok, belt_conf = \
                self._check_safety(roi, vcls)

            violation = self._classify_violation(
                vcls, helmet_ok, helmet_conf, belt_ok, belt_conf
            )

            detections.append(VehicleDetection(
                vehicle_class = vcls,
                bbox          = (x1, y1, x2, y2),
                confidence    = vconf,
                plate         = plate_result,
                helmet        = helmet_ok,
                helmet_conf   = helmet_conf,
                seatbelt      = belt_ok,
                seatbelt_conf = belt_conf,
                violation     = violation,
            ))

        return detections

    # ── Stage 1: YOLO detection ────────────────────────────────────

    def _detect_vehicles(self, frame: np.ndarray) -> list[dict]:
        results = []
        try:
            yolo_results = self.models.detector(
                frame,
                conf    = self.cfg.conf_thresh,
                iou     = self.cfg.iou_thresh,
                classes = list(_VEHICLE_CLASSES.keys()),
                verbose = False,
            )
            for r in yolo_results:
                boxes = r.boxes
                for i in range(len(boxes.xyxy)):
                    cls_id = int(boxes.cls[i])
                    if cls_id not in _VEHICLE_CLASSES:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i])
                    results.append({
                        "bbox" : (x1, y1, x2, y2),
                        "class": _VEHICLE_CLASSES[cls_id],
                        "conf" : float(boxes.conf[i]),
                    })
        except Exception as e:
            log.debug(f"YOLO inference error: {e}")
            # CPU fallback simulation (for demo without GPU)
            h, w = frame.shape[:2]
            results.append({
                "bbox" : (w // 6, h // 3, 5 * w // 6, 9 * h // 10),
                "class": "Car",
                "conf" : 0.75,
            })
        return results

    # ── Stage 2: Plate localisation ────────────────────────────────

    def _locate_plate_in_roi(self, roi: np.ndarray) -> Optional[tuple]:
        """
        Find the most plate-like rectangle in the vehicle ROI.
        Uses edge detection + contour aspect-ratio filtering.
        Returns (x1,y1,x2,y2) in ROI coordinates, or None.
        """
        h, w = roi.shape[:2]
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur  = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)
        cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]

        best = None
        best_area = 0
        for c in cnts:
            peri  = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                x, y, cw, ch = cv2.boundingRect(approx)
                area = cw * ch
                if area < self.cfg.plate_min_area:
                    continue
                aspect = cw / max(ch, 1)
                # Indian plates: roughly 2:1 to 5:1 aspect ratio
                if 1.8 <= aspect <= 6.0 and area > best_area:
                    best      = (x, y, x + cw, y + ch)
                    best_area = area

        # Fallback: use bottom-centre of ROI if no plate found
        if best is None:
            x1 = w // 4
            y1 = int(h * 0.6)
            x2 = 3 * w // 4
            y2 = int(h * 0.85)
            area = (x2 - x1) * (y2 - y1)
            if area >= self.cfg.plate_min_area:
                best = (x1, y1, x2, y2)

        return best

    # ── Stage 3 + 4: Enhancement + OCR ────────────────────────────

    def _detect_and_read_plate(
        self,
        full_frame: np.ndarray,
        vehicle_roi: np.ndarray,
        roi_x: int,
        roi_y: int,
    ) -> Optional[PlateResult]:
        plate_bbox_local = self._locate_plate_in_roi(vehicle_roi)
        if plate_bbox_local is None:
            return None

        lx1, ly1, lx2, ly2 = plate_bbox_local
        plate_crop = self._safe_crop(vehicle_roi, lx1, ly1, lx2, ly2)
        if plate_crop is None or plate_crop.size == 0:
            return None
        if plate_crop.shape[0] * plate_crop.shape[1] < self.cfg.plate_min_area:
            return None

        # Absolute bbox in original frame
        abs_bbox = (
            roi_x + lx1, roi_y + ly1,
            roi_x + lx2, roi_y + ly2,
        )

        enhanced       = False
        enhanced_crop  = plate_crop

        # Stage 3: GenAI enhancement (Real-ESRGAN)
        if self.cfg.use_genai and self.models.enhancer is not None:
            try:
                enhanced_crop, _ = self.models.enhancer.enhance(
                    plate_crop, outscale=self.cfg.esrgan_scale
                )
                enhanced = True
            except Exception as e:
                log.debug(f"Enhancement failed: {e}")
                enhanced_crop = plate_crop

        # Stage 4a: Pre-process
        ocr_input = self._preprocess_for_ocr(enhanced_crop)

        # Stage 4b: OCR
        raw_text, confidence = self._run_ocr(ocr_input)
        if not raw_text:
            return None

        # Stage 5: Correct + validate
        corrected = self._correct_plate_text(raw_text)

        return PlateResult(
            text       = corrected,
            confidence = confidence,
            bbox       = abs_bbox,
            plate_crop = enhanced_crop,
            enhanced   = enhanced,
            raw_text   = raw_text,
        )

    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """CLAHE equalisation + Otsu binarisation for better OCR."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
               if len(img.shape) == 3 else img.copy()
        # Resize to standard height
        scale = max(1, 64 / max(gray.shape[0], 1))
        if scale > 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        eq    = clahe.apply(gray)
        # Otsu threshold
        _, bw = cv2.threshold(eq, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw

    def _run_ocr(self, img: np.ndarray) -> tuple[str, float]:
        """Run EasyOCR on pre-processed plate image."""
        if self.models.ocr is None:
            return "", 0.0
        try:
            results = self.models.ocr.readtext(img, detail=1, paragraph=False)
            if not results:
                return "", 0.0
            # Concatenate all detected text blocks
            texts = [r[1] for r in results]
            confs = [r[2] for r in results]
            combined = "".join(texts).upper().strip()
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return combined, avg_conf
        except Exception as e:
            log.debug(f"OCR error: {e}")
            return "", 0.0

    # ── Stage 5: Post-processing ───────────────────────────────────

    def _correct_plate_text(self, raw: str) -> str:
        """
        Apply position-aware character correction for Indian plates.
        Format: SS DD LLL NNNN  (S=state letter, D=digit, L=series letter, N=number)
        """
        clean = re.sub(r"[^A-Z0-9]", "", raw.upper())
        if len(clean) < 6:
            return raw.upper().strip()

        # Build corrected string with position rules
        corrected = list(clean)
        letter_positions = {0, 1}        # state code — must be letters
        digit_positions  = {2, 3}        # district — must be digits
        # series (4..6) and number (rest): mixed

        for i, ch in enumerate(corrected):
            if i in letter_positions and ch.isdigit():
                # digit in letter slot — try reverse map
                rev = {v: k for k, v in _OCR_FIXES.items()}
                corrected[i] = rev.get(ch, ch)
            elif i in digit_positions and ch.isalpha():
                corrected[i] = _OCR_FIXES.get(ch, ch)

        return "".join(corrected)

    # ── Safety compliance ──────────────────────────────────────────

    def _check_safety(
        self, roi: np.ndarray, vehicle_class: str
    ) -> tuple[Optional[bool], float, Optional[bool], float]:
        """
        Returns (helmet_ok, helmet_conf, seatbelt_ok, seatbelt_conf).
        Uses MobileNetV3 classifier if available, otherwise heuristics.
        """
        helmet_ok   = None
        helmet_conf = 0.0
        belt_ok     = None
        belt_conf   = 0.0

        try:
            clf = self.models.safety_classifier
            if clf is None:
                raise AttributeError("No classifier")

            result = clf.classify(roi)
            helmet_ok   = result.get("helmet")
            helmet_conf = result.get("helmet_conf", 0.0)
            belt_ok     = result.get("seatbelt")
            belt_conf   = result.get("seatbelt_conf", 0.0)
        except Exception:
            # Heuristic fallback using brightness as crude proxy
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_bright = float(np.mean(gray))
            if vehicle_class == "Motorcycle":
                # Upper-third ROI brightness heuristic for helmet
                h = roi.shape[0]
                upper = cv2.cvtColor(roi[:h//3], cv2.COLOR_BGR2GRAY)
                helmet_ok   = float(np.mean(upper)) > 60
                helmet_conf = 0.65
            elif vehicle_class in ("Car", "Bus", "Truck"):
                belt_ok   = mean_bright > 50
                belt_conf = 0.60

        return helmet_ok, helmet_conf, belt_ok, belt_conf

    def _classify_violation(
        self,
        vehicle_class: str,
        helmet_ok: Optional[bool],  helmet_conf: float,
        belt_ok:   Optional[bool],  belt_conf:   float,
    ) -> str:
        if vehicle_class == "Motorcycle":
            if helmet_ok is False and helmet_conf >= self.cfg.helmet_conf:
                return "No Helmet"
        elif vehicle_class in ("Car", "Bus", "Truck"):
            if belt_ok is False and belt_conf >= self.cfg.seatbelt_conf:
                return "No Seat Belt"
        return "Compliant"

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _safe_crop(
        img: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        pad: int = 0,
    ) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2].copy()
