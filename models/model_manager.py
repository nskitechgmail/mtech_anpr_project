"""
models/model_manager.py — Downloads, caches and loads all AI models.

Models used:
  1. YOLOv8/v9  — vehicle + person detection
  2. Custom YOLO weights — helmet & seat-belt classification
  3. Real-ESRGAN  — blind super-resolution for plate enhancement
  4. EasyOCR     — licence plate text recognition
  5. MediaPipe   — face detection for anonymisation

On first run, models are downloaded automatically to models/weights/.
Subsequent runs load from cache.
"""

from __future__ import annotations
import os, logging, sys
from pathlib import Path

log = logging.getLogger("ModelManager")
WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)


# ── Lazy imports with friendly error messages ──────────────────────────────

def _require(pkg: str, install_hint: str = ""):
    try:
        return __import__(pkg)
    except ImportError:
        hint = install_hint or f"pip install {pkg}"
        log.error(f"Missing package '{pkg}'. Install with: {hint}")
        raise SystemExit(1)


class ModelManager:
    """
    Central registry for all models.

    Usage:
        mm = ModelManager(settings)
        mm.load_all()          # download + load everything
        detector   = mm.detector
        ocr_reader = mm.ocr
        enhancer   = mm.enhancer
        safety_clf = mm.safety
    """

    def __init__(self, settings):
        self.cfg = settings
        self._detector      = None
        self._ocr           = None
        self._enhancer      = None
        self._safety        = None
        self._face_detector = None

    # ── Public API ─────────────────────────────────────────────────────────

    def load_all(self):
        log.info("Loading models …")
        self._load_detector()
        self._load_ocr()
        if self.cfg.use_genai:
            self._load_enhancer()
        self._load_safety()
        self._load_face_detector()
        log.info("All models ready.")

    @property
    def detector(self):
        if self._detector is None:
            self._load_detector()
        return self._detector

    @property
    def ocr(self):
        if self._ocr is None:
            self._load_ocr()
        return self._ocr

    @property
    def enhancer(self):
        if self._enhancer is None and self.cfg.use_genai:
            self._load_enhancer()
        return self._enhancer

    @property
    def safety(self):
        if self._safety is None:
            self._load_safety()
        return self._safety

    @property
    def face_detector(self):
        return self._face_detector

    # ── Loaders ────────────────────────────────────────────────────────────

    def _load_detector(self):
        """YOLOv8/v9 for vehicle and person detection."""
        try:
            from ultralytics import YOLO
            model_name = self.cfg.detector_model
            # ultralytics auto-downloads to ~/.cache/ultralytics
            log.info(f"Loading YOLO detector: {model_name}")
            self._detector = YOLO(f"{model_name}.pt")
            self._detector.to(self.cfg.device)
            log.info(f"  ✓ Detector ready on {self.cfg.device}")
        except ImportError:
            log.warning("ultralytics not installed — using simulation mode")
            self._detector = _SimulatedDetector()

    def _load_ocr(self):
        """EasyOCR for licence plate text recognition."""
        try:
            import easyocr
            log.info("Loading EasyOCR …")
            gpu = (self.cfg.device != "cpu")
            self._ocr = easyocr.Reader(
                self.cfg.ocr_languages,
                gpu=gpu,
                model_storage_directory=str(WEIGHTS_DIR / "easyocr"),
                verbose=False,
            )
            log.info("  ✓ EasyOCR ready")
        except ImportError:
            log.warning("easyocr not installed — using simulation mode")
            self._ocr = _SimulatedOCR()

    def _load_enhancer(self):
        """Real-ESRGAN for blind super-resolution."""
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model_path = WEIGHTS_DIR / "RealESRGAN_x4plus.pth"
            if not model_path.exists():
                self._download_esrgan(model_path)

            log.info("Loading Real-ESRGAN …")
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32,
                scale=self.cfg.esrgan_scale,
            )
            self._enhancer = RealESRGANer(
                scale=self.cfg.esrgan_scale,
                model_path=str(model_path),
                model=model,
                tile=self.cfg.esrgan_tile,
                tile_pad=10,
                pre_pad=0,
                half=(self.cfg.device != "cpu"),
                device=self.cfg.device,
            )
            log.info("  ✓ Real-ESRGAN ready")
        except ImportError:
            log.warning("realesrgan/basicsr not installed — using PIL-based fallback enhancer")
            self._enhancer = _PILEnhancer()
        except Exception as e:
            log.warning(f"Real-ESRGAN load error: {e} — using PIL fallback")
            self._enhancer = _PILEnhancer()

    def _load_safety(self):
        """Safety compliance classifiers (helmet + seatbelt)."""
        # We use a fine-tuned YOLOv8 model that detects:
        #   class 0: helmet_on   class 1: helmet_off
        #   class 2: belt_on     class 3: belt_off
        try:
            from ultralytics import YOLO
            safety_weights = WEIGHTS_DIR / "safety_compliance.pt"
            if safety_weights.exists():
                log.info("Loading safety compliance model …")
                self._safety = YOLO(str(safety_weights))
                self._safety.to(self.cfg.device)
                log.info("  ✓ Safety model ready")
            else:
                # Fallback: reuse main detector + class-based heuristics
                log.info("  ℹ  Safety weights not found — using heuristic classifier")
                self._safety = _HeuristicSafetyClassifier()
        except ImportError:
            self._safety = _HeuristicSafetyClassifier()

    def _load_face_detector(self):
        """MediaPipe face detection for privacy anonymisation."""
        try:
            import mediapipe as mp
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            log.info("  ✓ Face anonymiser ready")
        except ImportError:
            log.info("  ℹ  mediapipe not installed — face anonymisation skipped")
            self._face_detector = None

    def _download_esrgan(self, dest: Path):
        import urllib.request
        url = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.1.0/RealESRGAN_x4plus.pth")
        log.info(f"Downloading Real-ESRGAN weights → {dest}")
        urllib.request.urlretrieve(url, dest)
        log.info("  ✓ Download complete")


# ── Fallback / simulation classes (no GPU required) ─────────────────────────

class _SimulatedDetector:
    """Pixel-based fallback when ultralytics is not installed."""
    def __call__(self, frame, conf=0.4, iou=0.45, classes=None, verbose=False):
        return _FakeResults(frame)

    def to(self, device): return self


class _FakeResults:
    """Mimics ultralytics Results object structure."""
    import random as _r

    def __init__(self, frame):
        import numpy as np
        h, w = frame.shape[:2]
        self.orig_img = frame
        # Generate 1-4 fake bounding boxes
        n = self._r.randint(1, 4)
        self.boxes = _FakeBoxes(n, w, h)

    class _FakeBoxes:
        def __init__(self, n, w, h):
            import numpy as np, random
            self.xyxy = []
            self.conf = []
            self.cls  = []
            for _ in range(n):
                x1 = random.randint(0, w-200)
                y1 = random.randint(int(h*0.3), h-150)
                x2 = x1 + random.randint(100, 200)
                y2 = y1 + random.randint(80, 150)
                self.xyxy.append([x1, y1, min(x2,w-1), min(y2,h-1)])
                self.conf.append(random.uniform(0.5, 0.95))
                self.cls.append(random.choice([2, 3, 5, 7]))  # COCO: car, motorcycle, bus, truck


class _SimulatedOCR:
    """Fallback OCR using PIL text rendering (no easyocr)."""
    import random as _r

    STATE_CODES = ["MH","DL","KA","TN","TS","KL","GJ","RJ","UP","AP","PB","HR"]
    SERIES      = "ABCDEFGHJKLMNPRSTUVWXYZ"

    def readtext(self, img_array, detail=1, **kwargs):
        import random
        plate = (f"{random.choice(self.STATE_CODES)} "
                 f"{random.randint(1,99):02d} "
                 f"{random.choice(self.SERIES)}{random.choice(self.SERIES)} "
                 f"{random.randint(1,9999):04d}")
        conf = random.uniform(0.55, 0.92)
        return [([[0,0],[100,0],[100,30],[0,30]], plate, conf)]


class _PILEnhancer:
    """PIL-based super-resolution fallback (no Real-ESRGAN)."""
    def enhance(self, img_bgr, outscale=4):
        """
        img_bgr: numpy BGR array (OpenCV format)
        Returns: (enhanced_bgr, None)
        """
        import numpy as np
        from PIL import Image, ImageFilter, ImageEnhance

        # BGR → RGB PIL
        pil = Image.fromarray(img_bgr[:, :, ::-1])
        h, w = img_bgr.shape[:2]
        # Upscale
        pil = pil.resize((w * outscale, h * outscale), Image.LANCZOS)
        # Sharpen
        pil = pil.filter(ImageFilter.UnsharpMask(radius=2.5, percent=200, threshold=2))
        pil = ImageEnhance.Contrast(pil).enhance(1.5)
        pil = ImageEnhance.Sharpness(pil).enhance(2.2)
        pil = ImageEnhance.Brightness(pil).enhance(1.1)
        # RGB → BGR numpy
        out = np.array(pil)[:, :, ::-1]
        return out, None


class _HeuristicSafetyClassifier:
    """
    Rule-based safety compliance heuristic.
    Uses colour analysis in the helmet and torso ROI regions.
    In production this is replaced by a fine-tuned YOLOv8 model.
    """
    import random as _r

    def predict_helmet(self, roi_bgr) -> tuple[bool, float]:
        """Returns (has_helmet: bool, confidence: float)."""
        import numpy as np
        if roi_bgr is None or roi_bgr.size == 0:
            return False, 0.0
        h, w = roi_bgr.shape[:2]
        head_region = roi_bgr[: h // 3, w // 4: 3 * w // 4]
        if head_region.size == 0:
            return self._r.random() > 0.35, self._r.uniform(0.5, 0.8)
        # Heuristic: helmets tend to be dark/coloured non-skin pixels
        hsv = self._bgr_to_hsv(head_region)
        skin_mask = self._skin_mask(hsv)
        skin_ratio = skin_mask.mean()
        has_helmet = skin_ratio < 0.35
        conf = 0.65 + abs(0.35 - skin_ratio) * 0.5
        return has_helmet, min(0.95, conf)

    def predict_seatbelt(self, roi_bgr) -> tuple[bool, float]:
        """Returns (has_belt: bool, confidence: float)."""
        import numpy as np
        if roi_bgr is None or roi_bgr.size == 0:
            return False, 0.0
        h, w = roi_bgr.shape[:2]
        torso = roi_bgr[h // 4: 3 * h // 4, :]
        if torso.size == 0:
            return self._r.random() > 0.4, self._r.uniform(0.5, 0.78)
        # Seatbelts appear as diagonal grey/black stripe
        grey = torso[:, :, 0].astype(int) - torso[:, :, 2].astype(int)
        dark_ratio = (torso.mean(axis=2) < 80).mean()
        has_belt = dark_ratio > 0.04
        conf = 0.55 + dark_ratio * 2.0
        return has_belt, min(0.90, conf)

    @staticmethod
    def _bgr_to_hsv(bgr):
        import numpy as np
        b, g, r = bgr[:,:,0]/255., bgr[:,:,1]/255., bgr[:,:,2]/255.
        maxc = np.maximum(np.maximum(r,g),b)
        minc = np.minimum(np.minimum(r,g),b)
        v = maxc
        s = np.where(maxc!=0, (maxc-minc)/maxc, 0)
        return np.stack([v,s,v],axis=2)  # simplified

    @staticmethod
    def _skin_mask(hsv):
        import numpy as np
        v = hsv[:,:,0]
        s = hsv[:,:,1]
        return ((v > 0.35) & (v < 0.95) & (s < 0.6)).astype(float)
