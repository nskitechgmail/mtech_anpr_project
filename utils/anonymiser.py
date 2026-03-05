"""
utils/anonymiser.py — Face anonymisation for DPDP Act 2023 compliance.

Uses MediaPipe BlazeFace as primary detector with OpenCV Haar cascade fallback.
Applies padded Gaussian blur to all detected faces before image storage.
"""
from __future__ import annotations
import cv2, logging
import numpy as np

log = logging.getLogger("FaceAnonymiser")

# Padding factor around detected face bbox
_PAD_FACTOR = 0.15
# Blur kernel size (must be odd)
_BLUR_KERNEL = (51, 51)


class FaceAnonymiser:
    """
    Detects and blurs all human faces in a frame.

    Initialization is lazy — models are loaded on first call to blur_faces().
    Operates entirely on CPU; ~30 ms per 1080p frame.
    """

    def __init__(self):
        self._mp_face    = None
        self._haar       = None
        self._initialized = False

    def _init_models(self):
        if self._initialized:
            return
        # Try MediaPipe first
        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0,          # short-range (< 2 m)
                min_detection_confidence=0.5,
            )
            log.info("FaceAnonymiser: MediaPipe BlazeFace loaded.")
        except Exception as e:
            log.warning(f"MediaPipe unavailable ({e}); falling back to Haar cascade.")

        # Always load Haar as backup
        try:
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            log.info("FaceAnonymiser: Haar cascade loaded.")
        except Exception as e:
            log.warning(f"Haar cascade unavailable: {e}")

        self._initialized = True

    def blur_faces(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect and blur all faces in *frame* (BGR).
        Returns a copy with blurred faces; original is unchanged.
        """
        self._init_models()
        out = frame.copy()
        h, w = out.shape[:2]

        bboxes = self._detect_faces(out)
        for (x1, y1, x2, y2) in bboxes:
            # Add padding
            pw = int((x2 - x1) * _PAD_FACTOR)
            ph = int((y2 - y1) * _PAD_FACTOR)
            rx1 = max(0, x1 - pw)
            ry1 = max(0, y1 - ph)
            rx2 = min(w, x2 + pw)
            ry2 = min(h, y2 + ph)
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            roi = out[ry1:ry2, rx1:rx2]
            out[ry1:ry2, rx1:rx2] = cv2.GaussianBlur(roi, _BLUR_KERNEL, 0)

        return out

    def _detect_faces(self, frame: np.ndarray) -> list[tuple]:
        """Return list of (x1, y1, x2, y2) face bounding boxes."""
        bboxes = []
        h, w = frame.shape[:2]

        if self._mp_face is not None:
            try:
                import mediapipe as mp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._mp_face.process(rgb)
                if results.detections:
                    for det in results.detections:
                        bb = det.location_data.relative_bounding_box
                        x1 = int(bb.xmin * w)
                        y1 = int(bb.ymin * h)
                        x2 = int((bb.xmin + bb.width)  * w)
                        y2 = int((bb.ymin + bb.height) * h)
                        bboxes.append((x1, y1, x2, y2))
                return bboxes
            except Exception as e:
                log.debug(f"MediaPipe face detection error: {e}")

        # Haar fallback
        if self._haar is not None:
            try:
                grey   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces  = self._haar.detectMultiScale(grey, 1.1, 4, minSize=(30, 30))
                for (fx, fy, fw, fh) in faces:
                    bboxes.append((fx, fy, fx + fw, fy + fh))
            except Exception as e:
                log.debug(f"Haar detection error: {e}")

        return bboxes
