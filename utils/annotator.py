"""
utils/annotator.py — Draws detection bounding boxes, plate text,
safety compliance badges, and HUD overlay onto video frames.
"""

from __future__ import annotations
import cv2, time
import numpy as np
from core.plate_recogniser import VehicleDetection, FrameStats

# ── Colour palette ─────────────────────────────────────────────────────────
COLOURS = {
    "Compliant"    : (50,  205, 50),   # green
    "No Helmet"    : (0,   0,  220),   # red (BGR)
    "No Seat Belt" : (0,   140, 255),  # orange
    "Speed"        : (0,   220, 220),  # yellow
    "Unknown"      : (180, 180, 180),  # grey
    "plate_box"    : (255, 200, 0),    # cyan
    "hud_bg"       : (20,  20,  30),
    "hud_text"     : (200, 230, 255),
    "hud_accent"   : (0,   200, 255),
    "hud_warning"  : (0,   80,  255),
}

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


class FrameAnnotator:
    def __init__(self, settings):
        self.cfg = settings

    # ── Main entry point ───────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        detections: list[VehicleDetection],
        stats: "FrameStats",
    ) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            self._draw_vehicle(out, det)
        self._draw_hud(out, detections, stats)
        return out

    # ── Vehicle bounding box + labels ─────────────────────────────────────

    def _draw_vehicle(self, frame: np.ndarray, det: VehicleDetection):
        x1, y1, x2, y2 = det.bbox
        col = COLOURS.get(det.violation, COLOURS["Unknown"])

        # Thick bounding box with corner highlights
        self._draw_rect_corners(frame, x1, y1, x2, y2, col)

        # Vehicle label bar
        label = f"{det.vehicle_class}  {det.confidence:.0%}"
        self._label_bar(frame, x1, y1, label, col, above=True)

        # Violation badge
        if det.violation != "Compliant":
            self._violation_badge(frame, x1, y2, det.violation, col)
        else:
            self._small_badge(frame, x1, y2, "✓ Compliant", COLOURS["Compliant"])

        # Safety indicators (small icons top-right of bbox)
        self._draw_safety_icons(frame, x2, y1, det)

        # Plate result
        if det.plate:
            self._draw_plate_overlay(frame, det)

    def _draw_rect_corners(self, frame, x1, y1, x2, y2, col, thick=2, corner=20):
        """Draw corner-style bounding box (no full rectangle, just corners)."""
        pts = [
            # top-left
            [(x1,y1),(x1+corner,y1)], [(x1,y1),(x1,y1+corner)],
            # top-right
            [(x2,y1),(x2-corner,y1)], [(x2,y1),(x2,y1+corner)],
            # bottom-left
            [(x1,y2),(x1+corner,y2)], [(x1,y2),(x1,y2-corner)],
            # bottom-right
            [(x2,y2),(x2-corner,y2)], [(x2,y2),(x2,y2-corner)],
        ]
        for p1,p2 in pts:
            cv2.line(frame, p1, p2, col, thick+1)
        # thin full rectangle
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 1)

    def _label_bar(self, frame, x, y, text, col, above=True):
        (tw, th), bl = cv2.getTextSize(text, FONT, 0.55, 1)
        by = y - 6 if above else y + th + 6
        cv2.rectangle(frame, (x, by-th-4), (x+tw+8, by+4), col, -1)
        cv2.putText(frame, text, (x+4, by), FONT, 0.55, (255,255,255), 1, cv2.LINE_AA)

    def _violation_badge(self, frame, x, y, text, col):
        badge = f"⚠ {text}"
        (tw, th), _ = cv2.getTextSize(badge, FONT, 0.60, 2)
        # Flashing effect based on time
        alpha = 0.85 + 0.15 * abs(np.sin(time.time() * 4))
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y+4), (x+tw+12, y+th+16), col, -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        cv2.putText(frame, badge, (x+6, y+th+8), FONT, 0.60, (255,255,255), 2, cv2.LINE_AA)

    def _small_badge(self, frame, x, y, text, col):
        (tw, th), _ = cv2.getTextSize(text, FONT_SMALL, 0.50, 1)
        cv2.rectangle(frame, (x, y+2), (x+tw+8, y+th+10), col, -1)
        cv2.putText(frame, text, (x+4, y+th+4), FONT_SMALL, 0.50, (0,0,0), 1, cv2.LINE_AA)

    def _draw_safety_icons(self, frame, x2, y1, det: VehicleDetection):
        """Draw small helmet / belt status icons at top-right of vehicle box."""
        ix = x2 + 4; iy = y1
        if det.helmet is not None:
            col  = COLOURS["Compliant"] if det.helmet else COLOURS["No Helmet"]
            icon = "H:OK" if det.helmet else "H:NO"
            cv2.rectangle(frame,(ix,iy),(ix+46,iy+20),col,-1)
            cv2.putText(frame,icon,(ix+2,iy+15),FONT_SMALL,0.45,(255,255,255),1,cv2.LINE_AA)
            iy += 24
        if det.seatbelt is not None:
            col  = COLOURS["Compliant"] if det.seatbelt else COLOURS["No Seat Belt"]
            icon = "B:OK" if det.seatbelt else "B:NO"
            cv2.rectangle(frame,(ix,iy),(ix+46,iy+20),col,-1)
            cv2.putText(frame,icon,(ix+2,iy+15),FONT_SMALL,0.45,(255,255,255),1,cv2.LINE_AA)

    def _draw_plate_overlay(self, frame: np.ndarray, det: VehicleDetection):
        plate = det.plate
        px1,py1,px2,py2 = plate.bbox
        conf_pct = int(plate.confidence * 100)

        # Plate bounding box
        cv2.rectangle(frame, (px1,py1), (px2,py2), COLOURS["plate_box"], 2)

        # Plate text below the plate box
        text  = plate.normalised()
        label = f"{text}  ({conf_pct}%)"
        if plate.enhanced:
            label += " [AI]"
        (tw,th),_ = cv2.getTextSize(label, FONT, 0.65, 2)
        tx = max(0, px1)
        ty = py2 + th + 10
        if ty > frame.shape[0] - 5:
            ty = py1 - 6
        cv2.rectangle(frame,(tx-2,ty-th-4),(tx+tw+4,ty+4),(0,30,80),-1)
        cv2.putText(frame, label, (tx, ty), FONT, 0.65, (0,255,255), 2, cv2.LINE_AA)

        # Inset enhanced plate thumbnail (top-right corner of vehicle box)
        if self.cfg.show_plate_crop and plate.plate_crop is not None:
            try:
                crop = plate.plate_crop
                th2 = 40
                tw2 = int(crop.shape[1] * th2 / crop.shape[0])
                thumb = cv2.resize(crop, (tw2, th2))
                vx2 = det.bbox[2]
                vy1 = det.bbox[1]
                if len(thumb.shape) == 2:
                    thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
                # Place inset
                ex = max(0, vx2-tw2-4)
                ey = vy1 + 24
                if ex >= 0 and ey >= 0 and ex+tw2 < frame.shape[1] and ey+th2 < frame.shape[0]:
                    frame[ey:ey+th2, ex:ex+tw2] = thumb
                    cv2.rectangle(frame,(ex,ey),(ex+tw2,ey+th2),(255,200,0),1)
            except Exception:
                pass

    # ── HUD overlay ────────────────────────────────────────────────────────

    def _draw_hud(
        self,
        frame: np.ndarray,
        detections: list[VehicleDetection],
        stats: "FrameStats",
    ):
        h, w = frame.shape[:2]
        violations = [d for d in detections if d.has_violation()]

        # Top HUD bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,36), COLOURS["hud_bg"], -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        title = "SMART CITY ANPR  |  SRM Institute of Science & Technology"
        cv2.putText(frame, title, (10,24), FONT_SMALL, 0.55,
                    COLOURS["hud_accent"], 1, cv2.LINE_AA)

        # Right side: camera + time
        ts = time.strftime("%d/%m/%Y  %H:%M:%S")
        cam_info = f"{self.cfg.camera_id}  |  {ts}"
        (cw,_),_ = cv2.getTextSize(cam_info, FONT_SMALL, 0.50, 1)
        cv2.putText(frame, cam_info, (w-cw-10, 24), FONT_SMALL, 0.50,
                    COLOURS["hud_text"], 1, cv2.LINE_AA)

        # Bottom stats bar
        bh = 32
        overlay2 = frame.copy()
        cv2.rectangle(overlay2,(0,h-bh),(w,h),COLOURS["hud_bg"],-1)
        cv2.addWeighted(overlay2,0.80,frame,0.20,0,frame)

        # Stats row
        stats_items = [
            ("FPS",       f"{stats.fps:.1f}"),
            ("Vehicles",  str(len(detections))),
            ("Plates",    str(sum(1 for d in detections if d.plate))),
            ("Violations",str(len(violations))),
            ("Frame",     str(stats.frame_num)),
            ("GenAI",     "ON" if self.cfg.use_genai else "OFF"),
        ]
        sx = 10
        for label, val in stats_items:
            col = COLOURS["hud_warning"] if (label=="Violations" and violations) \
                  else COLOURS["hud_accent"]
            text = f"{label}: {val}"
            cv2.putText(frame, text, (sx, h-10), FONT_SMALL, 0.50, col, 1, cv2.LINE_AA)
            (tw,_),_ = cv2.getTextSize(text, FONT_SMALL, 0.50, 1)
            sx += tw + 25

        # Violation alert banner
        if violations:
            banner = f"⚠  {len(violations)} VIOLATION(S) DETECTED"
            (bw,bh2),_ = cv2.getTextSize(banner, FONT, 0.75, 2)
            bx = (w - bw) // 2
            # Pulsing red bar
            alpha = 0.6 + 0.4 * abs(np.sin(time.time() * 3))
            ov3 = frame.copy()
            cv2.rectangle(ov3,(bx-10,38),(bx+bw+10,72),(0,0,180),-1)
            cv2.addWeighted(ov3,alpha,frame,1-alpha,0,frame)
            cv2.putText(frame, banner, (bx, 65), FONT, 0.75,
                        (255,255,255), 2, cv2.LINE_AA)
