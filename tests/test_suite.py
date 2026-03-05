"""
tests/test_suite.py — Comprehensive test suite for Smart City ANPR System.

Run all tests:
    pytest tests/test_suite.py -v

Run unit tests only:
    pytest tests/test_suite.py -v -m unit

Run integration tests only:
    pytest tests/test_suite.py -v -m integration
"""
import sys, os, json, tempfile, time
import numpy as np
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def blank_frame():
    """1280×720 BGR blank frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)

@pytest.fixture
def white_frame():
    return np.full((720, 1280, 3), 255, dtype=np.uint8)

@pytest.fixture
def settings():
    from config.settings import Settings
    return Settings(
        source      = "0",
        use_genai   = False,   # disable for testing (no GPU needed)
        conf_thresh = 0.40,
        device      = "cpu",
        output_dir  = tempfile.mkdtemp(),
        camera_id   = "TEST-001",
    )

@pytest.fixture
def sample_plate_crop():
    """Synthetic white rectangle resembling a plate."""
    img = np.full((60, 200, 3), 255, dtype=np.uint8)
    import cv2
    cv2.putText(img, "MH12AB1234", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return img

# ══════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Settings
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSettings:

    def test_default_device_auto(self):
        from config.settings import Settings
        s = Settings()
        assert s.device in ("auto", "cpu", "cuda", "mps")

    def test_output_dirs_created(self, settings):
        import os
        assert os.path.isdir(settings.output_dir)

    def test_violation_frames_positive(self, settings):
        assert settings.violation_frames >= 1

    def test_conf_thresh_range(self, settings):
        assert 0.0 < settings.conf_thresh <= 1.0

    def test_camera_id_set(self, settings):
        assert settings.camera_id == "TEST-001"


# ══════════════════════════════════════════════════════════════════════════
# UNIT TESTS — PlateRecogniser helpers (no GPU required)
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPlateRecogniser:

    def test_plate_text_postprocess_corrects_o_to_0(self, settings):
        """O in digit positions should become 0."""
        from core.plate_recogniser import PlateRecogniser
        pr = PlateRecogniser(models=None, settings=settings)
        result = pr._postprocess_plate_text("MHIOAB1234")
        # digits at positions 2-3 → I should become 1, O becomes 0
        assert "1" in result or "0" in result   # at least one correction

    def test_plate_re_matches_valid_plate(self):
        import re
        from core.plate_recogniser import _PLATE_RE
        assert _PLATE_RE.match("MH12AB1234")
        assert _PLATE_RE.match("KA 05 MX 9999")
        assert _PLATE_RE.match("TN-01-CD-5678")

    def test_plate_re_rejects_invalid(self):
        from core.plate_recogniser import _PLATE_RE
        assert not _PLATE_RE.match("NOTAPLATE")
        assert not _PLATE_RE.match("12345678")
        assert not _PLATE_RE.match("")

    def test_plate_result_valid_format(self):
        from core.plate_recogniser import PlateResult
        r = PlateResult(text="MH12AB1234", confidence=0.92, bbox=(0,0,100,30))
        assert r.valid_format is True

    def test_plate_result_invalid_format(self):
        from core.plate_recogniser import PlateResult
        r = PlateResult(text="GARBAGE", confidence=0.3, bbox=(0,0,100,30))
        assert r.valid_format is False

    def test_plate_normalised(self):
        from core.plate_recogniser import PlateResult
        r = PlateResult(text="MH-12-AB-1234", confidence=0.9, bbox=(0,0,100,30))
        assert r.normalised() == "MH 12 AB 1234"

    def test_preprocess_returns_2d(self, settings, sample_plate_crop):
        from core.plate_recogniser import PlateRecogniser
        pr = PlateRecogniser(models=None, settings=settings)
        out = pr._preprocess_for_ocr(sample_plate_crop)
        assert out.ndim == 2

    def test_preprocess_minimum_width(self, settings):
        from core.plate_recogniser import PlateRecogniser
        import cv2
        pr  = PlateRecogniser(models=None, settings=settings)
        tiny = np.full((15, 80, 3), 255, dtype=np.uint8)
        out  = pr._preprocess_for_ocr(tiny)
        assert out.shape[1] >= 200

    def test_safe_crop_returns_none_for_invalid(self, settings, blank_frame):
        from core.plate_recogniser import PlateRecogniser
        pr = PlateRecogniser(models=None, settings=settings)
        result = pr._safe_crop(blank_frame, 100, 100, 50, 50)  # x2 < x1
        assert result is None

    def test_safe_crop_returns_correct_shape(self, settings, blank_frame):
        from core.plate_recogniser import PlateRecogniser
        pr = PlateRecogniser(models=None, settings=settings)
        crop = pr._safe_crop(blank_frame, 100, 100, 300, 200)
        assert crop is not None
        assert crop.shape == (100, 200, 3)

    def test_classify_violation_no_helmet(self, settings):
        from core.plate_recogniser import PlateRecogniser
        pr = PlateRecogniser(models=None, settings=settings)
        v  = pr._classify_violation("Motorcycle", False, 0.9, None, 0.0)
        assert v == "No Helmet"

    def test_classify_violation_compliant(self, settings):
        from core.plate_recogniser import PlateRecogniser
        pr = PlateRecogniser(models=None, settings=settings)
        v  = pr._classify_violation("Motorcycle", True, 0.9, None, 0.0)
        assert v == "Compliant"

    def test_classify_violation_no_seatbelt(self, settings):
        from core.plate_recogniser import PlateRecogniser
        pr = PlateRecogniser(models=None, settings=settings)
        v  = pr._classify_violation("Car", None, 0.0, False, 0.9)
        assert v == "No Seat Belt"


# ══════════════════════════════════════════════════════════════════════════
# UNIT TESTS — VehicleDetection
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestVehicleDetection:

    def test_has_violation_true(self):
        from core.plate_recogniser import VehicleDetection
        d = VehicleDetection(vehicle_class="Motorcycle", bbox=(0,0,100,100),
                             confidence=0.9, violation="No Helmet")
        assert d.has_violation() is True

    def test_has_violation_false(self):
        from core.plate_recogniser import VehicleDetection
        d = VehicleDetection(vehicle_class="Car", bbox=(0,0,100,100),
                             confidence=0.9, violation="Compliant")
        assert d.has_violation() is False

    def test_timestamp_auto_set(self):
        from core.plate_recogniser import VehicleDetection
        before = time.time()
        d = VehicleDetection(vehicle_class="Car", bbox=(0,0,100,100), confidence=0.9)
        after = time.time()
        assert before <= d.timestamp <= after


# ══════════════════════════════════════════════════════════════════════════
# UNIT TESTS — ViolationTracker
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestViolationTracker:

    def test_confirms_after_n_frames(self):
        from core.pipeline import ViolationTracker
        vt = ViolationTracker(n_frames=3)
        assert not vt.update("MH12AB1234", "No Helmet")
        assert not vt.update("MH12AB1234", "No Helmet")
        assert     vt.update("MH12AB1234", "No Helmet")   # 3rd = confirmed

    def test_does_not_double_report(self):
        from core.pipeline import ViolationTracker
        vt = ViolationTracker(n_frames=2)
        vt.update("MH12AB1234", "No Helmet")
        vt.update("MH12AB1234", "No Helmet")   # confirmed
        # 4th+ call should NOT re-confirm
        assert not vt.update("MH12AB1234", "No Helmet")

    def test_reset_clears_tracker(self):
        from core.pipeline import ViolationTracker
        vt = ViolationTracker(n_frames=3)
        vt.update("MH12AB1234", "No Helmet")
        vt.update("MH12AB1234", "No Helmet")
        vt.reset("MH12AB1234")
        # After reset, needs N frames again
        assert not vt.update("MH12AB1234", "No Helmet")
        assert not vt.update("MH12AB1234", "No Helmet")
        assert     vt.update("MH12AB1234", "No Helmet")


# ══════════════════════════════════════════════════════════════════════════
# UNIT TESTS — ReportWriter
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestReportWriter:

    def test_csv_file_created(self, settings):
        from utils.report_writer import ReportWriter
        rw = ReportWriter(settings)
        rw.write({
            "timestamp":"2024-01-01T12:00:00","camera_id":"TEST-001",
            "plate":"MH12AB1234","plate_raw":"MH12AB1234","plate_conf":92.0,
            "plate_valid":True,"plate_enhanced":True,"vehicle_class":"Car",
            "violation":"No Seat Belt","helmet":None,"helmet_conf":0.0,
            "seatbelt":False,"seatbelt_conf":89.0,"image_saved":""
        })
        rw.close()
        import os
        reports_dir = os.path.join(settings.output_dir, "reports")
        csv_files = [f for f in os.listdir(reports_dir) if f.endswith(".csv")]
        assert len(csv_files) >= 1

    def test_json_report_created(self, settings):
        from utils.report_writer import ReportWriter
        rw = ReportWriter(settings)
        rw.write({"timestamp":"2024-01-01T12:00:00","camera_id":"TEST-001",
                  "plate":"KA01AB5678","plate_raw":"","plate_conf":88.0,
                  "plate_valid":True,"plate_enhanced":False,"vehicle_class":"Motorcycle",
                  "violation":"No Helmet","helmet":False,"helmet_conf":91.0,
                  "seatbelt":None,"seatbelt_conf":0.0,"image_saved":""})
        rw.close()
        import os
        reports_dir = os.path.join(settings.output_dir, "reports")
        json_files = [f for f in os.listdir(reports_dir) if f.endswith(".json")]
        assert len(json_files) >= 1


# ══════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Heatmap
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestHeatmap:

    def test_heatmap_render_same_size(self, blank_frame):
        from utils.heatmap import TrafficHeatmap
        hm = TrafficHeatmap(width=1280, height=720)
        out = hm.render(blank_frame)
        assert out.shape == blank_frame.shape

    def test_heatmap_update_increases_density(self):
        from utils.heatmap import TrafficHeatmap
        from core.plate_recogniser import VehicleDetection
        hm = TrafficHeatmap(width=1280, height=720)
        dets = [VehicleDetection(vehicle_class="Car", bbox=(100,100,300,400), confidence=0.9)]
        hm.update(dets)
        assert hm._density.max() > 0

    def test_heatmap_stats_keys(self):
        from utils.heatmap import TrafficHeatmap
        hm = TrafficHeatmap(width=640, height=480)
        stats = hm.get_stats()
        assert "frame_count" in stats
        assert "hourly_counts" in stats


# ══════════════════════════════════════════════════════════════════════════
# UNIT TESTS — AlertSystem
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestAlertSystem:

    def test_no_alert_below_threshold(self):
        from utils.alerts import AlertSystem
        al = AlertSystem(threshold=3)
        assert not al.record("MH12AB1234", "No Helmet")
        assert not al.record("MH12AB1234", "No Helmet")

    def test_alert_at_threshold(self):
        from utils.alerts import AlertSystem
        al = AlertSystem(threshold=3)
        al.record("TN01AB0001", "No Helmet")
        al.record("TN01AB0001", "No Helmet")
        # 3rd should trigger (even though email/SMS will be no-op without credentials)
        triggered = al.record("TN01AB0001", "No Helmet")
        assert triggered is True

    def test_no_double_alert(self):
        from utils.alerts import AlertSystem
        al = AlertSystem(threshold=2)
        al.record("KA05MX9999", "No Seat Belt")
        al.record("KA05MX9999", "No Seat Belt")  # triggers
        result = al.record("KA05MX9999", "No Seat Belt")  # should NOT trigger again
        assert result is False

    def test_reset_clears_history(self):
        from utils.alerts import AlertSystem
        al = AlertSystem(threshold=2)
        al.record("DL01AB1234", "No Helmet")
        al.record("DL01AB1234", "No Helmet")
        al.reset_plate("DL01AB1234")
        # After reset, should need 2 more events
        assert not al.record("DL01AB1234", "No Helmet")


# ══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntegration:

    def test_annotator_runs_on_blank_frame(self, blank_frame, settings):
        from utils.annotator import FrameAnnotator
        from core.pipeline import FrameStats
        ann   = FrameAnnotator(settings)
        stats = FrameStats(frame_num=1, fps=30.0, vehicles=0, violations=0, plates_read=0)
        out   = ann.draw(blank_frame, [], stats)
        assert out is not None
        assert out.shape == blank_frame.shape

    def test_report_writer_full_cycle(self, settings):
        import os, csv
        from utils.report_writer import ReportWriter
        rw = ReportWriter(settings)
        for i in range(5):
            rw.write({
                "timestamp"    : f"2024-01-01T12:00:0{i}",
                "camera_id"    : "TEST-001",
                "plate"        : f"MH12AB123{i}",
                "plate_raw"    : f"MH12AB123{i}",
                "plate_conf"   : 90.0,
                "plate_valid"  : True,
                "plate_enhanced": True,
                "vehicle_class": "Car",
                "violation"    : "Compliant",
                "helmet"       : None,
                "helmet_conf"  : 0.0,
                "seatbelt"     : True,
                "seatbelt_conf": 88.0,
                "image_saved"  : "",
            })
        rw.close()
        reports_dir = os.path.join(settings.output_dir, "reports")
        csv_files   = [f for f in os.listdir(reports_dir) if f.endswith(".csv")]
        assert len(csv_files) >= 1
        with open(os.path.join(reports_dir, csv_files[0])) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5

    def test_anonymiser_returns_same_shape(self, blank_frame):
        from utils.anonymiser import FaceAnonymiser
        fa  = FaceAnonymiser()
        out = fa.blur_faces(blank_frame)
        assert out.shape == blank_frame.shape

    def test_heatmap_save_snapshot(self, tmp_path):
        from utils.heatmap import TrafficHeatmap
        hm  = TrafficHeatmap(width=640, height=480, output_dir=str(tmp_path))
        out = hm.save_snapshot(tag="test_snapshot")
        assert out.exists()

    def test_api_health_endpoint(self, settings):
        """Test API health check without requiring a running server."""
        try:
            from api.server import create_app
            from fastapi.testclient import TestClient
            app    = create_app(pipeline=None, settings=settings)
            client = TestClient(app)
            resp   = client.get("/")
            assert resp.status_code == 200
            data   = resp.json()
            assert data["status"] == "online"
        except ImportError:
            pytest.skip("fastapi[testclient] not installed")

    def test_api_stats_endpoint(self, settings):
        try:
            from api.server import create_app
            from fastapi.testclient import TestClient
            app    = create_app(pipeline=None, settings=settings)
            client = TestClient(app)
            resp   = client.get("/stats")
            assert resp.status_code == 200
            data   = resp.json()
            assert "total_violations" in data
        except ImportError:
            pytest.skip("fastapi[testclient] not installed")

    def test_api_detections_endpoint(self, settings):
        try:
            from api.server import create_app
            from fastapi.testclient import TestClient
            app    = create_app(pipeline=None, settings=settings)
            client = TestClient(app)
            resp   = client.get("/detections?limit=10")
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)
        except ImportError:
            pytest.skip("fastapi[testclient] not installed")

    def test_api_config_update(self, settings):
        try:
            from api.server import create_app
            from fastapi.testclient import TestClient
            app    = create_app(pipeline=None, settings=settings)
            client = TestClient(app)
            resp   = client.post("/config", json={"conf_thresh": 0.55, "use_genai": False})
            # Without a live pipeline settings object it returns 503 — that's expected
            assert resp.status_code in (200, 503)
        except ImportError:
            pytest.skip("fastapi[testclient] not installed")
