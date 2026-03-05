"""
api/server.py — FastAPI REST API for Smart City ANPR System.

Endpoints:
    GET  /                   Health check
    GET  /detections         Latest N vehicle detections
    GET  /violations         All confirmed violations (paginated)
    GET  /stats              Live system statistics
    GET  /heatmap/stats      Traffic density heatmap statistics
    POST /config             Update live configuration (conf, genai, fps)
    GET  /docs               Auto-generated Swagger UI (FastAPI built-in)

Run standalone:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Or via main.py:
    python main.py --api-only
"""
from __future__ import annotations
import time, logging
from datetime  import datetime
from pathlib   import Path
from typing    import Optional

log = logging.getLogger("API")

# ── FastAPI import with graceful degradation ──────────────────────────────
try:
    from fastapi            import FastAPI, HTTPException, Query
    from fastapi.responses  import JSONResponse
    from pydantic           import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    log.warning("fastapi not installed — REST API disabled. Run: pip install fastapi uvicorn")

if _FASTAPI_AVAILABLE:

    # ── Pydantic models ────────────────────────────────────────────────────

    class ConfigUpdate(BaseModel):
        conf_thresh: Optional[float]  = Field(None, ge=0.1, le=1.0, description="Detection confidence threshold")
        use_genai:   Optional[bool]   = Field(None, description="Enable/disable Real-ESRGAN enhancement")
        fps_target:  Optional[int]    = Field(None, ge=1,   le=120, description="Target processing FPS")
        anonymise:   Optional[bool]   = Field(None, description="Enable/disable face anonymisation")

    class DetectionResponse(BaseModel):
        id:           int
        timestamp:    str
        camera_id:    str
        vehicle_class: str
        confidence:   float
        plate_text:   Optional[str]
        plate_conf:   Optional[float]
        plate_enhanced: bool
        plate_valid:  bool
        helmet:       Optional[bool]
        seatbelt:     Optional[bool]
        violation:    str

    class ViolationResponse(BaseModel):
        id:           int
        timestamp:    str
        camera_id:    str
        plate:        str
        vehicle_class: str
        violation:    str
        plate_conf:   float
        plate_enhanced: bool
        image_saved:  str

    class StatsResponse(BaseModel):
        uptime_seconds:    float
        frame_count:       int
        fps:               float
        total_vehicles:    int
        total_plates:      int
        total_violations:  int
        genai_enabled:     bool
        camera_id:         str
        source:            str

    # ── App factory ────────────────────────────────────────────────────────

    def create_app(pipeline=None, settings=None) -> FastAPI:
        """
        Create and return the FastAPI application.

        *pipeline* — ANPRPipeline instance (optional; API works in demo mode without it)
        *settings* — Settings instance
        """
        app = FastAPI(
            title       = "Smart City ANPR System API",
            description = (
                "REST API for the Multi-Modal Vehicle Detection and "
                "License Plate Recognition system.\n\n"
                "Built with YOLOv9, Real-ESRGAN, EasyOCR, MobileNetV3.\n"
                "SRM Institute of Science and Technology, 2024–25."
            ),
            version     = "1.0.0",
            docs_url    = "/docs",
            redoc_url   = "/redoc",
        )

        # In-memory stores
        _detections : list[dict] = []
        _violations : list[dict] = []
        _det_counter            = {"n": 0}
        _viol_counter           = {"n": 0}
        _start_time             = time.time()

        # ── State sync helpers ─────────────────────────────────────────────

        def _sync_from_pipeline():
            """Pull latest state from the running pipeline."""
            if pipeline is None:
                return
            frame, stats, dets = pipeline.get_latest()
            for det in dets:
                _det_counter["n"] += 1
                _detections.append({
                    "id"            : _det_counter["n"],
                    "timestamp"     : datetime.now().isoformat(),
                    "camera_id"     : settings.camera_id if settings else "CCTV-001",
                    "vehicle_class" : det.vehicle_class,
                    "confidence"    : round(det.confidence, 3),
                    "plate_text"    : det.plate.text         if det.plate else None,
                    "plate_conf"    : round(det.plate.confidence, 3) if det.plate else None,
                    "plate_enhanced": det.plate.enhanced     if det.plate else False,
                    "plate_valid"   : det.plate.valid_format if det.plate else False,
                    "helmet"        : det.helmet,
                    "seatbelt"      : det.seatbelt,
                    "violation"     : det.violation,
                })
                if det.has_violation() and det.plate:
                    _viol_counter["n"] += 1
                    _violations.append({
                        "id"            : _viol_counter["n"],
                        "timestamp"     : datetime.now().isoformat(),
                        "camera_id"     : settings.camera_id if settings else "CCTV-001",
                        "plate"         : det.plate.normalised(),
                        "vehicle_class" : det.vehicle_class,
                        "violation"     : det.violation,
                        "plate_conf"    : round(det.plate.confidence * 100, 1),
                        "plate_enhanced": det.plate.enhanced,
                        "image_saved"   : "",
                    })
            # Keep lists bounded
            if len(_detections) > 1000:
                _detections.clear()

        # ── Routes ─────────────────────────────────────────────────────────

        @app.get("/", tags=["Health"])
        async def root():
            """Health check endpoint."""
            return {
                "status"  : "online",
                "service" : "Smart City ANPR System",
                "version" : "1.0.0",
                "uptime"  : round(time.time() - _start_time, 1),
            }

        @app.get("/detections", response_model=list[DetectionResponse], tags=["Data"])
        async def get_detections(
            limit : int = Query(50,  ge=1, le=500, description="Max results"),
            offset: int = Query(0,   ge=0,          description="Pagination offset"),
            vehicle_class: Optional[str] = Query(None, description="Filter by vehicle class"),
        ):
            """Return the most recent vehicle detections."""
            _sync_from_pipeline()
            data = _detections[-500:]  # last 500
            if vehicle_class:
                data = [d for d in data if d["vehicle_class"].lower() == vehicle_class.lower()]
            return data[offset: offset + limit]

        @app.get("/violations", response_model=list[ViolationResponse], tags=["Data"])
        async def get_violations(
            limit : int = Query(100, ge=1, le=1000),
            offset: int = Query(0,   ge=0),
            violation_type: Optional[str] = Query(None, description="Filter: 'No Helmet' or 'No Seat Belt'"),
        ):
            """Return all confirmed violations."""
            _sync_from_pipeline()
            data = list(_violations)
            if violation_type:
                data = [v for v in data if v["violation"].lower() == violation_type.lower()]
            return data[offset: offset + limit]

        @app.get("/stats", response_model=StatsResponse, tags=["Data"])
        async def get_stats():
            """Return live system statistics."""
            _sync_from_pipeline()
            frame, stats, _ = (None, None, None) if pipeline is None else pipeline.get_latest()
            return {
                "uptime_seconds"   : round(time.time() - _start_time, 1),
                "frame_count"      : stats.frame_num    if stats else len(_detections),
                "fps"              : round(stats.fps, 1) if stats else 0.0,
                "total_vehicles"   : len(_detections),
                "total_plates"     : sum(1 for d in _detections if d.get("plate_text")),
                "total_violations" : len(_violations),
                "genai_enabled"    : settings.use_genai  if settings else True,
                "camera_id"        : settings.camera_id  if settings else "CCTV-001",
                "source"           : str(settings.source) if settings else "unknown",
            }

        @app.get("/heatmap/stats", tags=["Heatmap"])
        async def get_heatmap_stats():
            """Return traffic density heatmap statistics."""
            if pipeline and hasattr(pipeline, "heatmap"):
                return pipeline.heatmap.get_stats()
            return {"message": "Heatmap not active. Start pipeline first."}

        @app.post("/config", tags=["Control"])
        async def update_config(update: ConfigUpdate):
            """
            Update live system configuration.
            Changes take effect on the next processed frame.
            """
            if settings is None:
                raise HTTPException(status_code=503, detail="Pipeline not running.")
            changed = {}
            if update.conf_thresh is not None:
                settings.conf_thresh = update.conf_thresh
                changed["conf_thresh"] = update.conf_thresh
            if update.use_genai is not None:
                settings.use_genai = update.use_genai
                changed["use_genai"] = update.use_genai
            if update.fps_target is not None:
                settings.fps_target = update.fps_target
                changed["fps_target"] = update.fps_target
            if update.anonymise is not None:
                settings.anonymise_faces = update.anonymise
                changed["anonymise_faces"] = update.anonymise
            log.info(f"Config updated: {changed}")
            return {"status": "ok", "updated": changed}

        return app

    # Standalone app instance (used by uvicorn)
    app = create_app()

else:
    # Stub so import doesn't crash when fastapi is absent
    app = None
