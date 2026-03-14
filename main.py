"""
main.py — Smart City ANPR System Entry Point

Usage examples:
    python main.py                                  # GUI + webcam
    python main.py --source traffic.mp4             # GUI + video file
    python main.py --source rtsp://192.168.1.100/stream  # GUI + RTSP
    python main.py --headless --source 0            # Headless mode
    python main.py --headless --api --source 0      # Headless + REST API
    python main.py --api-only                       # API stub (no pipeline)
    python main.py --no-genai --source 0            # Disable Real-ESRGAN
    python main.py --device cpu --source 0          # Force CPU mode
    python main.py --image path/to/image.jpg        # Single image mode
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import threading
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smart City ANPR System — Multi-Modal Vehicle Detection & LPR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source",    default="0",
                   help="Video source: 0=webcam, path to video file, or rtsp://...")
    p.add_argument("--conf",      type=float, default=0.40,
                   help="YOLO detection confidence threshold")
    p.add_argument("--device",    default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Inference device")
    p.add_argument("--output",    default="outputs",
                   help="Output directory for reports and violation images")
    p.add_argument("--camera-id", default="CCTV-001",
                   help="Camera identifier used in reports")
    p.add_argument("--no-genai",  action="store_true",
                   help="Disable Real-ESRGAN super-resolution enhancement")
    p.add_argument("--no-anon",   action="store_true",
                   help="Disable face anonymisation")
    p.add_argument("--headless",  action="store_true",
                   help="Run without GUI (console output only)")
    p.add_argument("--api",       action="store_true",
                   help="Start FastAPI REST server alongside pipeline")
    p.add_argument("--api-only",  action="store_true",
                   help="Start REST API without a live pipeline (stub mode)")
    p.add_argument("--api-port",  type=int, default=8000,
                   help="REST API port")
    p.add_argument("--heatmap",   action="store_true",
                   help="Enable traffic density heatmap overlay")
    p.add_argument("--image",     default=None,
                   help="Process a single image and exit")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity")
    return p


def _setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_settings(args):
    from config.settings import Settings
    return Settings(
        source          = args.source,
        camera_id       = args.camera_id,
        conf_thresh     = args.conf,
        device          = args.device,
        use_genai       = not args.no_genai,
        anonymise_faces = not args.no_anon,
        output_dir      = args.output,
        enable_heatmap  = args.heatmap,
    )


def _start_api(pipeline, settings, port: int):
    """Start uvicorn in a daemon thread."""
    try:
        import uvicorn
        from api.server import create_app
        app = create_app(pipeline=pipeline, settings=settings)
        config = uvicorn.Config(app, host="0.0.0.0", port=port,
                                log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        print(f"  ✓ REST API running at http://localhost:{port}/docs")
        return server
    except ImportError:
        print("  ✗ uvicorn not installed — REST API disabled. "
              "Install with: pip install 'uvicorn[standard]'")
        return None


def main():
    parser = _build_parser()
    args   = parser.parse_args()
    _setup_logging(args.log_level)

    log = logging.getLogger("main")
    log.info("Smart City ANPR System — SRM IST 2024-25")
    log.info(f"Python {sys.version.split()[0]} | PID {os.getpid()}")

    # ── API-only stub mode ─────────────────────────────────────────────────
    if args.api_only:
        settings = _build_settings(args)
        server   = _start_api(None, settings, args.api_port)
        if server:
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        return

    settings = _build_settings(args)

    # ── Single-image mode ──────────────────────────────────────────────────
    if args.image:
        import cv2
        from core.pipeline import ANPRPipeline
        pipeline = ANPRPipeline(settings)
        annotated, dets = pipeline.process_single_image(args.image)
        out_path = Path(settings.output_dir) / "single_result.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved annotated image → {out_path}")
        for d in dets:
            plate_txt = d.plate.normalised() if d.plate else "N/A"
            print(f"  {d.vehicle_class:12s} | plate={plate_txt:20s} | {d.violation}")
        return

    from core.pipeline import ANPRPipeline
    pipeline = ANPRPipeline(settings)

    # ── Optionally start API ───────────────────────────────────────────────
    if args.api or args.headless:
        _start_api(pipeline, settings, args.api_port)

    # ── Headless mode ──────────────────────────────────────────────────────
    if args.headless:
        pipeline.run_headless()
        return

    # ── GUI mode ───────────────────────────────────────────────────────────
    try:
        from ui.dashboard import ANPRDashboard
        dashboard = ANPRDashboard(pipeline, settings)
        dashboard.run()
    except ImportError as e:
        log.error(f"Tkinter / PIL not available: {e}. Falling back to headless mode.")
        pipeline.run_headless()


if __name__ == "__main__":
    main()
