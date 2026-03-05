#!/usr/bin/env python3
"""
Smart City ANPR System — Entry Point
SRM Institute of Science and Technology · Dept. of Computational Intelligence

Usage:
    python main.py                              # GUI dashboard (webcam)
    python main.py --source video.mp4           # GUI with video file
    python main.py --source rtsp://ip/stream    # RTSP CCTV camera
    python main.py --headless --source 0        # Headless / server mode
    python main.py --api                        # REST API only (port 8000)
    python main.py --headless --api --source 0  # Headless + REST API combined
    python main.py --no-genai --source 0        # Disable Real-ESRGAN (CPU demo)
"""

import sys, os, argparse, logging, threading
sys.path.insert(0, os.path.dirname(__file__))

from core.pipeline   import ANPRPipeline
from ui.dashboard    import ANPRDashboard
from config.settings import Settings

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("ANPR-Main")

BANNER = """
╔══════════════════════════════════════════════════════════╗
║      Smart City ANPR System  ·  SRM Institute           ║
║  YOLOv9 · Real-ESRGAN · EasyOCR · MobileNetV3           ║
╚══════════════════════════════════════════════════════════╝"""


def parse_args():
    p = argparse.ArgumentParser(description="Smart City ANPR System")
    p.add_argument("--source",    default="0",
                   help="Camera index, video path, image, or RTSP URL")
    p.add_argument("--headless",  action="store_true",
                   help="Run without GUI")
    p.add_argument("--api",       action="store_true",
                   help="Launch FastAPI REST server on --api-port")
    p.add_argument("--api-port",  type=int, default=8000)
    p.add_argument("--genai",     action="store_true", default=True)
    p.add_argument("--no-genai",  dest="genai", action="store_false")
    p.add_argument("--conf",      type=float, default=0.40)
    p.add_argument("--device",    default="auto",
                   choices=["auto","cpu","cuda","mps"])
    p.add_argument("--output",    default="outputs")
    p.add_argument("--camera-id", default="CCTV-001")
    p.add_argument("--no-anon",   action="store_true",
                   help="Disable face anonymisation")
    p.add_argument("--heatmap",   action="store_true",
                   help="Enable traffic density heatmap")
    return p.parse_args()


def start_api_server(pipeline, settings, port):
    try:
        import uvicorn
        from api.server import create_app
        app = create_app(pipeline=pipeline, settings=settings)
        log.info("REST API → http://0.0.0.0:%d  (Swagger: /docs)", port)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    except ImportError:
        log.warning("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")


def main():
    args = parse_args()
    print(BANNER)

    cfg = Settings(
        source          = args.source,
        use_genai       = args.genai,
        conf_thresh     = args.conf,
        device          = args.device,
        output_dir      = args.output,
        camera_id       = args.camera_id,
        anonymise_faces = not args.no_anon,
        enable_heatmap  = getattr(args, "heatmap", False),
    )

    log.info("Source: %s | GenAI: %s | Device: %s | Camera: %s",
             cfg.source,
             "ON" if cfg.use_genai else "OFF",
             cfg.device,
             cfg.camera_id)

    if args.api and args.source == "0" and not args.headless:
        start_api_server(pipeline=None, settings=cfg, port=args.api_port)
        return

    pipeline = ANPRPipeline(cfg)

    if args.api:
        threading.Thread(
            target=start_api_server,
            args=(pipeline, cfg, args.api_port),
            daemon=True,
        ).start()

    if args.headless:
        pipeline.run_headless()
    else:
        ANPRDashboard(cfg).run()


if __name__ == "__main__":
    main()
