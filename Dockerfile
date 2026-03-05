# ══════════════════════════════════════════════════════════════════
#  Smart City ANPR System — Multi-Stage Dockerfile
#  Base: nvidia/cuda for GPU inference (falls back to CPU-only)
#  Build: docker build -t anpr-system:latest .
#  Run:   docker-compose up           (see docker-compose.yml)
#         docker run --gpus all -p 8000:8000 anpr-system:latest
# ══════════════════════════════════════════════════════════════════

# ── Stage 1: Builder — installs Python deps ────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed for OpenCV + compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1-mesa-glx \
        libgstreamer1.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into an isolated prefix for clean copy
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Sathish Kumar Nagalingam <sv2447@srmist.edu.in>"
LABEL description="Smart City ANPR — Multi-Modal Vehicle Detection & LPR with GenAI"
LABEL version="1.0.0"

# Runtime system libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1-mesa-glx \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# App source
WORKDIR /app
COPY . .

# Create output dirs
RUN mkdir -p outputs/violations outputs/reports outputs/heatmaps

# Pre-download YOLO weights (cached in layer)
# Real-ESRGAN weights are auto-downloaded on first run (~64 MB)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || true

# ── Environment ─────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ANPR_DEVICE=auto
ENV ANPR_CAMERA_ID=CCTV-001
ENV ANPR_OUTPUT_DIR=/app/outputs
ENV ANPR_CONF_THRESH=0.40
ENV ANPR_GENAI=true

# ── Ports ────────────────────────────────────────────────────────
# REST API
EXPOSE 8000
# Optionally expose VNC/display for GUI (if running with X11 forwarding)
# EXPOSE 5900

# ── Healthcheck ──────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# ── Default command: headless + API ─────────────────────────────
CMD ["python", "main.py", \
     "--headless", \
     "--source", "0", \
     "--device", "auto", \
     "--output", "/app/outputs"]
