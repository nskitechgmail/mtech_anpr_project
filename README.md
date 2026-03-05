# 🚦 Smart City ANPR System
### Multi-Modal Vehicle Detection & License Plate Recognition using Generative AI

**SRM Institute of Science and Technology — Department of Computational Intelligence**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![YOLOv9](https://img.shields.io/badge/detector-YOLOv9-00C2E0.svg)](https://github.com/WongKinYiu/yolov9)
[![Real-ESRGAN](https://img.shields.io/badge/GenAI-Real--ESRGAN-green.svg)](https://github.com/xinntao/Real-ESRGAN)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

Production-ready, privacy-compliant, AI-powered ANPR for Indian smart city traffic monitoring.

| Tech | Purpose | Result |
|------|---------|--------|
| YOLOv9 | Vehicle detection | 92.4% mAP · 72 FPS |
| Real-ESRGAN ×4 | Plate super-resolution | +33.2% at night |
| EasyOCR + correction | Plate text reading | 84.5% overall accuracy |
| MobileNetV3 | Helmet + seat-belt | F1: 90.4% / 86.4% |
| MediaPipe | Face anonymisation | DPDP Act 2023 compliant |
| FastAPI | REST API | Swagger at /docs |
| Docker | Deployment | GPU + CPU builds |

---

## ⚡ Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/anpr-genai-system.git
cd anpr-genai-system
pip install -r requirements.txt

# GUI dashboard (webcam)
python main.py

# Video file
python main.py --source traffic.mp4

# RTSP CCTV
python main.py --source rtsp://192.168.1.100:554/stream

# Headless + REST API
python main.py --headless --api --source 0

# CPU-only demo (no GPU)
python main.py --no-genai --source 0

# Docker (one command)
docker-compose up
```

## 🌐 REST API

Visit **http://localhost:8000/docs** for Swagger UI.

```bash
GET  /               # Health check
GET  /detections     # Latest vehicle detections
GET  /violations     # Confirmed violations
GET  /stats          # Live system statistics
GET  /heatmap/stats  # Traffic density stats
POST /config         # Update live config
```

## 🧪 Tests

```bash
pytest tests/test_suite.py -v          # all 30 tests
pytest tests/test_suite.py -v -m unit  # unit tests only (no GPU)
```

## 📁 Structure

```
├── main.py              # Entry point
├── config/settings.py   # All configuration
├── core/
│   ├── pipeline.py      # Capture loop + orchestration
│   └── plate_recogniser.py  # 5-stage recognition pipeline
├── models/model_manager.py  # Model registry
├── ui/dashboard.py      # Tkinter GUI
├── api/server.py        # FastAPI REST API
├── utils/
│   ├── annotator.py     # Frame drawing
│   ├── report_writer.py # CSV/JSON reports
│   ├── anonymiser.py    # Face blurring
│   ├── heatmap.py       # Traffic density heatmaps
│   └── alerts.py        # Email/SMS repeat-violator alerts
├── tests/test_suite.py  # 30 unit + integration tests
├── Dockerfile
└── docker-compose.yml
```

## 👥 Team

Sathish Kumar Nagalingam (sv2447@srmist.edu.in) · S. Venkatesh (venkates9@srmist.edu.in)
Department of Computational Intelligence, SRM IST Chennai · 2024–25

## 📄 License

MIT — see [LICENSE](LICENSE)
