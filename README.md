# 🚦 Smart City ANPR System
### Multi-Modal Vehicle Detection & License Plate Recognition using Generative AI

**SRM Institute of Science and Technology — Department of Computational Intelligence**
**Authors:** Sathish Kumar Nagalingam · S. Venkatesh | Academic Year 2024–25

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![YOLOv9](https://img.shields.io/badge/detector-YOLOv9-00C2E0.svg)](https://github.com/WongKinYiu/yolov9)
[![Real-ESRGAN](https://img.shields.io/badge/GenAI-Real--ESRGAN-green.svg)](https://github.com/xinntao/Real-ESRGAN)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

Production-ready, privacy-compliant, AI-powered ANPR for Indian smart city traffic monitoring.

| Technology | Purpose | Result |
|---|---|---|
| YOLOv9 | Vehicle detection | 92.4% mAP · 72 FPS |
| Real-ESRGAN ×4 | Plate super-resolution | +33.2% accuracy at night |
| EasyOCR + correction | Plate text reading | 84.5% overall accuracy |
| MobileNetV3 | Helmet + seat-belt compliance | F1: 90.4% / 86.4% |
| MediaPipe | Face anonymisation | DPDP Act 2023 compliant |
| FastAPI | REST API | Swagger at /docs |
| Docker | Deployment | GPU + CPU builds |

---

## ⚡ Quick Start

```bash
git clone https://github.com/nskitechgmail/mtech_anpr_project.git
cd mtech_anpr_project
pip install -r requirements.txt

# GUI dashboard (webcam)
python main.py

# Video file
python main.py --source traffic.mp4

# RTSP CCTV stream
python main.py --source rtsp://192.168.1.100:554/stream

# Headless + REST API
python main.py --headless --api --source 0

# CPU-only demo (no GPU required)
python main.py --no-genai --device cpu --source 0

# Single image
python main.py --image path/to/image.jpg

# Docker (one command)
docker-compose up
```

---

## 🌐 REST API

Visit **http://localhost:8000/docs** for interactive Swagger UI.

```
GET  /               Health check
GET  /detections     Latest vehicle detections (live frame)
GET  /violations     All confirmed violations (paginated)
GET  /stats          System statistics (FPS, counts, alerts)
GET  /heatmap/stats  Traffic density heatmap stats
POST /config         Update live configuration
```

---

## 🧪 Tests

```bash
pytest tests/test_suite.py -v                    # all 30 tests
pytest tests/test_suite.py -v -m unit            # unit tests only (no GPU)
pytest tests/test_suite.py -v -m integration     # integration tests
pytest tests/test_suite.py --cov=. --cov-report=term-missing
```

---

## 📁 Project Structure

```
├── main.py                      # Entry point (CLI)
├── config/
│   └── settings.py              # All runtime configuration
├── core/
│   ├── pipeline.py              # Capture loop + orchestration
│   └── plate_recogniser.py      # 5-stage recognition pipeline
├── models/
│   └── model_manager.py         # Lazy-loaded model registry + fallbacks
├── ui/
│   └── dashboard.py             # Tkinter GUI dashboard (1400×860)
├── api/
│   └── server.py                # FastAPI REST API
├── utils/
│   ├── annotator.py             # Frame drawing / bounding boxes
│   ├── report_writer.py         # CSV/JSON violation reports
│   ├── anonymiser.py            # MediaPipe face blurring
│   ├── heatmap.py               # Traffic density heatmap
│   └── alerts.py                # Email/SMS repeat-violator alerts
├── tests/
│   └── test_suite.py            # 30 unit + integration tests
├── weights/                     # Auto-downloaded model weights
├── outputs/                     # Reports, violation images, heatmaps
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🧩 9-Stage Pipeline

```
INPUT → YOLOv9 → PLATE LOC → REAL-ESRGAN → EasyOCR
      → SAFETY CLF → TEMP SMOOTHER → FACE BLUR → REPORTER → REST API
```

1. **Input** — Webcam / video file / RTSP CCTV stream
2. **YOLOv9** — Vehicle detection @ 72 FPS (92.4% mAP@0.5)
3. **Plate Localisation** — Contour + aspect-ratio filter
4. **Real-ESRGAN ×4** — Blind super-resolution enhancement (+19.1% OCR)
5. **EasyOCR** — CRAFT text detection + CRNN recognition + position correction
6. **Safety Classifier** — MobileNetV3 helmet + seatbelt detection
7. **Temporal Smoother** — N-frame confirmation (reduces false positives)
8. **Face Anonymiser** — MediaPipe BlazeFace privacy blur
9. **Reporter / API** — CSV/JSON reports + FastAPI /detections endpoint

---

## 🌍 Environment Variables (Alerts)

Copy `.env.example` to `.env` and fill credentials:

```env
ANPR_ALERT_SMTP_HOST=smtp.gmail.com
ANPR_ALERT_SMTP_PORT=587
ANPR_ALERT_SMTP_USER=your@email.com
ANPR_ALERT_SMTP_PASS=app_password
ANPR_ALERT_EMAIL_FROM=your@email.com
ANPR_ALERT_EMAIL_TO=traffic@authority.gov.in
ANPR_TWILIO_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANPR_TWILIO_TOKEN=your_auth_token
ANPR_ALERT_SMS_TO=+919876543210
ANPR_ALERT_SMS_FROM=+1415XXXXXXX
```

---

## 📊 Performance Results

| Condition | Traditional | GenAI | Gain |
|---|---|---|---|
| Good Lighting | 92.5% | 94.8% | +2.3% |
| Low Light | 68.3% | 87.6% | +19.3% |
| Night + Glare | 45.2% | 78.4% | **+33.2%** |
| Motion Blur | 58.7% | 82.3% | +23.6% |
| Rain / Fog | 52.1% | 79.7% | +27.6% |
| **Overall** | **65.4%** | **84.5%** | **+19.1%** |

---

## 👥 Team

| Name | Email | Role |
|---|---|---|
| Sathish Kumar Nagalingam | sv2447@srmist.edu.in | Lead Developer |
| S. Venkatesh | venkates9@srmist.edu.in | Co-Researcher |

Department of Computational Intelligence · SRM IST Chennai · 2024–25

---

## 📄 License

MIT — see [LICENSE](LICENSE)
