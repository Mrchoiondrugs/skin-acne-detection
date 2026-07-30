# 🔬 Skin Acne Detector

A YOLO11-based object detection app that identifies and classifies acne lesions in skin photos — blackheads, whiteheads, papules, pustules, and cysts — with a Streamlit interface for easy use.

## 🚀 Live Demo

👉 [https://skin-acne-detection-eopt554w7bvhpqswkbtgsg.streamlit.app/](https://skin-acne-detection-eopt554w7bvhpqswkbtgsg.streamlit.app/)

## Features

- **Single image detection** — upload a photo or capture one live via webcam
- **Batch analysis** — run detection across multiple images at once and export a summary CSV
- **Session history** — review past scans without re-running them
- **Severity meter** — a simple weighted heuristic (Clear / Mild / Moderate / Severe) based on lesion type and count
- **Lesion close-ups** — auto-cropped thumbnails of the highest-confidence detections
- **Exports** — download the annotated image (PNG) and detection list (CSV)
- Adjustable confidence / IoU thresholds and inference resolution from the sidebar

## Model

- **Architecture:** YOLO11n (Ultralytics)
- **Classes (5):** Blackheads, Cyst, Papules, Pustules, Whiteheads
- **Input resolution:** 1024×1024 (configurable at inference time)
- **Weights file:** `skinyolo11.pt`

> **Note:** Validation metrics were high during training, but train/val overlap in the source dataset wasn't fully ruled out — treat reported accuracy as optimistic until verified on a clean, held-out test set.

## Tech Stack

- [Ultralytics YOLO11](https://docs.ultralytics.com/) for detection
- [Streamlit](https://streamlit.io/) for the interface
- Python 3, Pillow

## Setup

```bash
git clone <your-repo-url>
cd skinacne
pip install -r requirements.txt
```

Make sure `skinyolo11.pt` is in the project root, alongside `app.py`.

## Usage

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`), upload or capture a photo, adjust thresholds if needed, and click **Predict**.

## Project Structure

```
skinacne/
├── app.py              # Streamlit app (frontend + inference logic)
├── requirements.txt     # Python dependencies
├── skinyolo11.pt         # trained YOLO11 weights
└── README.md
```

## Disclaimer

This tool is intended for educational and experimental purposes only. It is **not a medical device** and should not be used to diagnose or guide treatment of any skin condition. Consult a dermatologist for medical advice.

## Acknowledgments

- Training data sourced from [Roboflow Universe](https://universe.roboflow.com/)
- Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
