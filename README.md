# Person Tracking with YOLOv8 & DeepSORT

> **End-to-end multi-object tracking system for surveillance videos using YOLOv8 and DeepSORT, with trajectory reconstruction, visualization and quantitative analysis.**



---

## Overview

This project implements an end-to-end **Computer Vision pipeline for detecting, tracking and analyzing people in surveillance videos**.

The system combines:

* **YOLOv8** for real-time person detection
* **DeepSORT** for multi-object tracking
* **OpenCV** for video processing and visualization
* **Trajectory analysis** for reconstructing individual movements
* **CSV export** for downstream data analysis

The main objective is to build a modular and reproducible **Multi-Object Tracking (MOT)** system that can be extended toward intelligent video surveillance applications.

---

## Objectives

The project aims to answer the following questions:

* How can people be reliably detected in surveillance videos?
* How can each detected person be assigned a persistent identity?
* How can individual trajectories be reconstructed over time?
* How can tracking data be extracted for quantitative analysis?
* How does the system behave under occlusions and crowded scenes?
* What are the computational trade-offs between detection accuracy and real-time performance?

---

# System Architecture

```text
                    INPUT VIDEO
                         │
                         ▼
                  ┌─────────────┐
                  │   OpenCV    │
                  │ Video Reader│
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   YOLOv8    │
                  │   Detector  │
                  └──────┬──────┘
                         │
                  Person Detections
                  [bbox + confidence]
                         │
                         ▼
                  ┌─────────────┐
                  │  DeepSORT   │
                  │   Tracker   │
                  └──────┬──────┘
                         │
                     Track IDs
                         │
                         ▼
                ┌─────────────────┐
                │    Trajectory   │
                │     Analyzer    │
                └────────┬────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         Centroids    Distance    CSV Data
              │
              ▼
       Trajectory History
              │
              ▼
      Annotated Video Output
```

---

# Detection

The first stage uses **YOLOv8** to detect objects belonging to the `person` class.

For every detected person, the detector provides:

```text
Bounding Box
Confidence Score
Class ID
```

Example:

```python
{
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.93,
    "class_id": 0
}
```

The system currently filters detections to the COCO `person` class:

```text
class_id = 0
```

---

# Multi-Object Tracking

YOLOv8 performs detection independently on each frame. It does **not** maintain object identities across frames.

DeepSORT is therefore used to associate detections over time.

Example:

```text
Frame 1

Person A → ID 1
Person B → ID 2


Frame 2

Person A → ID 1
Person B → ID 2


Frame 3

Person A → ID 1
Person B → ID 2
```

This allows the system to maintain persistent identities and reconstruct individual trajectories.

---

# Trajectory Reconstruction

For every tracked person, the system computes the centroid of the bounding box:

```text
center_x = (x1 + x2) / 2

center_y = (y1 + y2) / 2
```

The position history is then represented as:

```text
Track ID 1

Frame 1 → (450, 320)
Frame 2 → (455, 324)
Frame 3 → (462, 331)
Frame 4 → (470, 339)
...
```

These points form the trajectory of the tracked person.

The trajectory can then be visualized directly on the video.

---

# Data Export

Tracking information can be exported as CSV for further analysis.

Example:

```csv
frame,track_id,center_x,center_y
1,1,452.5,318.0
2,1,456.0,321.5
3,1,461.0,326.0
4,1,468.0,334.0
1,2,720.5,301.0
2,2,717.0,305.5
```

This data can be used for:

* trajectory visualization
* movement analysis
* distance estimation
* speed estimation
* zone analysis
* statistical evaluation

---

# Output

The current pipeline generates an annotated video containing:

* Bounding boxes
* Persistent tracking IDs
* Number of active tracks
* Processing FPS
* Current frame number
* Person trajectories

Example:

```text
┌─────────────────────────────────────────┐
│                                         │
│      ┌───────────────┐                  │
│      │ Person ID: 1  │                  │
│      └───────────────┘                  │
│          ╲                              │
│           ╲                             │
│            ●──●──●                     │
│                                         │
│                    ┌───────────────┐    │
│                    │ Person ID: 2  │    │
│                    └───────────────┘    │
│                                         │
│  Active tracks: 2                       │
│  FPS: 18.4                              │
│  Frame: 347                             │
└─────────────────────────────────────────┘
```

> A real demonstration GIF/video will be added to the repository once the pipeline and visualization are finalized.

---

# Project Structure

```text
person-tracking-yolov8-deepsort/
│
├── assets/
│   ├── architecture.png
│   └── demo.gif
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── input/
│   └── output/
│
├── models/
│
├── notebooks/
│   └── exploration.ipynb
│
├── results/
│   ├── metrics/
│   ├── trajectories/
│   └── videos/
│
├── scripts/
│   ├── analyze_trajectories.py
│   └── run_tracking.py
│
├── src/
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_detector.py
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── deepsort_tracker.py
│   │
│   ├── trajectory/
│   │   ├── __init__.py
│   │   └── trajectory_analyzer.py
│   │
│   ├── visualization/
│   │   └── visualizer.py
│   │
│   └── main.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/<Fallou-Diouf>/person-tracking-yolov8-deepsort.git

cd person-tracking-yolov8-deepsort
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
ultralytics
opencv-python
deep-sort-realtime==1.3.2
numpy
scipy
pandas
matplotlib
PyYAML
tqdm
setuptools==81.0.0
```

---

# Usage

Place a surveillance video in:

```text
data/input/surveillance.mp4
```

Then run:

```bash
python -m src.main
```

The processed video will be generated at:

```text
data/output/tracked_video.mp4
```

Trajectory data will be exported to:

```text
results/trajectories/tracks.csv
```

---

# Configuration

The main parameters are centralized in:

```text
configs/config.yaml
```

Example:

```yaml
model:
  name: yolov8n.pt
  confidence_threshold: 0.5
  iou_threshold: 0.5
  classes:
    - 0

tracking:
  max_age: 30
  n_init: 3
  max_cosine_distance: 0.2

trajectory:
  max_history: 100
  save_csv: true
```

This allows experiments to be reproduced without modifying the source code.

---

# Current Performance

Performance depends on:

* video resolution
* hardware
* YOLO model size
* number of people
* scene complexity

The pipeline currently reports:

```text
Average FPS
Average processing time per frame
Number of processed frames
Number of active tracks
```

Detailed benchmark results will be added after systematic experiments.

---

# Experimental Evaluation

The project will progressively evaluate the system under different conditions:

### Detection

* Detection confidence
* Precision
* Recall
* mAP

### Tracking

* IDF1
* MOTA
* HOTA
* ID switches

### Computational Performance

* FPS
* inference time
* memory consumption
* model size

### Scene Conditions

* low-light scenes
* crowded scenes
* occlusions
* different camera viewpoints
* different resolutions

---

# Roadmap

## Phase 1 — Project Initialization

* [x] Repository structure
* [x] Python virtual environment
* [x] Dependencies
* [x] Configuration file
* [x] Initial documentation

## Phase 2 — Person Detection

* [x] YOLOv8 integration
* [x] Person class filtering
* [x] Bounding box visualization
* [x] Confidence scores
* [x] Video processing
* [x] FPS measurement

## Phase 3 — Multi-Object Tracking

* [x] DeepSORT integration
* [x] Persistent track IDs
* [x] Track management
* [x] Annotated tracking video

## Phase 4 — Trajectory Analysis

* [x] Centroid computation
* [x] Trajectory history
* [x] Trajectory visualization
* [x] CSV export
* [x] Approximate distance calculation

## Phase 5 — Advanced Analysis

* [ ] Full trajectory history
* [ ] Speed estimation
* [ ] Entry / exit detection
* [ ] Regions of Interest (ROI)
* [ ] Zone occupancy
* [ ] Dwell time analysis

## Phase 6 — Evaluation

* [ ] Benchmark YOLOv8 model variants
* [ ] FPS comparison
* [ ] Detection metrics
* [ ] Tracking metrics
* [ ] ID switch analysis
* [ ] Robustness experiments

## Phase 7 — Portfolio & Documentation

* [ ] Architecture diagram
* [ ] Demo GIF
* [ ] Experimental results
* [ ] Performance tables
* [ ] Failure case analysis
* [ ] Final technical report

---

# Limitations

The current system operates in image-space coordinates.

Therefore:

```text
Distance = pixels
```

and not:

```text
Distance = meters
```

Accurate real-world measurements would require camera calibration, scene geometry or a perspective transformation such as a homography.

Tracking performance can also degrade in:

* heavy occlusions
* very crowded scenes
* strong illumination changes
* very small objects
* fast camera motion

These limitations will be investigated in the experimental evaluation.

---

# Future Improvements

Potential extensions include:

* Camera calibration
* Homography-based ground-plane mapping
* Real-world distance estimation
* Speed estimation
* Entry/exit counting
* Restricted-zone detection
* Crowd density estimation
* Person re-identification
* Tracking performance benchmarking
* GPU optimization
* Real-time deployment

---

# Technologies

| Technology | Purpose                   |
| ---------- | ------------------------- |
| Python     | Main development language |
| YOLOv8     | Person detection          |
| DeepSORT   | Multi-object tracking     |
| OpenCV     | Video processing          |
| NumPy      | Numerical computation     |
| SciPy      | Scientific computation    |
| Pandas     | Data analysis             |
| Matplotlib | Visualization             |
| PyYAML     | Configuration             |

---

# Author

**Fallou Diouf**

Master 2 — Vision et Machine Intelligente
**Université Paris Cité**

### Areas of interest

* Computer Vision
* Deep Learning
* Object Detection
* Multi-Object Tracking
* Self-Supervised Learning
* Image Retrieval
* Machine Learning
* Images Processing

---

# 📄 License

This project is released under the MIT License.
