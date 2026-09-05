# person-tracking-yolov8-deepsort
End-to-end multi-object tracking system for surveillance videos using YOLOv8 and DeepSORT, with trajectory analysis and video annotation.



# Person Tracking with YOLOv8 and DeepSORT

> End-to-end multi-object tracking system for surveillance videos using YOLOv8 and DeepSORT.

## Overview

This project implements a computer vision pipeline for detecting and tracking people in surveillance videos.

The system combines **YOLOv8** for person detection with **DeepSORT** for multi-object tracking. It assigns persistent identities to detected people, reconstructs their trajectories and generates annotated videos for visualization and further analysis.

## Objectives

The main objectives are to:

* Detect people in surveillance videos.
* Track multiple people across video frames.
* Assign persistent IDs to tracked individuals.
* Reconstruct and visualize individual trajectories.
* Generate annotated output videos.
* Export tracking data for further analysis.
* Evaluate the computational performance of the system.

## Pipeline

```text
Input Video
     │
     ▼
  YOLOv8
     │
     │ Person detections
     ▼
  DeepSORT
     │
     │ Track IDs
     ▼
Trajectory Analysis
     │
     ├── Trajectories
     ├── Tracking statistics
     └── Motion analysis
     │
     ▼
Annotated Video + CSV Results
```

## Technologies

| Technology | Role                               |
| ---------- | ---------------------------------- |
| Python     | Main programming language          |
| YOLOv8     | Person detection                   |
| DeepSORT   | Multi-object tracking              |
| OpenCV     | Video processing and visualization |
| NumPy      | Numerical computation              |
| Pandas     | Tracking data analysis             |
| Matplotlib | Trajectory visualization           |

## Project Structure

```text
person-tracking-yolov8-deepsort/
│
├── configs/
├── data/
├── models/
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── trajectory/
│   └── visualization/
├── scripts/
├── notebooks/
├── results/
├── assets/
├── requirements.txt
└── README.md
```

## Current Status

🚧 **Project under development**

### Roadmap

* [ ] Project initialization
* [ ] YOLOv8 person detection
* [ ] DeepSORT integration
* [ ] Multi-object tracking
* [ ] Trajectory reconstruction
* [ ] Trajectory visualization
* [ ] CSV export
* [ ] Performance evaluation
* [ ] Experimental analysis
* [ ] Final documentation

## Author

**Fallou Diouf**

Master 2 — Vision et Machine Intelligente
Université Paris Cité

