# Person Tracking with YOLOv8 and DeepSORT

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python\&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111111)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv\&logoColor=white)
![DeepSORT](https://img.shields.io/badge/Tracking-DeepSORT-orange)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)


## Overview

This project is a computer vision system for detecting and tracking people in videos.

It uses **YOLOv8** for person detection and **DeepSORT** for multi-object tracking. The system gives a unique ID to each detected person and keeps track of their position across video frames.

The project also includes a **trajectory analysis module** that stores the positions of tracked people and draws their trajectories on the video.

---

## Project Goals

The main goals of this project are to:

* Detect people in a video using YOLOv8.
* Track people across different frames using DeepSORT.
* Keep a persistent ID for each tracked person.
* Calculate the center point of each bounding box.
* Store the positions of each person.
* Draw the movement trajectory of each person.
* Generate an annotated output video.
* Export tracking data for further analysis.

---

## Pipeline
The complete pipeline is:

The complete pipeline is:

```text
Input Video
     │
     ▼
   YOLOv8
     │
     ▼
Person Detection
     │
     ▼
Prepare Detections
     │
     ▼
  DeepSORT
     │
     ▼
Track ID + Bounding Box
     │
     ▼
Calculate Centroid
     │
     ▼
Trajectory Analyzer
     │
     ▼
Store Positions
     │
     ▼
Draw Trajectories
     │
     ▼
Annotated Video + CSV Results
```

### 1. Person Detection

YOLOv8 analyzes each video frame and detects objects.

For this project, only the **person class** is kept.

Each detection contains:

* Bounding box
* Confidence score
* Class ID

The bounding box uses the following format:

```text
[x1, y1, x2, y2]
```

---

### 2. Multi-Object Tracking

The detections are sent to DeepSORT.

DeepSORT is responsible for tracking people across different frames.

For example:

```text
Frame 1 → Person → ID 1
Frame 2 → Person → ID 1
Frame 3 → Person → ID 1
```

The goal is to keep the same ID for the same person while they move through the video.

---

### 3. Centroid Calculation

For each bounding box, the center point is calculated.

The formula is:

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

For example:

```text
Bounding box:
[100, 50, 300, 450]

Centroid:
(200, 250)
```

This point is used to represent the position of the person.

---

### 4. Trajectory Analysis

The `TrajectoryAnalyzer` stores the position of each tracked person.

The `track_id` is used as the key.

For example:

```text
ID 1 → [(100,200), (110,205), (120,210)]
ID 2 → [(400,100), (390,105), (380,110)]
```

The project uses a `deque` to keep a limited number of recent positions.

This helps to draw the recent movement of each person.

---

### 5. Trajectory Visualization

The stored points are connected using OpenCV:

```python
cv2.line()
```

This creates a visual trajectory for each tracked person.

Example:

```text
●────●────●────●
```
## Technologies

| Technology | Role                               |
| ---------- | ---------------------------------- |
| Python     | Main programming language          |
| YOLOv8     | Person detection                   |
| DeepSORT   | Multi-object tracking              |
| OpenCV     | Video processing and visualization |
| NumPy      | Numerical computation              |

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
---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Fallou-Diouf/person-tracking-yolov8-deepsort
cd person_tracking_yolov8_deepsort
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

On Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Usage

Put your input video in:

```text
data/input/video.mp4
```

Then run:

```bash
python -m src.main
```

The program will:

1. Read the input video.
2. Detect people with YOLOv8.
3. Track people with DeepSORT.
4. Calculate their center positions.
5. Store their trajectories.
6. Draw bounding boxes and IDs.
7. Draw movement trajectories.
8. Save the processed video.

The output video is saved to:

```text
data/output/tracked_video.mp4
```
##  Current Results
=======
## Current Results
The system was tested on a video with the following characteristics:

| Metric                  |         Result |
| ----------------------- | -------------: |
| Resolution              |      478 × 850 |
| Original FPS            |         30 FPS |
| Number of frames        |            638 |
| Processed frames        |            638 |
| Average processing FPS  |          12.60 |
| Average processing time | 79.34 ms/frame |

The complete video was successfully processed and the annotated result was generated.

---

## What I Learned

Through this project, I learned how to build a complete computer vision pipeline.

### Object Detection

I learned:

* Bounding boxes
* Confidence scores
* Class IDs
* IoU
* Non-Maximum Suppression
* YOLOv8 detection

### Object Tracking

I learned:

* Kalman Filter concept
* Track IDs
* Data association
* DeepSORT
* Multi-object tracking

### Trajectory Analysis

I learned:

* Centroid calculation
* Position history
* `defaultdict`
* `deque`
* Trajectory visualization with OpenCV

---

## Limitations

This version has some limitations:

* The trajectory distance is measured in pixels.
* Real-world distance in meters is not implemented yet.
* Camera calibration is not implemented.
* Very crowded scenes can make tracking more difficult.
* Occlusions can sometimes affect tracking.
* The current pipeline runs slower than the original video FPS.

---

## Future Improvements

The next improvements planned for this project are:

* [ ] Calculate the distance traveled by each person.
* [ ] Estimate movement speed.
* [ ] Detect entry and exit events.
* [ ] Create zones of interest (ROI).
* [ ] Calculate people occupancy in each zone.
* [ ] Calculate time spent in a zone.
* [ ] Improve tracking in crowded scenes.
* [ ] Add tracking evaluation metrics.
* [ ] Measure ID switches.
* [ ] Add MOTA, IDF1 and HOTA metrics.
* [ ] Improve inference speed.
* [ ] Add camera calibration and homography.
* [ ] Explore person re-identification.
* [ ] Prepare the system for deployment.

---

## Project Architecture

The main idea of the project can be summarized as:

```text
YOLOv8
Detection
   ↓
DeepSORT
Tracking
   ↓
Track ID
Identity
   ↓
Centroid
Position
   ↓
TrajectoryAnalyzer
Movement History
   ↓
OpenCV
Visualization
```

This architecture can later be extended to build more advanced video analytics systems.

---

## Author

**Fallou Diouf**
