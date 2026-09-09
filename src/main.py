import time
from pathlib import Path

import cv2

from src.detection.yolo_detector import YOLOPersonDetector
from src.tracking.deepsort_tracker import DeepSORTTracker
from src.trajectory.trajectory_analyzer import TrajectoryAnalyzer


INPUT_VIDEO = "data/input/input.mp4"
OUTPUT_VIDEO = "data/output/tracked_video.mp4"


def draw_trajectories(frame, trajectory_analyzer):
    """
    Draw historical trajectories for all tracked objects.
    """

    annotated_frame = frame.copy()

    trajectories = (
        trajectory_analyzer.get_all_trajectories()
    )

    for track_id, points in trajectories.items():

        if len(points) < 2:
            continue

        for i in range(1, len(points)):

            _, x1, y1 = points[i - 1]
            _, x2, y2 = points[i]

            cv2.line(
                annotated_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2,
            )

    return annotated_frame

def draw_tracks(frame, tracks):
    """Draw tracked persons and their IDs."""

    annotated_frame = frame.copy()

    for track in tracks:

        track_id = track["track_id"]

        x1, y1, x2, y2 = map(
            int,
            track["bbox"],
        )

        # Bounding box
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # Track ID
        label = f"Person ID: {track_id}"

        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return annotated_frame


def main():

    # --------------------------------------------------
    # 1. Initialize models
    # --------------------------------------------------

    detector = YOLOPersonDetector(
        model_name="yolov8n.pt",
        confidence_threshold=0.5,
        iou_threshold=0.5,
    )

    tracker = DeepSORTTracker(
        max_age=30,
        n_init=3,
        max_cosine_distance=0.2,
    )
    
    trajectory_analyzer = TrajectoryAnalyzer(
    max_history=100
    )

    # --------------------------------------------------
    # 2. Open video
    # --------------------------------------------------

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {INPUT_VIDEO}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print("\n========== VIDEO ==========")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")

    # --------------------------------------------------
    # 3. Output
    # --------------------------------------------------

    Path(OUTPUT_VIDEO).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height),
    )

    # --------------------------------------------------
    # 4. Processing
    # --------------------------------------------------

    frame_count = 0
    total_processing_time = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        start_time = time.perf_counter()

        # YOLO detection
        detections = detector.detect(frame)

        # DeepSORT tracking
        tracks = tracker.update(
            detections,
            frame,
        )
        
        trajectory_analyzer.update(
            tracks,
            frame_count,
        )

        # Visualization
        annotated_frame = draw_tracks(
            frame,
            tracks,
        )

        annotated_frame = draw_trajectories(
            annotated_frame,
            trajectory_analyzer,
        )
        
        # Performance
        processing_time = (
            time.perf_counter() - start_time
        )

        total_processing_time += processing_time

        frame_count += 1

        inference_fps = 1.0 / processing_time

        # --------------------------------------------------
        # Global statistics
        # --------------------------------------------------

        cv2.putText(
            annotated_frame,
            f"Active tracks: {len(tracks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {inference_fps:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_count}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Write frame
        writer.write(annotated_frame)

        # Preview
        cv2.imshow(
            "YOLOv8 + DeepSORT",
            annotated_frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------
    # 5. Cleanup
    # --------------------------------------------------

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    trajectory_analyzer.export_csv(
        "results/trajectories/tracks.csv"
    )

    # --------------------------------------------------
    # 6. Final statistics
    # --------------------------------------------------

    if frame_count > 0:

        average_processing_time = (
            total_processing_time / frame_count
        )

        average_fps = (
            1.0 / average_processing_time
        )

        print("\n========== RESULTS ==========")
        print(
            f"Processed frames: {frame_count}"
        )
        print(
            f"Average FPS: {average_fps:.2f}"
        )
        print(
            f"Average processing time: "
            f"{average_processing_time * 1000:.2f} ms/frame"
        )
        print(
            f"Output: {OUTPUT_VIDEO}"
        )


if __name__ == "__main__":
    main()
