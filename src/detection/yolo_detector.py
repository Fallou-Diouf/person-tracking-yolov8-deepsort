from pathlib import Path

import cv2
from ultralytics import YOLO


class YOLOPersonDetector:
    """YOLOv8-based detector specialized for person detection."""

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def detect(self, frame):
        """
        Detect people in a single video frame.

        Returns:
            list[dict]: Detected persons with bounding boxes
                        and confidence scores.
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )

        detections = []

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": self.PERSON_CLASS_ID,
                }
            )

        return detections

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and confidence scores."""

        annotated_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = map(
                int,
                detection["bbox"]
            )

            confidence = detection["confidence"]

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = f"Person {confidence:.2f}"

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return annotated_frame