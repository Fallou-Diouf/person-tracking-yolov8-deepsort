from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSORTTracker:
    """DeepSORT-based multi-object tracker."""

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.2,
    ):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nms_max_overlap=1.0,
        )

    def update(self, detections, frame):
        """
        Update the tracker using YOLO detections.

        Parameters
        ----------
        detections : list
            YOLO detections.

        frame : numpy.ndarray
            Current video frame.

        Returns
        -------
        list
            Active tracks.
        """

        raw_detections = []

        for detection in detections:

            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]

            width = x2 - x1
            height = y2 - y1

            raw_detections.append(
                (
                    [x1, y1, width, height],
                    confidence,
                    "person",
                )
            )

        tracks = self.tracker.update_tracks(
            raw_detections,
            frame=frame,
        )

        active_tracks = []

        for track in tracks:

            if not track.is_confirmed():
                continue

            if track.time_since_update > 1:
                continue

            track_id = track.track_id

            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = ltrb

            active_tracks.append(
                {
                    "track_id": int(track_id),
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                }
            )

        return active_tracks
