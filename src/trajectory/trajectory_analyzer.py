from collections import defaultdict, deque
from math import sqrt

import csv 
from pathlib import Path


class TrajectoryAnalyzer:
    """
    Store and analyze the trajectories of tracked objects.
    """

    def __init__(self, max_history=100):

        self.max_history = max_history

        # track_id -> deque of (frame, center_x, center_y)
        self.trajectories = defaultdict(
            lambda: deque(maxlen=max_history)
        )

    @staticmethod
    def compute_centroid(bbox):
        """
        Compute the center point of a bounding box.

        bbox format:
        [x1, y1, x2, y2]
        """

        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return center_x, center_y

    def update(self, tracks, frame_number):
        """
        Update trajectories with the current tracks.
        """

        for track in tracks:

            track_id = track["track_id"]
            bbox = track["bbox"]

            center_x, center_y = self.compute_centroid(
                bbox
            )

            self.trajectories[track_id].append(
                (
                    frame_number,
                    center_x,
                    center_y,
                )
            )

    def get_trajectory(self, track_id):
        """
        Return the trajectory of a specific track.
        """

        return list(
            self.trajectories.get(track_id, [])
        )

    def get_all_trajectories(self):
        """
        Return all stored trajectories.
        """

        return {
            track_id: list(points)
            for track_id, points
            in self.trajectories.items()
        }

    def compute_distance(self, track_id):
        """
        Compute the approximate distance traveled
        by a tracked person in pixel coordinates.
        """

        trajectory = self.get_trajectory(track_id)

        if len(trajectory) < 2:
            return 0.0

        total_distance = 0.0

        for i in range(1, len(trajectory)):

            _, x1, y1 = trajectory[i - 1]
            _, x2, y2 = trajectory[i]

            distance = sqrt(
                (x2 - x1) ** 2
                + (y2 - y1) ** 2
            )

            total_distance += distance

        return total_distance
    
    def export_csv(self, output_path): 
        
        """ Export all trajectory points to a CSV file. """ 
        
        output_path = Path(output_path) 
        
        output_path.parent.mkdir(
            parents=True, 
            exist_ok=True, 
        ) 
        
        with open( 
                  output_path, 
                  "w", 
                  newline="", 
                  encoding="utf-8", 
                ) as csv_file:

                    writer = csv.writer(csv_file)
                     
                    writer.writerow( 
                                        [ 
                                            "frame", 
                                            "track_id", 
                                            "center_x", 
                                            "center_y", 
                                        ] 
                                    ) 
                    for track_id, points in self.trajectories.items():
                         
                        for frame, x, y in points: 
                            
                            writer.writerow( 
                                                [ 
                                                    frame, 
                                                    track_id, 
                                                    x, 
                                                    y, 
                                                ] 
                                            )
