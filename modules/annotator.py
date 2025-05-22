# modules/annotator.py

import cv2
import numpy as np

class Annotator:
    def draw_pose(self, frame, boxes, keypoints):
        """
        Draws bounding boxes (green) and keypoints (yellow) on the frame.
        """
        # Draw boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw keypoints
        for person in keypoints:
            for (x, y) in person:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        return frame

    def draw_pose_with_ids(self, frame, boxes, keypoints, pid_labels):
        """
        Draws each box & keypoints and overlays the given player ID.
        pid_labels must be the same length as boxes (one ID per detection).
        """
        for (box, pid) in zip(boxes, pid_labels):
            x1, y1, x2, y2 = [int(v) for v in box]
            # green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            # ID label in green
            cv2.putText(frame, f"ID:{pid}",
                        (x1, max(y1-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 1, cv2.LINE_AA)
        # then draw keypoints in yellow
        for person in keypoints:
            for (x,y) in person:
                cv2.circle(frame, (int(x),int(y)), 3, (0,255,255), -1)
        return frame
        
    def draw_annotations(self, frame, step_metrics, errors):
        """
        Overlay each player’s first-step time and mistake list.
        """
        for idx, m in enumerate(step_metrics):
            pid = m['player_id']
            fs  = m.get('first_step_time_ms', 0)
            # First-step time
            cv2.putText(frame,
                        f"P{pid} FS:{fs:.0f}ms",
                        (10, 100 + idx*20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1)
            # Mistakes
            errs = errors[idx]['mistakes']
            if errs:
                cv2.putText(frame,
                            ", ".join(errs),
                            (200, 100 + idx*20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1)
        return frame

    def draw_ball_box(self, frame, ball_box):
        """
        Draw a blue rectangle around the ball.
        """
        x1, y1, x2, y2 = [int(v) for v in ball_box]
        # BGR blue is (255,0,0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return frame