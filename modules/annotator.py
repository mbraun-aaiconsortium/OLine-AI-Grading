import cv2

class Annotator:
    def draw_pose_with_ids(self, frame, boxes, keypoints, pid_labels):
        """
        Draws bounding boxes + keypoints + provided player IDs.
        :param pid_labels: list of same length as boxes, giving the pid for each detection.
        """
        for (x1, y1, x2, y2), pid in zip(boxes, pid_labels):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # ID label
            cv2.putText(
                frame,
                f"ID:{pid}",
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
        # Draw keypoints
        for person in keypoints:
            for (x, y) in person:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
        return frame

    def draw_annotations(self, frame, step_metrics, errors):
        # existing code for metrics overlay...
        for idx, m in enumerate(step_metrics):
            pid = m['player_id']
            fs  = m.get('first_step_time_ms') or 0
            cv2.putText(
                frame,
                f"P{pid} FS:{fs:.0f}ms",
                (10, 100 + 20 * idx),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )
            errs = errors[idx]['mistakes']
            if errs:
                cv2.putText(
                    frame,
                    ", ".join(errs),
                    (200, 100 + 20 * idx),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )
        return frame
