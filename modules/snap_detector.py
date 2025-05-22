# modules/snap_detector.py

import numpy as np
from ultralytics import YOLO

class SnapDetector:
    def __init__(self,
                 ball_confidence: float = 0.5,
                 ball_drop_px: float = 5.0,
                 group_motion_px: float = 3.0):
        # YOLOv8n object model for COCO (sports ball = class 32)
        self.ball_model   = YOLO('models/yolov8n.pt')
        self.prev_ball_y  = None
        self.prev_hip_centers = None

        # thresholds
        self.ball_confidence   = ball_confidence
        self.ball_drop_px      = ball_drop_px
        self.group_motion_px   = group_motion_px

        self._snap_reported = False

    def _detect_ball_center(self, frame):
        """Return (x,y) of largest sports-ball detection, or None."""
        res   = self.ball_model.predict(source=frame,
                                        conf=self.ball_confidence,
                                        classes=[32],
                                        verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None
        # pick largest box by area
        areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
        idx   = int(np.argmax(areas))
        x1,y1,x2,y2 = boxes[idx]
        return np.array([(x1+x2)/2, (y1+y2)/2])

    def detect_snap(self, frame, kps_list):
        """
        Call each frame with:
          frame    = BGR image
          kps_list = list of N×2 keypoint arrays (NumPy) for each detected player
        Returns True exactly once (the snap frame), thereafter always False.
        """
        if self._snap_reported:
            return False

        ### 1) Ball‐drop signal ###
        ball = self._detect_ball_center(frame)
        drop_signal = False
        if ball is not None:
            y = ball[1]
            if self.prev_ball_y is not None:
                if (y - self.prev_ball_y) > self.ball_drop_px:
                    drop_signal = True
            self.prev_ball_y = y

        ### 2) Group‐movement signal ###
        # Compute hip‐centers for each player with valid keypoints[11,12]
        hip_centers = []
        for kps in kps_list:
            if kps.shape[0] >= 13:
                hc = ((kps[11][0]+kps[12][0])/2,
                      (kps[11][1]+kps[12][1])/2)
                hip_centers.append(hc)
        move_signal = False
        if self.prev_hip_centers is not None and hip_centers:
            # match by index; take min length
            m = min(len(self.prev_hip_centers), len(hip_centers))
            dists = [np.linalg.norm(np.array(hip_centers[i]) - 
                                    np.array(self.prev_hip_centers[i]))
                     for i in range(m)]
            if np.mean(dists) > self.group_motion_px:
                move_signal = True
        if hip_centers:
            self.prev_hip_centers = hip_centers

        # If either signal is True, that’s our snap
        if drop_signal or move_signal:
            self._snap_reported = True
            return True

        return False

    def detect_ball_box(self, frame):
        """
        Return the xyxy of the largest sports-ball box in the frame, or None.
        """
        # Run the same object‐detect call used in _detect_ball_center
        res   = self.ball_model.predict(
                    source=frame,
                    conf=self.ball_confidence,
                    classes=[32],
                    verbose=False
                )[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None
        areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
        idx   = int(np.argmax(areas))
        return boxes[idx]   # one (4,) array [x1, y1, x2, y2]