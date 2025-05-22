import numpy as np
from ultralytics import YOLO

class SnapDetector:
    def __init__(self):
        # general COCO model for ball detection
        self.model = YOLO('models/yolov8n.pt')
        self.previous_y = None

    def detect_ball(self, frame):
        res = self.model.predict(source=frame, conf=0.5, classes=[32], verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None
        areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
        idx = int(np.argmax(areas))
        x1,y1,x2,y2 = boxes[idx]
        return np.array([(x1+x2)/2, (y1+y2)/2])

    def detect_snap(self, frame):
        ball = self.detect_ball(frame)
        if ball is None:
            return False
        y = ball[1]
        if self.previous_y is None:
            self.previous_y = y
            return False
        moved = (y - self.previous_y) > 5
        self.previous_y = y
        return moved
