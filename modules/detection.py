from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_path):
        print(f"Loading YOLOv8-Pose model from {model_path}...")
        self.model = YOLO(model_path)
        print("Model loaded successfully.")

    def detect_frame(self, frame):
        results = self.model.predict(source=frame, conf=0.3, verbose=False)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints else []
        return boxes, keypoints
