import numpy as np

class SnapDetector:
    def __init__(self):
        self.previous_ball_pos = None

    def detect_snap(self, ball_pos):
        """
        ball_pos: np.array([x,y]) proxy for ball location
        """
        if self.previous_ball_pos is None:
            self.previous_ball_pos = ball_pos
            return False
        movement = np.linalg.norm(ball_pos - self.previous_ball_pos)
        self.previous_ball_pos = ball_pos
        return movement > 5  # px threshold for snap
