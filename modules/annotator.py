import cv2

class Annotator:
    def draw_annotations(self, frame, step_metrics):
        for m in step_metrics:
            pid = m['player_id']
            txt = f"P{pid} FS:{m['first_step_time_ms']:.0f}ms"
            cv2.putText(frame, txt, (10, 100+20*pid),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        return frame
