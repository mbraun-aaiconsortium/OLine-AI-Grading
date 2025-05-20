# modules/error_detector.py

class OLErrorDetector:
    def __init__(self, cfg):
        """
        :param cfg: dict loaded from coach_rules.json
        """
        self.cfg = cfg

    def detect_errors(self, step_metrics, position_data, errors, extra=None):
        """
        Compare each metric to coach thresholds and list mistakes.
        :returns: list of {'player_id':…, 'mistakes':[…]}
        """
        errs = []
        for m in step_metrics:
            pid = m['player_id']
            mistakes = []
            c = self.cfg

            # 1) Late first step, if config provides a frame threshold
            frame_thresh = c.get('step_timing_threshold_frames')
            if frame_thresh is not None:
                time_ms = m.get('first_step_time_ms')
                if time_ms is not None:
                    # convert frames to ms (assuming 30fps)
                    thresh_ms = frame_thresh * (1000.0 / 30.0)
                    if time_ms > thresh_ms:
                        mistakes.append(f"Late first step (> {int(thresh_ms)} ms)")

            # 2) Knee bend angle
            knee = m.get('knee_bend_angle_deg')
            if knee is not None:
                if not (c['knee_bend_min_deg'] <= knee <= c['knee_bend_max_deg']):
                    mistakes.append("Bad knee bend")

            # 3) Elbow bend angle
            elbow = m.get('elbow_bend_angle_deg')
            if elbow is not None:
                if not (c['elbow_bend_min_deg'] <= elbow <= c['elbow_bend_max_deg']):
                    mistakes.append("Bad elbow bend")

            # 4) Hand placement relative to defender
            hd = m.get('hand_to_defender_in')
            if hd is not None:
                if hd > c['hand_defender_distance_max_in']:
                    mistakes.append("Hands too far from defender")

            errs.append({'player_id': pid, 'mistakes': mistakes})

        return errs
