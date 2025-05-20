# modules/play_classifier.py

from modules.utils import px_to_in

class PlayClassifier:
    def __init__(self, cfg):
        """
        :param cfg: dictionary loaded from coach_rules.json
        """
        self.cfg = cfg

    def classify_play(self, ol_kps_history, snap_frame, fps):
        """
        Classify Run vs Pass by average hip displacement over run_displacement_time_s.
        Guards against missing or short keypoint sequences.
        """
        window = int(self.cfg['run_displacement_time_s'] * fps)
        disps = []

        # If we never detected a snap, default to pass
        if snap_frame is None:
            return {'play_type': 'pass', 'run_type': None}

        # For each lineman's keypoint history
        for pid, seq in ol_kps_history.items():
            # Must have enough frames after snap to measure
            idx0 = snap_frame
            idx1 = snap_frame + window
            if idx1 >= len(seq):
                continue  # skip sequences that are too short

            k0 = seq[idx0]
            k1 = seq[idx1]

            # Must have full 17 keypoints at both times
            if k0.shape[0] < 13 or k1.shape[0] < 13:
                continue

            # Compute hip positions (y-coordinate) at snap and at window
            hip0 = (k0[11][1] + k0[12][1]) / 2
            hip1 = (k1[11][1] + k1[12][1]) / 2
            disps.append(abs(hip1 - hip0))

        # Compute average pixel displacement, convert to inches
        avg_px = (sum(disps) / len(disps)) if disps else 0.0
        avg_in = px_to_in(avg_px, self.cfg)

        # Decide run vs pass
        if avg_in >= self.cfg.get('run_displacement_min_in', 0):
            return {'play_type': 'run', 'run_type': 'Inside Zone'}
        else:
            return {'play_type': 'pass', 'run_type': None}
