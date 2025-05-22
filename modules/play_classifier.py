import numpy as np
from modules.utils import px_to_in

class PlayClassifier:
    def __init__(self, cfg):
        self.cfg = cfg

    def classify_play(self, ol_kps_history, snap_frame, fps):
        window = int(self.cfg['run_displacement_time_s'] * fps)
        if snap_frame is None:
            return {'play_type':'pass','run_type':None}

        forwards, laterals = [], []
        for seq in ol_kps_history.values():
            if len(seq) <= snap_frame + window:
                continue
            k0, k1 = seq[snap_frame], seq[snap_frame+window]
            if k0.shape[0] < 13 or k1.shape[0] < 13:
                continue
            hip0 = (k0[11][1]+k0[12][1]) / 2
            hip1 = (k1[11][1]+k1[12][1]) / 2
            hipx0 = (k0[11][0]+k0[12][0]) / 2
            hipx1 = (k1[11][0]+k1[12][0]) / 2
            forwards.append(hip1-hip0)
            laterals.append(hipx1-hipx0)

        avg_f_in = px_to_in(abs(np.mean(forwards)) if forwards else 0, self.cfg)
        avg_l_in = px_to_in(abs(np.mean(laterals)) if laterals else 0, self.cfg)

        if avg_f_in >= self.cfg['run_displacement_min_in']:
            if abs(avg_l_in) < 3:
                rt = 'Power'
            elif avg_l_in > 0:
                rt = 'Outside Zone'
            else:
                rt = 'Inside Zone'
            return {'play_type':'run','run_type':rt}
        else:
            return {'play_type':'pass','run_type':None}
