import numpy as np

class OLPositionTracker:
    def track_positions(self, ol_kps_seq, frame_interval=15):
        data = []
        for t, frame_kps in enumerate(ol_kps_seq):
            if t % frame_interval != 0: continue
            for pid, k in frame_kps.items():
                if k.shape[0] != 17: continue
                hip = (k[11][1]+k[12][1])/2
                pad = k[5][1] - hip
                base = float(np.linalg.norm(k[15]-k[16]))
                data.append({
                    'frame': t,
                    'player_id': pid,
                    'hip_height_px': hip,
                    'pad_level_px': pad,
                    'base_width_px': base
                })
        return data
