import numpy as np
from modules.utils import calc_distance, calc_angle, px_to_in

class OLStepGrader:
    def __init__(self, cfg, frame_height):
        self.cfg = cfg
        self.frame_height = frame_height

    def grade_steps(self, ol_kps_history, snap_frame, fps):
        results = []
        for pid, seq in ol_kps_history.items():
            if len(seq) <= snap_frame: continue

            # First step detection
            base_fp = seq[snap_frame][15]  # left foot
            first_frame = None
            for i in range(snap_frame+1, len(seq)):
                if calc_distance(seq[i][15], base_fp) > self.cfg['first_step_distance_threshold_px']:
                    first_frame = i
                    first_px    = calc_distance(seq[i][15], base_fp)
                    break

            # Second step detection
            second_frame = None
            if first_frame:
                base2 = seq[first_frame][15]
                for j in range(first_frame+1, len(seq)):
                    if calc_distance(seq[j][15], base2) > self.cfg['second_step_distance_threshold_px']:
                        second_frame = j
                        second_px    = calc_distance(seq[j][15], base2)
                        break

            ms_per_frame = 1000/fps
            first_ms  = (first_frame - snap_frame)*ms_per_frame if first_frame else None
            second_ms = (second_frame - snap_frame)*ms_per_frame if second_frame else None

            # Body metrics at first step
            knee_a = hip_h = elbow_a = hand_def = disp_in = None
            if first_frame:
                k = seq[first_frame]
                # Knee
                try:
                    k1 = calc_angle(k[11], k[13], k[15])
                    k2 = calc_angle(k[12], k[14], k[16])
                    knee_a = (k1+k2)/2
                except: pass
                # Hip height (px from bottom)
                try:
                    hy = (k[11][1]+k[12][1])/2
                    hip_px = self.frame_height - hy
                    hip_h  = px_to_in(hip_px, self.cfg)
                except: pass
                # Elbow
                try:
                    e1 = calc_angle(k[6], k[8], k[10])
                    e2 = calc_angle(k[5], k[7], k[9])
                    elbow_a = (e1+e2)/2
                except: pass
                # Hand-to-defender (nearest other hip)
                try:
                    hand = k[9]
                    min_px = float('inf')
                    for pid2, seq2 in ol_kps_history.items():
                        if pid2==pid or len(seq2)<=first_frame: continue
                        k2 = seq2[first_frame]
                        hip2 = np.array([(k2[11][0]+k2[12][0])/2,(k2[11][1]+k2[12][1])/2])
                        d_px = calc_distance(hand, hip2)
                        min_px = min(min_px, d_px)
                    hand_def = px_to_in(min_px, self.cfg)
                except: pass
                # Displacement at 1.5s
                try:
                    tgt = snap_frame+int(self.cfg['run_displacement_time_s']*fps)
                    if tgt < len(seq):
                        hip0 = (seq[snap_frame][11][1]+seq[snap_frame][12][1])/2
                        hip1 = (seq[tgt][11][1]+seq[tgt][12][1])/2
                        disp_in = px_to_in(abs(hip1-hip0), self.cfg)
                except: pass

            results.append({
                'player_id': pid,
                'first_step_distance_in': px_to_in(first_px, self.cfg) if first_frame else None,
                'first_step_time_ms':    first_ms,
                'second_step_distance_in': px_to_in(second_px, self.cfg) if second_frame else None,
                'second_step_time_ms':    second_ms,
                'knee_bend_angle_deg':    knee_a,
                'hip_height_in':          hip_h,
                'elbow_bend_angle_deg':   elbow_a,
                'hand_to_defender_in':     hand_def,
                'displacement_in':         disp_in
            })
        return results
