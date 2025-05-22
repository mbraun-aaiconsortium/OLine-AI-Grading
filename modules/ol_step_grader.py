import numpy as np
from modules.utils import calc_distance, calc_angle, px_to_in

class OLStepGrader:
    def __init__(self, cfg, frame_height):
        self.cfg = cfg
        self.frame_height = frame_height

    def grade_steps(self, ol_kps_history, snap_frame, fps):
        results = []
        for pid, seq in ol_kps_history.items():
            if snap_frame is None or snap_frame >= len(seq): continue
            k0 = seq[snap_frame]
            if k0.shape[0] < 17: continue

            init_L, init_R = k0[15], k0[16]
            first_frame = second_frame = first_px = second_px = None
            # first step
            for i in range(snap_frame+1, len(seq)):
                ki = seq[i]
                if ki.shape[0]<17: continue
                dL = calc_distance(ki[15], init_L)
                dR = calc_distance(ki[16], init_R)
                if dL>self.cfg['first_step_distance_threshold_px'] or \
                   dR>self.cfg['first_step_distance_threshold_px']:
                    first_frame, first_px = i, max(dL,dR)
                    foot = 'left' if dL> dR else 'right'
                    break
            # second step
            if first_frame:
                base = seq[first_frame][15] if foot=='left' else seq[first_frame][16]
                for j in range(first_frame+1, len(seq)):
                    kj = seq[j]
                    if kj.shape[0]<17: continue
                    d = calc_distance(kj[15] if foot=='left' else kj[16], base)
                    if d>self.cfg['second_step_distance_threshold_px']:
                        second_frame, second_px = j, d
                        break
            msf = 1000.0/fps
            f_ms = (first_frame-snap_frame)*msf if first_frame else None
            s_ms = (second_frame-snap_frame)*msf if second_frame else None

            # body metrics at first step
            knee = hip_h = elbow = hand_def = None
            if first_frame:
                kf = seq[first_frame]
                try:
                    k1 = calc_angle(kf[11],kf[13],kf[15])
                    k2 = calc_angle(kf[12],kf[14],kf[16])
                    knee = (k1+k2)/2
                except: pass
                try:
                    hip_px = self.frame_height - ((kf[11][1]+kf[12][1])/2)
                    hip_h = px_to_in(hip_px, self.cfg)
                except: pass
                try:
                    e1 = calc_angle(kf[5],kf[7],kf[9])
                    e2 = calc_angle(kf[6],kf[8],kf[10])
                    elbow = (e1+e2)/2
                except: pass
                try:
                    hand_pt = kf[9]
                    min_d = float('inf')
                    for pid2, seq2 in ol_kps_history.items():
                        if pid2==pid or len(seq2)<=first_frame: continue
                        k2 = seq2[first_frame]
                        if k2.shape[0]<17: continue
                        hip2 = ((k2[11][0]+k2[12][0])/2,(k2[11][1]+k2[12][1])/2)
                        dpx = calc_distance(hand_pt, hip2)
                        min_d = min(min_d, dpx)
                    hand_def = px_to_in(min_d, self.cfg)
                except: pass

            results.append({
                'player_id': pid,
                'first_step_distance_in': px_to_in(first_px, self.cfg) if first_px else None,
                'first_step_time_ms': f_ms,
                'second_step_distance_in': px_to_in(second_px, self.cfg) if second_px else None,
                'second_step_time_ms': s_ms,
                'knee_bend_angle_deg': knee,
                'hip_height_in': hip_h,
                'elbow_bend_angle_deg': elbow,
                'hand_to_defender_in': hand_def
            })
        return results
