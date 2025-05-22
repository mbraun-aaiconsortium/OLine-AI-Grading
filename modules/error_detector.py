class OLErrorDetector:
    def __init__(self, cfg):
        self.cfg = cfg

    def detect_errors(self, step_metrics, *_):
        errs = []
        c = self.cfg
        for m in step_metrics:
            pid = m['player_id']
            mistakes = []
            ft = m.get('first_step_time_ms')
            tf = c.get('step_timing_threshold_frames')
            if ft is not None and tf is not None:
                if ft > tf*(1000.0/30.0):
                    mistakes.append("Late first step")
            ka = m.get('knee_bend_angle_deg')
            if ka is not None and not (c['knee_bend_min_deg']<=ka<=c['knee_bend_max_deg']):
                mistakes.append("Bad knee bend")
            ea = m.get('elbow_bend_angle_deg')
            if ea is not None and not (c['elbow_bend_min_deg']<=ea<=c['elbow_bend_max_deg']):
                mistakes.append("Bad elbow bend")
            hd = m.get('hand_to_defender_in')
            if hd is not None and hd>c['hand_defender_distance_max_in']:
                mistakes.append("Hands too far from defender")
            errs.append({'player_id':pid,'mistakes':mistakes})
        return errs
