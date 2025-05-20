import os, cv2, numpy as np
from modules.config_loader     import load_config
from modules.detection         import PoseDetector
from modules.snap_detector     import SnapDetector
from modules.play_classifier   import PlayClassifier
from modules.ol_step_grader    import OLStepGrader
from modules.ol_position_tracker import OLPositionTracker
from modules.error_detector    import OLErrorDetector
from modules.annotator         import Annotator
from modules.reporter          import Reporter

def main():
    # Paths & setup
    cfg = load_config('configs/coach_rules.json')
    vid = 'videos/practice_video.mp4'
    mdl = 'models/yolov8n-pose.pt'
    out_vid = 'outputs/annotated_videos/graded_output.mp4'
    rpt_dir = 'outputs/grading_reports'
    os.makedirs(os.path.dirname(out_vid), exist_ok=True)
    os.makedirs(rpt_dir, exist_ok=True)
    
    
    # Modules
    detector   = PoseDetector(mdl)
    snapper    = SnapDetector()
    classifier = PlayClassifier(cfg)
    cap        = cv2.VideoCapture(vid)
    fps        = cap.get(cv2.CAP_PROP_FPS)
    fh         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    step_grader = OLStepGrader(cfg, frame_height=fh)
    tracker     = OLPositionTracker()
    err_det     = OLErrorDetector(cfg)
    annotator   = Annotator()
    reporter    = Reporter()
    
    
    
    # Video writer
    
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))

    # State
    frame_idx      = 0
    snap_frame     = None
    ol_kps_hist    = {}
    boxes_hist     = []
    

    # Loop
    while True:
        ret, frame = cap.read()
        if not ret: break

        boxes, kps = detector.detect_frame(frame)
        
        # ── Draw bounding boxes ───────────────────────────────────
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # optionally: draw keypoints
        for person_kps in kps:
            for (x, y) in person_kps:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        
        boxes_hist.append(boxes)
        for pid, kp in enumerate(kps):
            ol_kps_hist.setdefault(pid, []).append(kp)

        # Snap
        if len(boxes) and snap_frame is None:
            x1,y1,x2,y2 = boxes[0]
            bp = np.array([(x1+x2)/2,(y1+y2)/2])
            if snapper.detect_snap(bp):
                snap_frame = frame_idx

        # Annotate
        frame = annotator.draw_annotations(frame, [])

        writer.write(frame)
        frame_idx += 1
    

    
    cap.release()
    writer.release()

    # Post‐process
    step_metrics   = step_grader.grade_steps(ol_kps_hist, snap_frame, fps)

    # Track positions every 0.5s (half-second intervals)
    # Build an explicit list of {player_id: keypoints} for each frame
    ol_kps_seq = []
    for f in range(frame_idx):
        frame_kps = {}
        for pid, seq in ol_kps_hist.items():
            if f < len(seq):
                frame_kps[pid] = seq[f]
        ol_kps_seq.append(frame_kps)

    # Now sample that sequence every half-second
    pos_data = tracker.track_positions(ol_kps_seq,
                                       frame_interval=int(fps / 2))

                          
    play_info      = classifier.classify_play(ol_kps_hist, snap_frame, fps)
    errors_detected= err_det.detect_errors(step_metrics, None, None, None)

    # Reports
    reporter.export(step_metrics, pos_data, errors_detected, rpt_dir)


    print("✅ Done. Video:", out_vid, "Reports:", rpt_dir)

if __name__ == "__main__":
    main()
