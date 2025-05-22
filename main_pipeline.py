# main_pipeline.py

import os
import cv2
import numpy as np

from modules.config_loader       import load_config
from modules.detection           import PoseDetector
from modules.snap_detector       import SnapDetector
from modules.play_classifier     import PlayClassifier
from modules.ol_step_grader      import OLStepGrader
from modules.ol_position_tracker import OLPositionTracker
from modules.error_detector      import OLErrorDetector
from modules.annotator           import Annotator
from modules.reporter            import Reporter
from sklearn.neighbors           import NearestNeighbors

def main():
    print("✅ main_pipeline.py is running!")
    # ── Paths & Config ────────────────────────────────────────
    cfg      = load_config('configs/coach_rules.json')
    video_fp = 'videos/practice_video.mp4'
    model_fp = 'models/yolov8n-pose.pt'
    out_vid  = 'outputs/annotated_videos/graded_output.mp4'
    rpt_dir  = 'outputs/grading_reports'
    os.makedirs(os.path.dirname(out_vid), exist_ok=True)
    os.makedirs(rpt_dir, exist_ok=True)

    # ── Init Modules ──────────────────────────────────────────
    detector    = PoseDetector(model_fp)
    # defaults: ball_confidence=0.5, ball_drop_px=5, group_motion_px=3
    snapper     = SnapDetector()
    classifier  = PlayClassifier(cfg)
    step_grader = OLStepGrader(cfg, frame_height=None)  # set after reading first frame
    tracker     = OLPositionTracker()
    err_det     = OLErrorDetector(cfg)
    annotator   = Annotator()
    reporter    = Reporter()

    # ── FIRST PASS: Snap Detection & Pose Collection ──────────
    cap = cv2.VideoCapture(video_fp)
    if not cap.isOpened():
        print(f"❌ Error: cannot open video {video_fp}")
        return

    ret, frame = cap.read()
    if not ret:
        print("❌ Error: cannot read first frame")
        return

    h, w    = frame.shape[:2]
    fps     = cap.get(cv2.CAP_PROP_FPS)
    step_grader.frame_height = h

    frame_idx      = 0
    snap_frame     = None
    ol_kps_history = {}  # pid → list of keypoints arrays

    while ret:
        # 1) Pose detection
        boxes, kps = detector.detect_frame(frame)

        # 2) Snap detection (ball + group motion)
        if snap_frame is None and snapper.detect_snap(frame, kps):
            snap_frame = frame_idx
            print(f"🔔 Snap detected at frame {snap_frame}")

        # 3) Build OL keypoint history
        for pid, person_kps in enumerate(kps):
            ol_kps_history.setdefault(pid, []).append(person_kps)

        # next frame
        ret, frame = cap.read()
        frame_idx += 1

    cap.release()

    # fallback if still no snap
    if snap_frame is None:
        print("⚠️ No snap detected; defaulting to frame 0")
        snap_frame = 0

    # ── COMPUTE METRICS ───────────────────────────────────────
    print("▶️ OL keypoint history lengths:", 
          {pid: len(seq) for pid, seq in ol_kps_history.items()})

    step_metrics    = step_grader.grade_steps(ol_kps_history, snap_frame, fps)
    print("▶️ step_metrics:", step_metrics)

    play_info       = classifier.classify_play(ol_kps_history, snap_frame, fps)
    print(f"🏷️  Play classified: {play_info}")

    errors_detected = err_det.detect_errors(step_metrics, None, None)
    print("▶️ errors_detected:", errors_detected)

    # optional: sample positions every half‐second
    ol_seq = []
    for f in range(frame_idx):
        frame_kps = {pid: seq[f] for pid, seq in ol_kps_history.items() if f < len(seq)}
        ol_seq.append(frame_kps)
    position_data = tracker.track_positions(ol_seq, frame_interval=int(fps/2))
    print("▶️ position_data rows:", len(position_data))
    
    
    # For each pid, compute its hip‐center at snap_frame for matching
    tracked_centers, tracked_pids = [], []
    for pid, seq in ol_kps_history.items():
        if snap_frame < len(seq) and seq[snap_frame].shape[0] >= 13:
            k0 = seq[snap_frame]
            # average hip x,y
            cx = (k0[11][0] + k0[12][0]) / 2
            cy = (k0[11][1] + k0[12][1]) / 2
            tracked_centers.append([cx, cy])
            tracked_pids.append(pid)

    if tracked_centers:
        nn = NearestNeighbors(n_neighbors=1).fit(tracked_centers)
    else:
        nn = None
    

    # ── SECOND PASS: Annotate & Write Video ───────────────────
    cap    = cv2.VideoCapture(video_fp)
    writer = cv2.VideoWriter(
        out_vid,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- DEBUG: print how many players detected this frame ---
        boxes, kps = detector.detect_frame(frame)
        #use below to debug player detection
        #print(f"[Frame {frame_idx}] Detected {len(boxes)} players")

        # --- Optional: draw the football box in blue ---
        ball_box = snapper.detect_ball_box(frame)
        if ball_box is not None:
            frame = annotator.draw_ball_box(frame, ball_box)

        # --- Assign persistent IDs via nearest‐neighbor matching ---
        if nn is not None and len(boxes) > 0:
            det_centers = [((x1 + x2) / 2, (y1 + y2) / 2) 
                           for x1, y1, x2, y2 in boxes]
            _, indices = nn.kneighbors(det_centers, return_distance=True)
            pid_labels = [tracked_pids[idx] for idx in indices.flatten()]
        else:
            # fallback: label in detection order
            pid_labels = list(range(len(boxes)))

        # --- Draw boxes + keypoints + IDs ---
        frame = annotator.draw_pose_with_ids(frame, boxes, kps, pid_labels)
        # If you ever want just boxes/keypoints (no IDs), use:
        # frame = annotator.draw_pose(frame, boxes, kps)

        # --- Overlay grades & errors ---
        frame = annotator.draw_annotations(frame, step_metrics, errors_detected)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # ── EXPORT REPORTS ────────────────────────────────────────
    reporter.export(step_metrics, position_data, errors_detected, rpt_dir)
    print("✅ Done.")
    print("   Video →", out_vid)
    print("   Reports →", rpt_dir)


if __name__ == "__main__":
    main()
