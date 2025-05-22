# evaluate.py

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
)
from main_pipeline import run_pipeline

# 1) Load your exported ground truth
with open("ground_truth.json", "r") as f:
    gt_data = json.load(f)

# Prepare lists
y_gt_snap, y_gt_play, y_gt_time, y_gt_dist = [], [], [], []
y_pr_snap, y_pr_play, y_pr_time, y_pr_dist = [], [], [], []

# 2) Iterate through each labeled task
for task in gt_data:
    path = task["data"]["video"]
    ann  = task["annotations"][0]["result"]

    # unpack ground truth
    gt = {}
    for r in ann:
        lbl = r["label"][0]
        if "number" in r["value"]:
            gt[lbl] = r["value"]["number"]
        elif "choice" in r["value"]:
            gt[lbl] = r["value"]["choice"]

    y_gt_snap.append(int(gt["snapFrame"]))
    y_gt_play.append(gt["playType"])
    y_gt_time.append(float(gt["firstStepFrame"]))       # in frames
    y_gt_dist.append(float(gt["firstStepDistanceIn"]))  # in inches

    # 3) Run your pipeline for this video
    pred = run_pipeline(path)
    y_pr_snap.append(pred["snap_frame"])
    y_pr_play.append(pred["play_type"])
    # convert ms back to frames: ms ÷ (1000/fps)
    fps = pred["fps"]
    y_pr_time.append(pred["first_step_time_ms"] / (1000.0/fps))
    y_pr_dist.append(pred["first_step_distance_in"])

# 4) Metrics for snap detection (within ±2 frames = “correct”)
tol = 2
y_gt_snap_bin = [1]*len(y_gt_snap)
y_pr_snap_bin = [1 if abs(p-t)<=tol else 0 for p,t in zip(y_pr_snap, y_gt_snap)]
snap_prec = precision_score(y_gt_snap_bin, y_pr_snap_bin)
snap_rec  = recall_score(y_gt_snap_bin, y_pr_snap_bin)

# 5) Play‐type accuracy + confusion matrix
play_acc = accuracy_score(y_gt_play, y_pr_play)
cm = confusion_matrix(y_gt_play, y_pr_play, labels=list(sorted(set(y_gt_play))))

# 6) MAE for first‐step
mae_time = mean_absolute_error(y_gt_time, y_pr_time)
mae_dist = mean_absolute_error(y_gt_dist, y_pr_dist)

# 7) Print results
print(f"Snap Detection → Precision: {snap_prec:.2f}, Recall: {snap_rec:.2f}")
print(f"Play Classification Accuracy: {play_acc:.2f}")
print(f"MAE First‐Step Time (frames): {mae_time:.2f}")
print(f"MAE First‐Step Distance (in): {mae_dist:.2f}")

# 8) Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=sorted(set(y_gt_play)),
            yticklabels=sorted(set(y_gt_play)))
plt.title("Play Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 9) Error histograms
plt.figure()
plt.hist(np.array(y_pr_time)-np.array(y_gt_time), bins=10)
plt.title("First‐Step Time Error (pred – GT) in Frames")
plt.xlabel("Error (frames)")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.hist(np.array(y_pr_dist)-np.array(y_gt_dist), bins=10)
plt.title("First‐Step Distance Error (pred – GT) in Inches")
plt.xlabel("Error (inches)")
plt.ylabel("Count")
plt.show()
