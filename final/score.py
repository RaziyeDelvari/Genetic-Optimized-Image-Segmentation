import os
import cv2
import numpy as np
import csv

output_dir = "output"
gt_dir = "gt"
results_csv = "scores.csv"

def stem(filename):
    return os.path.splitext(filename)[0]

def confusion_matrix_elements(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    return tp, fp, fn, tn

def compute_iou(tp, fp, fn):
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 1.0

def compute_dice(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom > 0 else 1.0

# --------------------------------------------------
# Build GT lookup table (by filename stem)
# --------------------------------------------------
gt_lookup = {}
for gt_file in os.listdir(gt_dir):
    gt_lookup[stem(gt_file)] = os.path.join(gt_dir, gt_file)

ious = []
dices = []
total_tp = total_fp = total_fn = total_tn = 0

with open(results_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "IoU", "Dice/F1", "TP", "FP", "FN", "TN"])

    for pred_file in os.listdir(output_dir):
        pred_stem = stem(pred_file)

        if pred_stem not in gt_lookup:
            print(f"[WARNING] No GT match for {pred_file}, skipping.")
            continue

        pred_path = os.path.join(output_dir, pred_file)
        gt_path = gt_lookup[pred_stem]

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            print(f"[ERROR] Failed to read {pred_file}, skipping.")
            continue

        if pred.shape != gt.shape:
            pred = cv2.resize(
                pred,
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        pred_bin = pred > 0
        gt_bin = gt > 0

        tp, fp, fn, tn = confusion_matrix_elements(pred_bin, gt_bin)
        iou = compute_iou(tp, fp, fn)
        dice = compute_dice(tp, fp, fn)

        ious.append(iou)
        dices.append(dice)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        print(
            f"{pred_file:25s} | IoU: {iou:.4f} | Dice: {dice:.4f} | "
            f"TP: {tp} FP: {fp} FN: {fn} TN: {tn}"
        )

        writer.writerow([
            pred_file,
            f"{iou:.4f}",
            f"{dice:.4f}",
            tp, fp, fn, tn
        ])

# --------------------------------------------------
# Final aggregated metrics
# --------------------------------------------------
mean_iou = np.mean(ious) if ious else 0.0
mean_dice = np.mean(dices) if dices else 0.0

print("\n=== Aggregated Metrics ===")
print(f"Mean IoU      : {mean_iou:.4f}")
print(f"Mean Dice/F1  : {mean_dice:.4f}")
print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, TN: {total_tn}")

conf_matrix = np.array([
    [total_tn, total_fp],
    [total_fn, total_tp]
])

print("\nConfusion Matrix (Global):")
print(conf_matrix)
