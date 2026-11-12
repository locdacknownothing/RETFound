import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score


def confusion_from_binary(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
) -> Tuple[int, int, int, int]:
    """Compute (TP, FP, TN, FN) inside `fov_mask` (if provided)."""
    pred_b = pred_bin.astype(bool)
    gt_b = gt_bin.astype(bool)
    if fov_mask is not None:
        m = fov_mask.astype(bool)
        pred_b = pred_b[m]
        gt_b = gt_b[m]
    else:
        pred_b = pred_b.ravel()
        gt_b = gt_b.ravel()

    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, np.logical_not(gt_b)).sum())
    tn = int(np.logical_and(np.logical_not(pred_b), np.logical_not(gt_b)).sum())
    fn = int(np.logical_and(np.logical_not(pred_b), gt_b).sum())
    return tp, fp, tn, fn


def metrics_from_confusion(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """Compute Dice, Jaccard, Sensitivity, Specificity, Accuracy, and MCC from confusion counts."""
    eps = 1e-8
    tp_f, fp_f, tn_f, fn_f = float(tp), float(fp), float(tn), float(fn)
    total = tp_f + fp_f + tn_f + fn_f

    dice = (2.0 * tp_f) / (2.0 * tp_f + fp_f + fn_f + eps)
    jaccard = tp_f / (tp_f + fp_f + fn_f + eps)
    se = tp_f / (tp_f + fn_f + eps)
    sp = tn_f / (tn_f + fp_f + eps)
    acc = (tp_f + tn_f) / (total + eps)

    denom = math.sqrt(
        max((tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f), 0.0)
    )
    mcc = (tp_f * tn_f - fp_f * fn_f) / (denom + eps)

    return {
        "dice": float(dice),
        "jaccard": float(jaccard),
        "se": float(se),
        "sp": float(sp),
        "acc": float(acc),
        "mcc": float(mcc),
    }


def compute_image_metrics(
    prob: np.ndarray,
    gt: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], Tuple[int, int, int, int]]:
    """Compute per-image metrics: FOV-masked thresholded metrics + full-image AUC-ROC."""
    gt_bin = (gt > 0.5).astype(np.uint8)
    pred_bin = (prob >= threshold).astype(np.uint8)
    if fov_mask is not None:
        pred_bin = pred_bin * (fov_mask > 0).astype(np.uint8)

    tp, fp, tn, fn = confusion_from_binary(pred_bin, gt_bin, fov_mask=fov_mask)
    m = metrics_from_confusion(tp, fp, tn, fn)

    # AUC is computed over all pixels (not restricted to FOV mask)
    flat_gt = gt_bin.ravel()
    flat_prob = prob.ravel().astype(np.float32)
    if flat_gt.min() != flat_gt.max():
        m["auc"] = float(roc_auc_score(flat_gt, flat_prob))
    else:
        m["auc"] = 0.0

    return m, (tp, fp, tn, fn)


def find_best_threshold(
    probs: List[np.ndarray],
    gts: List[np.ndarray],
    fov_masks: Optional[List[np.ndarray]] = None,
    num_steps: int = 81,
) -> Tuple[float, float]:
    """Sweep thresholds in [0.10, 0.90] to maximize pooled FOV Dice."""
    thresholds = np.linspace(0.10, 0.90, num_steps)
    best_tau = 0.5
    best_dice = -1.0

    # Flatten FOV pixels once for fast sweep
    flat_probs = []
    flat_gts = []
    for i, (p, g) in enumerate(zip(probs, gts)):
        if fov_masks is not None and fov_masks[i] is not None:
            m = fov_masks[i].astype(bool)
            flat_probs.append(p[m].astype(np.float32))
            flat_gts.append((g[m] > 0.5).astype(bool))
        else:
            flat_probs.append(p.ravel().astype(np.float32))
            flat_gts.append((g.ravel() > 0.5).astype(bool))

    all_p = np.concatenate(flat_probs, axis=0)
    all_g = np.concatenate(flat_gts, axis=0)
    gt_sum = float(all_g.sum())

    for tau in thresholds:
        pred_b = all_p >= tau
        inter = float(np.logical_and(pred_b, all_g).sum())
        pred_sum = float(pred_b.sum())
        dice = (2.0 * inter) / (pred_sum + gt_sum + 1e-8)
        if dice > best_dice:
            best_dice = float(dice)
            best_tau = float(tau)

    return best_tau, best_dice


def evaluate_dataset_predictions(
    probs: List[np.ndarray],
    gts: List[np.ndarray],
    fov_masks: Optional[List[np.ndarray]] = None,
    names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Compute per-image, mean±std, and pooled dataset metrics."""
    per_image: List[Dict[str, object]] = []
    total_tp = total_fp = total_tn = total_fn = 0
    all_probs_full = []
    all_gts_full = []

    for idx, (prob, gt) in enumerate(zip(probs, gts)):
        fov = fov_masks[idx] if fov_masks is not None else None
        img_m, (tp, fp, tn, fn) = compute_image_metrics(
            prob, gt, fov_mask=fov, threshold=threshold
        )
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        name = names[idx] if names is not None else f"img_{idx:03d}"
        row: Dict[str, object] = {"name": name, **img_m}
        per_image.append(row)

        all_probs_full.append(prob.ravel().astype(np.float32))
        all_gts_full.append((gt.ravel() > 0.5).astype(np.uint8))

    pooled = metrics_from_confusion(total_tp, total_fp, total_tn, total_fn)
    concat_gt = np.concatenate(all_gts_full, axis=0)
    concat_prob = np.concatenate(all_probs_full, axis=0)
    if concat_gt.min() != concat_gt.max():
        pooled["auc"] = float(roc_auc_score(concat_gt, concat_prob))
    else:
        pooled["auc"] = 0.0

    keys = ["dice", "jaccard", "se", "sp", "acc", "mcc", "auc"]
    mean_metrics = {
        k: float(np.mean([float(r[k]) for r in per_image])) for k in keys
    }
    std_metrics = {
        k: float(np.std([float(r[k]) for r in per_image], ddof=0)) for k in keys
    }

    return {
        "threshold": float(threshold),
        "pooled": pooled,
        "mean": mean_metrics,
        "std": std_metrics,
        "per_image": per_image,
    }
