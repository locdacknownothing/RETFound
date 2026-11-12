import csv
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util.lr_sched as lr_sched
import util.misc as misc
from util.seg_datasets import pad_to_multiple
from util.seg_metrics import evaluate_dataset_predictions, find_best_threshold


def train_one_epoch_seg(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 1.0,
    log_writer=None,
    args=None,
) -> Dict[str, float]:
    """Train the vessel segmentation model for one epoch with bf16 autocast."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    print_freq = 10
    accum_iter = getattr(args, "accum_iter", 1)
    optimizer.zero_grad(set_to_none=True)

    num_steps = len(data_loader)
    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, f"Epoch: [{epoch}]")
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / num_steps + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(samples)
            loss, loss_dict = criterion(outputs, targets)

        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        scaled_loss = loss / accum_iter
        scaled_loss.backward()

        if (data_iter_step + 1) % accum_iter == 0 or (data_iter_step + 1) == num_steps:
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        if "bce" in loss_dict:
            metric_logger.update(bce=loss_dict["bce"])
        if "dice" in loss_dict:
            metric_logger.update(dice_loss=loss_dict["dice"])

        max_lr = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(lr=max_lr)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / num_steps + epoch) * 1000)
            log_writer.add_scalar("loss/train", loss_value, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _forward_single_view(
    model: nn.Module,
    img: torch.Tensor,
    crop_size: int = 448,
    sliding_window: bool = False,
) -> torch.Tensor:
    """Run forward pass on a `(1, 3, H, W)` image tensor and return probabilities `(1, 1, H, W)`."""
    _, _, H, W = img.shape
    if not sliding_window:
        padded, (orig_h, orig_w) = pad_to_multiple(img, multiple=16, mode="reflect")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(padded)
            if isinstance(logits, dict):
                logits = logits["logits"]
        probs = torch.sigmoid(logits.float())[:, :, :orig_h, :orig_w]
        return probs

    # Sliding-window fallback with 50% overlap
    stride = crop_size // 2
    pad_h = max(0, crop_size - H)
    pad_w = max(0, crop_size - W)
    if pad_h > 0 or pad_w > 0:
        img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        img_pad = img
    _, _, Hp, Wp = img_pad.shape

    ys = sorted(set(list(range(0, max(1, Hp - crop_size + 1), stride)) + [Hp - crop_size]))
    xs = sorted(set(list(range(0, max(1, Wp - crop_size + 1), stride)) + [Wp - crop_size]))

    acc_prob = torch.zeros((1, 1, Hp, Wp), device=img.device, dtype=torch.float32)
    acc_weight = torch.zeros((1, 1, Hp, Wp), device=img.device, dtype=torch.float32)

    for y in ys:
        for x in xs:
            patch = img_pad[:, :, y : y + crop_size, x : x + crop_size]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(patch)
                if isinstance(logits, dict):
                    logits = logits["logits"]
            p = torch.sigmoid(logits.float())
            acc_prob[:, :, y : y + crop_size, x : x + crop_size] += p
            acc_weight[:, :, y : y + crop_size, x : x + crop_size] += 1.0

    probs = (acc_prob / acc_weight.clamp_min(1.0))[:, :, :H, :W]
    return probs


@torch.no_grad()
def infer_single_image(
    model: nn.Module,
    img: torch.Tensor,
    crop_size: int = 448,
    sliding_window: bool = False,
    tta: bool = False,
) -> torch.Tensor:
    """Predict vessel probability map `(1, 1, H, W)` with optional 4x flip TTA."""
    p0 = _forward_single_view(model, img, crop_size=crop_size, sliding_window=sliding_window)
    if not tta:
        return p0

    # Horizontal flip
    img_h = torch.flip(img, dims=[-1])
    ph = torch.flip(
        _forward_single_view(model, img_h, crop_size=crop_size, sliding_window=sliding_window),
        dims=[-1],
    )
    # Vertical flip
    img_v = torch.flip(img, dims=[-2])
    pv = torch.flip(
        _forward_single_view(model, img_v, crop_size=crop_size, sliding_window=sliding_window),
        dims=[-2],
    )
    # HV flip
    img_hv = torch.flip(img, dims=[-2, -1])
    phv = torch.flip(
        _forward_single_view(model, img_hv, crop_size=crop_size, sliding_window=sliding_window),
        dims=[-2, -1],
    )
    return 0.25 * (p0 + ph + pv + phv)


@torch.no_grad()
def evaluate_seg(
    data_loader: Iterable,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    args,
    epoch: int,
    mode: str = "val",
    log_writer=None,
    save_preds: bool = False,
    opt_threshold: Optional[float] = None,
) -> Tuple[Dict[str, float], float, float]:
    """Evaluate segmentation model on validation or test split.

    Returns:
      - stats_dict: dictionary of pooled and mean metrics
      - val_score: pooled FOV Dice at threshold 0.5 (used for checkpoint selection)
      - best_tau: threshold in [0.1, 0.9] maximizing pooled FOV Dice on this split
    """
    model.eval()
    task_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(task_dir, exist_ok=True)

    probs_list: List[np.ndarray] = []
    gts_list: List[np.ndarray] = []
    fovs_list: List[np.ndarray] = []
    names_list: List[str] = []
    losses: List[float] = []

    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        fovs = batch["fov"].numpy()
        names = batch["name"]

        B = images.shape[0]
        for b in range(B):
            img_b = images[b : b + 1]
            lbl_b = labels[b : b + 1]
            prob_b = infer_single_image(
                model,
                img_b,
                crop_size=args.crop_size,
                sliding_window=getattr(args, "sliding_window", False),
                tta=getattr(args, "tta", False) if mode == "test" else False,
            )
            # Compute validation loss from logit inverse of prob_b
            eps = 1e-6
            logit_b = torch.logit(prob_b.clamp(eps, 1.0 - eps))
            loss_b, _ = criterion(logit_b, lbl_b)
            losses.append(float(loss_b.item()))

            prob_np = prob_b.squeeze().cpu().numpy().astype(np.float32)
            gt_np = lbl_b.squeeze().cpu().numpy().astype(np.uint8)
            fov_np = fovs[b].astype(np.uint8)

            probs_list.append(prob_np)
            gts_list.append(gt_np)
            fovs_list.append(fov_np)
            names_list.append(str(names[b]))

    mean_loss = float(np.mean(losses)) if losses else 0.0
    res_05 = evaluate_dataset_predictions(
        probs_list, gts_list, fov_masks=fovs_list, names=names_list, threshold=0.5
    )

    if opt_threshold is None:
        best_tau, best_dice = find_best_threshold(
            probs_list, gts_list, fov_masks=fovs_list
        )
    else:
        best_tau = float(opt_threshold)
        res_opt = evaluate_dataset_predictions(
            probs_list, gts_list, fov_masks=fovs_list, names=names_list, threshold=best_tau
        )
        best_dice = float(res_opt["pooled"]["dice"])

    pooled = res_05["pooled"]
    mean_m = res_05["mean"]
    std_m = res_05["std"]

    print(
        f"[{mode.upper()} Epoch {epoch}] loss: {mean_loss:.4f} | "
        f"Pooled@0.5 -> Dice: {pooled['dice']:.4f}, IoU: {pooled['jaccard']:.4f}, "
        f"AUC: {pooled['auc']:.4f}, Se: {pooled['se']:.4f}, Sp: {pooled['sp']:.4f}, "
        f"Acc: {pooled['acc']:.4f}, MCC: {pooled['mcc']:.4f} | "
        f"Dice@{best_tau:.2f}: {best_dice:.4f}"
    )
    print(
        f"[{mode.upper()} Per-Img Mean±Std@0.5] "
        f"Dice: {mean_m['dice']:.4f}±{std_m['dice']:.4f}, "
        f"AUC: {mean_m['auc']:.4f}±{std_m['auc']:.4f}, "
        f"Se: {mean_m['se']:.4f}±{std_m['se']:.4f}, "
        f"Sp: {mean_m['sp']:.4f}±{std_m['sp']:.4f}, "
        f"MCC: {mean_m['mcc']:.4f}±{std_m['mcc']:.4f}"
    )

    if log_writer is not None:
        log_writer.add_scalar(f"loss/{mode}", mean_loss, epoch)
        for k, v in pooled.items():
            log_writer.add_scalar(f"perf_{mode}_pooled/{k}", v, epoch)
        for k, v in mean_m.items():
            log_writer.add_scalar(f"perf_{mode}_mean/{k}", v, epoch)
        log_writer.add_scalar(f"perf_{mode}_pooled/dice_opt_tau", best_dice, epoch)

    # Save CSV metrics
    fieldnames = [
        "epoch",
        "split",
        "name",
        "threshold",
        "loss",
        "dice",
        "jaccard",
        "auc",
        "se",
        "sp",
        "acc",
        "mcc",
    ]
    if mode == "val":
        csv_path = os.path.join(task_dir, "metrics_val.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "epoch": epoch,
                    "split": "val_pooled",
                    "name": "POOLED_0.5",
                    "threshold": 0.5,
                    "loss": f"{mean_loss:.6f}",
                    **{k: f"{pooled[k]:.6f}" for k in ["dice", "jaccard", "auc", "se", "sp", "acc", "mcc"]},
                }
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "split": "val_mean",
                    "name": "MEAN_0.5",
                    "threshold": 0.5,
                    "loss": f"{mean_loss:.6f}",
                    **{k: f"{mean_m[k]:.6f}" for k in ["dice", "jaccard", "auc", "se", "sp", "acc", "mcc"]},
                }
            )
    elif mode == "test":
        csv_path = os.path.join(task_dir, "metrics_test.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in res_05["per_image"]:
                writer.writerow(
                    {
                        "epoch": epoch,
                        "split": "test_image",
                        "name": row["name"],
                        "threshold": 0.5,
                        "loss": f"{mean_loss:.6f}",
                        **{k: f"{row[k]:.6f}" for k in ["dice", "jaccard", "auc", "se", "sp", "acc", "mcc"]},
                    }
                )
            writer.writerow(
                {
                    "epoch": epoch,
                    "split": "test_pooled",
                    "name": "POOLED_0.5",
                    "threshold": 0.5,
                    "loss": f"{mean_loss:.6f}",
                    **{k: f"{pooled[k]:.6f}" for k in ["dice", "jaccard", "auc", "se", "sp", "acc", "mcc"]},
                }
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "split": "test_mean",
                    "name": "MEAN_0.5",
                    "threshold": 0.5,
                    "loss": f"{mean_loss:.6f}",
                    **{k: f"{mean_m[k]:.6f}" for k in ["dice", "jaccard", "auc", "se", "sp", "acc", "mcc"]},
                }
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "split": "test_std",
                    "name": "STD_0.5",
                    "threshold": 0.5,
                    "loss": 0.0,
                    **{k: f"{std_m[k]:.6f}" for k in ["dice", "jaccard", "auc", "se", "sp", "acc", "mcc"]},
                }
            )

    if save_preds:
        pred_dir = os.path.join(task_dir, "pred")
        os.makedirs(pred_dir, exist_ok=True)
        for name, prob_np, fov_np in zip(names_list, probs_list, fovs_list):
            prob_u8 = np.clip(np.round(prob_np * 255.0), 0, 255).astype(np.uint8)
            bin_u8 = (((prob_np >= 0.5) & (fov_np > 0)).astype(np.uint8)) * 255
            Image.fromarray(prob_u8, mode="L").save(
                os.path.join(pred_dir, f"{name}_prob.png")
            )
            Image.fromarray(bin_u8, mode="L").save(
                os.path.join(pred_dir, f"{name}_bin.png")
            )

    stats_out = {
        "loss": mean_loss,
        **{f"pooled_{k}": v for k, v in pooled.items()},
        **{f"mean_{k}": v for k, v in mean_m.items()},
        **{f"std_{k}": v for k, v in std_m.items()},
        "best_tau": best_tau,
        "dice_at_best_tau": best_dice,
    }
    return stats_out, float(pooled["dice"]), float(best_tau)
