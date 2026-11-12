#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import util.lr_decay as lrd
import util.misc as misc
from engine_segment import evaluate_seg, train_one_epoch_seg
from models_seg import build_seg_model
from util.lora import trainable_and_buffer_state_dict
from util.seg_datasets import (
    VesselSegDataset,
    build_group_aware_train_val_split,
    collate_multi_crop,
    infer_dataset_name,
    pair_split_files,
)
from util.seg_losses import VesselSegmentationLoss

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_args_parser():
    parser = argparse.ArgumentParser(
        "RETFound Retinal Vessel Segmentation Fine-Tuning", add_help=True
    )

    # ---- Dataset & split parameters
    parser.add_argument(
        "--dataset",
        default="drive",
        choices=["drive", "chase", "stare"],
        help="Dataset type for filename pairing and source-ID grouping",
    )
    parser.add_argument(
        "--data_path",
        default="./data/drive_aug",
        type=str,
        help="Path to dataset root containing train/ and test/",
    )
    parser.add_argument(
        "--val_ids",
        default="",
        type=str,
        help="Optional comma-separated list of source IDs to hold out for validation",
    )
    parser.add_argument(
        "--n_val_groups",
        default=0,
        type=int,
        help="Number of source IDs to hold out for validation (0 = dataset default: 4 for DRIVE)",
    )
    parser.add_argument("--datasets_seed", default=2026, type=int)

    # ---- Model & adaptation parameters
    parser.add_argument(
        "--model",
        default="RETFound_mae",
        choices=["unet", "RETFound_mae", "RETFound_dinov2"],
        help="Segmentation model architecture",
    )
    parser.add_argument(
        "--finetune",
        default="RETFound_mae_natureCFP",
        type=str,
        help="Pre-trained checkpoint ID on HuggingFace Hub or local .pth path",
    )
    parser.add_argument(
        "--adaptation",
        default="lora",
        choices=["frozen", "lora", "finetune"],
        help="ViT backbone adaptation mode",
    )
    parser.add_argument("--lora_rank", default=16, type=int)
    parser.add_argument("--lora_alpha", default=16.0, type=float)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument(
        "--vit_layers",
        default="5,11,17,23",
        type=str,
        help="0-based ViT block indices to extract multi-scale pyramid features from",
    )
    parser.add_argument("--base_ch", default=32, type=int, help="Base channels for CNN branch")
    parser.add_argument("--drop_path", default=0.1, type=float, help="ViT drop path rate")
    parser.add_argument(
        "--deep_supervision",
        action="store_true",
        default=True,
        help="Enable auxiliary deep supervision heads at strides 2, 4, 8",
    )
    parser.add_argument(
        "--no_deep_supervision",
        action="store_false",
        dest="deep_supervision",
    )
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        default=False,
        help="Enable ViT gradient checkpointing to reduce peak VRAM (e.g. for parallel runs)",
    )

    # ---- Resolution & augmentation
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--crops_per_image", default=2, type=int)
    parser.add_argument(
        "--enhance",
        action="store_true",
        default=False,
        help="Apply CLAHE contrast enhancement before normalization",
    )

    # ---- Training & optimization
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--lr_decoder", default=3e-4, type=float)
    parser.add_argument("--lr_lora", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=3e-5, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--layer_decay", default=0.75, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--clip_grad", default=1.0, type=float)
    parser.add_argument(
        "--loss",
        default="bce,dice",
        type=str,
        help="Comma-separated loss terms: bce,dice[,cldice,skelrec]",
    )
    parser.add_argument("--pos_weight", default=0.0, type=float)

    # ---- Evaluation & runtime
    parser.add_argument("--eval_every", default=5, type=int)
    parser.add_argument("--tta", action="store_true", default=False)
    parser.add_argument("--sliding_window", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--output_dir", default="./output_dir", type=str)
    parser.add_argument("--log_dir", default="./output_logs", type=str)
    parser.add_argument("--task", default="", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--savemodel", action="store_true", default=True)

    return parser


def build_optimizer_param_groups(model, args) -> List[Dict[str, object]]:
    """Build differential LR parameter groups compatible with util/lr_sched.py."""
    args.lr = float(args.lr_decoder)
    param_groups: List[Dict[str, object]] = []

    dec_decay, dec_no_decay = [], []
    lora_decay, lora_no_decay = [], []

    vit_prefix = "vit_pyramid.vit."
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith(vit_prefix):
            if "lora_A" in name or "lora_B" in name:
                if param.ndim == 1 or name.endswith(".bias"):
                    lora_no_decay.append(param)
                else:
                    lora_decay.append(param)
            # Non-LoRA ViT parameters in 'finetune' mode are handled below via param_groups_lrd
        else:
            if param.ndim == 1 or name.endswith(".bias") or "norm" in name or "bn" in name:
                dec_no_decay.append(param)
            else:
                dec_decay.append(param)

    if dec_decay:
        param_groups.append(
            {"params": dec_decay, "lr_scale": 1.0, "weight_decay": args.weight_decay, "group_name": "decoder_decay"}
        )
    if dec_no_decay:
        param_groups.append(
            {"params": dec_no_decay, "lr_scale": 1.0, "weight_decay": 0.0, "group_name": "decoder_no_decay"}
        )

    lora_scale = float(args.lr_lora) / float(args.lr_decoder)
    if lora_decay:
        param_groups.append(
            {"params": lora_decay, "lr_scale": lora_scale, "weight_decay": args.weight_decay, "group_name": "lora_decay"}
        )
    if lora_no_decay:
        param_groups.append(
            {"params": lora_no_decay, "lr_scale": lora_scale, "weight_decay": 0.0, "group_name": "lora_no_decay"}
        )

    if getattr(model, "use_vit", False) and args.adaptation == "finetune":
        vit_module = model.vit_pyramid.vit
        no_wd = vit_module.no_weight_decay() if hasattr(vit_module, "no_weight_decay") else []
        backbone_scale = float(args.lr_backbone) / float(args.lr_decoder)
        vit_groups = lrd.param_groups_lrd(
            vit_module,
            weight_decay=args.weight_decay,
            no_weight_decay_list=no_wd,
            layer_decay=args.layer_decay,
        )
        for g in vit_groups:
            g["params"] = [p for p in g["params"] if p.requires_grad]
            if g["params"]:
                g["lr_scale"] = float(g["lr_scale"]) * backbone_scale
                param_groups.append(g)

    return param_groups


def save_seg_checkpoint(
    path: str,
    model_without_ddp: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_dice: float,
    best_tau: float,
    args,
):
    payload = {
        "model": trainable_and_buffer_state_dict(model_without_ddp),
        "epoch": int(epoch),
        "best_val_dice": float(best_val_dice),
        "best_tau": float(best_tau),
        "args": vars(args),
    }
    torch.save(payload, path)


def main(args):
    if not args.task:
        adapt_tag = args.adaptation if args.model != "unet" else "full"
        args.task = f"{args.model}_{args.dataset}_{adapt_tag}_seed{args.seed}"

    task_out_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(task_out_dir, exist_ok=True)

    # Set random seeds for reproducibility
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    device = torch.device(args.device)
    dataset_name = infer_dataset_name(args.data_path, args.dataset)
    args.dataset = dataset_name

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(json.dumps(vars(args), indent=2))

    # ---- Build group-aware train/val split and test split
    val_ids = [x.strip() for x in args.val_ids.split(",") if x.strip()] if args.val_ids else None
    train_records, val_records, split_info = build_group_aware_train_val_split(
        data_path=args.data_path,
        dataset_name=dataset_name,
        n_val_groups=args.n_val_groups if args.n_val_groups > 0 else None,
        val_ids=val_ids,
        datasets_seed=args.datasets_seed,
    )
    test_records = pair_split_files(args.data_path, "test", dataset_name)
    split_info["num_test_files"] = len(test_records)
    split_info["test_groups"] = sorted({str(r["group_id"]) for r in test_records})

    split_json_path = os.path.join(task_out_dir, "split.json")
    with open(split_json_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(
        f"[Dataset {dataset_name.upper()}] train_files={len(train_records)} "
        f"(groups={split_info['train_groups']}), "
        f"val_files={len(val_records)} (groups={split_info['val_groups']}), "
        f"test_files={len(test_records)}"
    )

    dataset_train = VesselSegDataset(
        train_records,
        is_train=True,
        crop_size=args.crop_size,
        crops_per_image=args.crops_per_image,
        enhance=args.enhance,
    )
    dataset_val = VesselSegDataset(
        val_records,
        is_train=False,
        crop_size=args.crop_size,
        crops_per_image=1,
        enhance=args.enhance,
    )
    dataset_test = VesselSegDataset(
        test_records,
        is_train=False,
        crop_size=args.crop_size,
        crops_per_image=1,
        enhance=args.enhance,
    )

    g = torch.Generator()
    g.manual_seed(seed)
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_multi_crop,
        generator=g,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # ---- Build model & criterion
    model = build_seg_model(args)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        msg = model.load_state_dict(ckpt["model"], strict=False)
        print(f"[Resume] Loaded state from {args.resume} (missing={len(msg.missing_keys)})")

    model.to(device)
    model_without_ddp = model

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[Model] {args.model} (adaptation={args.adaptation}) | "
        f"trainable params: {n_trainable / 1e6:.2f}M / total: {n_total / 1e6:.2f}M"
    )

    criterion = VesselSegmentationLoss(
        loss_terms=args.loss,
        pos_weight=args.pos_weight if args.pos_weight > 0 else None,
    ).to(device)

    param_groups = build_optimizer_param_groups(model_without_ddp, args)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr_decoder)

    log_writer = None
    if args.log_dir and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))

    if args.eval:
        evaluate_seg(
            data_loader_test,
            model,
            criterion,
            device,
            args,
            epoch=0,
            mode="test",
            log_writer=None,
            save_preds=True,
        )
        return

    print(f"Start vessel segmentation training for {args.epochs} epochs")
    start_time = time.time()
    best_val_dice = -1.0
    best_val_tau = 0.5
    best_epoch = -1
    ckpt_best_path = os.path.join(task_out_dir, "checkpoint-best.pth")

    for epoch in range(args.epochs):
        train_stats = train_one_epoch_seg(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_norm=args.clip_grad,
            log_writer=log_writer,
            args=args,
        )

        should_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == 0) or (epoch + 1 == args.epochs)
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_trainable_params": n_trainable,
        }

        if should_eval:
            val_stats, val_dice, val_tau = evaluate_seg(
                data_loader_val,
                model,
                criterion,
                device,
                args,
                epoch=epoch,
                mode="val",
                log_writer=log_writer,
                save_preds=False,
            )
            log_stats.update({f"val_{k}": v for k, v in val_stats.items()})
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_val_tau = val_tau
                best_epoch = epoch
                if args.savemodel:
                    save_seg_checkpoint(
                        ckpt_best_path,
                        model_without_ddp,
                        optimizer,
                        epoch,
                        best_val_dice,
                        best_val_tau,
                        args,
                    )
            print(
                f"Best val epoch = {best_epoch}, Best val Dice@0.5 = {best_val_dice:.4f} "
                f"(val opt_tau = {best_val_tau:.2f})"
            )

        if log_writer is not None:
            log_writer.flush()

        with open(os.path.join(task_out_dir, "log.txt"), "a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    # ---- Final Test Evaluation with Best Validation Checkpoint
    if os.path.isfile(ckpt_best_path):
        ckpt = torch.load(ckpt_best_path, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(ckpt["model"], strict=False)
        model.to(device)
        best_epoch = ckpt.get("epoch", best_epoch)
        best_val_tau = ckpt.get("best_tau", best_val_tau)
        print(
            f"\n=== Evaluating on TEST split with best checkpoint from epoch {best_epoch} "
            f"(val Dice@0.5={ckpt.get('best_val_dice', 0.0):.4f}) ==="
        )
    else:
        print("\n=== Evaluating on TEST split with final epoch model ===")

    test_stats, test_dice, _ = evaluate_seg(
        data_loader_test,
        model,
        criterion,
        device,
        args,
        epoch=best_epoch,
        mode="test",
        log_writer=log_writer,
        save_preds=True,
        opt_threshold=best_val_tau,
    )

    summary_path = os.path.join(task_out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": args.task,
                "model": args.model,
                "adaptation": args.adaptation,
                "seed": args.seed,
                "best_val_epoch": best_epoch,
                "best_val_dice_05": best_val_dice,
                "best_val_tau": best_val_tau,
                "test": test_stats,
            },
            f,
            indent=2,
        )

    total_time = time.time() - start_time
    print(f"Training & evaluation completed in {datetime.timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    parser = get_args_parser()
    parsed_args = parser.parse_args()
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
