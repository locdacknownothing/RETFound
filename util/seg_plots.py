import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_SERIES: Tuple[Tuple[str, str, Dict[str, Any]], ...] = (
    (
        "train_loss",
        "Train Total Loss",
        {"color": "#1f2937", "linestyle": "-", "linewidth": 1.8, "alpha": 0.9},
    ),
    (
        "val_loss",
        "Val Total Loss",
        {
            "color": "#1f77b4",
            "linestyle": "--",
            "linewidth": 1.8,
            "marker": "o",
            "markersize": 3.2,
            "alpha": 0.95,
        },
    ),
    (
        "train_bce",
        "Train BCE Loss",
        {"color": "#ff7f0e", "linestyle": "-", "linewidth": 1.4, "alpha": 0.85},
    ),
    (
        "train_dice_loss",
        "Train Dice Loss",
        {"color": "#9467bd", "linestyle": "-", "linewidth": 1.4, "alpha": 0.85},
    ),
    (
        "train_cldice_loss",
        "Train clDice Loss",
        {"color": "#8c564b", "linestyle": "-", "linewidth": 1.4, "alpha": 0.85},
    ),
    (
        "train_skelrec_loss",
        "Train SkelRec Loss",
        {"color": "#bcbd22", "linestyle": "-", "linewidth": 1.4, "alpha": 0.85},
    ),
)

METRIC_SERIES: Tuple[Tuple[str, str, Dict[str, Any]], ...] = (
    (
        "val_pooled_dice",
        "Val DSC@0.5",
        {
            "color": "#2ca02c",
            "linestyle": "-",
            "linewidth": 2.0,
            "marker": "o",
            "markersize": 3.8,
        },
    ),
    (
        "val_mean_cldice",
        "Val clDice@0.5",
        {
            "color": "#d62728",
            "linestyle": "-",
            "linewidth": 2.0,
            "marker": "^",
            "markersize": 3.8,
        },
    ),
)


def load_training_log(log_path: str) -> List[Dict[str, Any]]:
    """Load JSON-line records from log.txt, deduplicated and sorted by epoch."""
    if not os.path.isfile(log_path):
        return []

    by_epoch: Dict[int, Dict[str, Any]] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            epoch = int(record["epoch"])
            by_epoch[epoch] = record

    return [by_epoch[ep] for ep in sorted(by_epoch)]


def _load_summary(summary_path: Optional[str]) -> Dict[str, Any]:
    """Load optional summary.json metadata if present."""
    if not summary_path or not os.path.isfile(summary_path):
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_best_val_epoch(
    records: List[Dict[str, Any]],
    summary_path: Optional[str] = None,
    best_epoch: Optional[int] = None,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Resolve best validation epoch along with its DSC@0.5 and clDice@0.5."""
    summary = _load_summary(summary_path)

    resolved_ep: Optional[int] = None
    if best_epoch is not None and best_epoch >= 0:
        resolved_ep = int(best_epoch)
    elif summary.get("best_val_epoch") is not None and int(summary["best_val_epoch"]) >= 0:
        resolved_ep = int(summary["best_val_epoch"])
    else:
        val_records = [
            r for r in records if ("val_select_score" in r or "val_pooled_dice" in r)
        ]
        if val_records:
            best_rec = max(
                val_records,
                key=lambda r: float(
                    r.get("val_select_score", r.get("val_pooled_dice", float("-inf")))
                ),
            )
            resolved_ep = int(best_rec["epoch"])

    if resolved_ep is None:
        return None, None, None

    rec_by_ep = {int(r["epoch"]): r for r in records}
    best_rec = rec_by_ep.get(resolved_ep, {})

    best_dsc = best_rec.get("val_pooled_dice", summary.get("best_val_dice_05"))
    best_cldice = best_rec.get("val_mean_cldice", summary.get("best_val_cldice_05"))
    return (
        resolved_ep,
        float(best_dsc) if best_dsc is not None else None,
        float(best_cldice) if best_cldice is not None else None,
    )


def _extract_series(
    records: List[Dict[str, Any]], key: str
) -> Tuple[List[int], List[float]]:
    """Extract (epoch, value) pairs for records containing `key`."""
    epochs: List[int] = []
    values: List[float] = []
    for rec in records:
        val = rec.get(key)
        if val is not None:
            epochs.append(int(rec["epoch"]))
            values.append(float(val))
    return epochs, values


def plot_training_curves(
    task_dir: str,
    output_filename: str = "training_curves.png",
    best_epoch: Optional[int] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """Plot losses and validation DSC/clDice metrics from log.txt into a single dual-Y figure."""
    log_path = os.path.join(task_dir, "log.txt")
    records = load_training_log(log_path)
    if not records:
        return None

    summary_path = os.path.join(task_dir, "summary.json")
    summary = _load_summary(summary_path)
    best_ep, best_dsc, best_cldice = resolve_best_val_epoch(
        records, summary_path=summary_path, best_epoch=best_epoch
    )

    epochs_all = [int(r["epoch"]) for r in records]
    min_ep, max_ep = min(epochs_all), max(epochs_all)

    fig, ax_loss = plt.subplots(figsize=(12, 6.5), dpi=150)
    ax_metric = ax_loss.twinx()

    for key, label, style in LOSS_SERIES:
        xs, ys = _extract_series(records, key)
        if xs:
            ax_loss.plot(xs, ys, label=label, **style)

    for key, label, style in METRIC_SERIES:
        xs, ys = _extract_series(records, key)
        if xs:
            ax_metric.plot(xs, ys, label=label, **style)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_ylim(bottom=0.0)
    if min_ep == max_ep:
        ax_loss.set_xlim(min_ep - 0.5, max_ep + 0.5)
    else:
        ax_loss.set_xlim(min_ep, max_ep)
    ax_loss.grid(True, linestyle="--", alpha=0.3)

    ax_metric.set_ylabel("Metric (DSC / clDice)")
    ax_metric.set_ylim(0.0, 1.02)
    ax_metric.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    if best_ep is not None:
        ax_loss.axvline(
            best_ep,
            color="#4b5563",
            linestyle=":",
            linewidth=1.6,
            alpha=0.9,
            label=f"Best Epoch ({best_ep})",
        )
        dsc_txt = f"{best_dsc:.4f}" if best_dsc is not None else "N/A"
        cldice_txt = f"{best_cldice:.4f}" if best_cldice is not None else "N/A"
        annot_text = f"Best ep {best_ep}\nDSC: {dsc_txt}\nclDice: {cldice_txt}"
        place_left = max_ep > min_ep and best_ep >= (min_ep + 0.65 * (max_ep - min_ep))
        ax_metric.annotate(
            annot_text,
            xy=(best_ep, 0.96),
            xytext=(-8 if place_left else 8, 0),
            textcoords="offset points",
            ha="right" if place_left else "left",
            va="top",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#9ca3af",
                "alpha": 0.85,
            },
        )

    base_title = title or os.path.basename(os.path.abspath(task_dir))
    loss_cfg = summary.get("loss")
    select_cfg = summary.get("select_metric")
    if loss_cfg and select_cfg:
        full_title = f"{base_title}\nloss={loss_cfg} | select_metric={select_cfg}"
    else:
        full_title = base_title
    ax_loss.set_title(full_title, fontsize=11, pad=10)

    h_loss, l_loss = ax_loss.get_legend_handles_labels()
    h_metric, l_metric = ax_metric.get_legend_handles_labels()
    ax_loss.legend(
        h_loss + h_metric,
        l_loss + l_metric,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=True,
        fontsize=8.5,
    )

    out_path = os.path.join(task_dir, output_filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_existing_output_dirs(
    output_dir: str,
    prefix: str = "unified",
    output_filename: str = "training_curves.png",
) -> List[str]:
    """Generate training curve plots for all matching subdirectories containing log.txt."""
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    generated: List[str] = []
    for entry_name in sorted(os.listdir(output_dir)):
        if prefix and not entry_name.startswith(prefix):
            continue
        entry_path = os.path.join(output_dir, entry_name)
        if not os.path.isdir(entry_path):
            continue
        if not os.path.isfile(os.path.join(entry_path, "log.txt")):
            continue
        out_path = plot_training_curves(entry_path, output_filename=output_filename)
        if out_path is not None:
            generated.append(out_path)
            print(f"Generated {out_path}")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot segmentation training loss and metric curves from log.txt"
    )
    parser.add_argument("--output_dir", default="./output_dir", type=str)
    parser.add_argument("--prefix", default="unified", type=str)
    parser.add_argument("--task_dir", default="", type=str)
    parser.add_argument("--filename", default="training_curves.png", type=str)
    args = parser.parse_args()

    if args.task_dir:
        out_path = plot_training_curves(args.task_dir, output_filename=args.filename)
        if out_path is not None:
            print(f"Generated {out_path}")
        else:
            print(f"No log records found in {args.task_dir}")
    else:
        paths = plot_existing_output_dirs(
            output_dir=args.output_dir,
            prefix=args.prefix,
            output_filename=args.filename,
        )
        print(f"Total plots generated: {len(paths)}")


if __name__ == "__main__":
    main()
