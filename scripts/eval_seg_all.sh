#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Dedicated retfound conda environment Python binary (per AGENTS.md / GEMINI.md rules)
PYTHON="/workspace/miniconda3/envs/retfound/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Unified RETFound 3-Dataset 4x-Flip TTA Eval-Only Benchmark:
#   Evaluates checkpoint-best.pth from scripts/train_seg_all.sh runs across:
#     DRIVE (592), STARE (704), CHASE-DB1 (1008) x seeds {0, 1, 2}
#   Outputs are written to separate *_tta task directories to preserve non-TTA results.
MODEL="RETFound_mae"
MAE_CKPT="${MAE_CKPT:-RETFound_mae_natureCFP}"
ADAPTATION="${ADAPTATION:-frozen}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
DROP_PATH="${DROP_PATH:-0.05}"

CROPS_PER_IMAGE="${CROPS_PER_IMAGE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

LOSS_STR="bce,dice,cldice"
BCE_WEIGHT="${BCE_WEIGHT:-1.0}"
DICE_WEIGHT="${DICE_WEIGHT:-1.0}"
CLDICE_WEIGHT="${CLDICE_WEIGHT:-0.3}"
SKELREC_WEIGHT="${SKELREC_WEIGHT:-0.0}"
CLDICE_ITER="${CLDICE_ITER:-10}"
TOPO_WARMUP_EPOCHS="${TOPO_WARMUP_EPOCHS:-10}"
POS_WEIGHT="${POS_WEIGHT:-0.0}"
SELECT_METRIC="dice_cldice"

DRIVE_PATH="${DRIVE_PATH:-./data/drive_aug}"
DRIVE_CROP=592

STARE_PATH="${STARE_PATH:-./data/stare_aug}"
STARE_CROP=704

CHASE_PATH="${CHASE_PATH:-./data/chase_aug}"
CHASE_CROP=1008

run_eval() {
  local seed="$1"
  local dataset="$2"
  local data_path="$3"
  local crop_size="$4"

  local train_task="unified_retfound_mae_${ADAPTATION}_enh_cldice_${crop_size}_${dataset}_seed${seed}"
  local ckpt_path="./output_dir/${train_task}/checkpoint-best.pth"
  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[ERROR] Missing checkpoint: ${ckpt_path}" >&2
    return 1
  fi

  local eval_task="${train_task}_tta"
  local out_dir="./output_dir/${eval_task}"
  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/summary.json" ]]; then
    echo "[SKIP] ${eval_task} already evaluated."
    return 0
  fi

  echo "[START] ${eval_task} (resume=${ckpt_path}, crop=${crop_size}, enhance=true, tta=true) at $(date -u +%H:%M:%S)"
  "${PYTHON}" main_segment.py \
    --eval \
    --tta \
    --enhance \
    --resume "${ckpt_path}" \
    --dataset "${dataset}" \
    --data_path "${data_path}" \
    --model "${MODEL}" \
    --finetune "${MAE_CKPT}" \
    --adaptation "${ADAPTATION}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --drop_path "${DROP_PATH}" \
    --crop_size "${crop_size}" \
    --crops_per_image "${CROPS_PER_IMAGE}" \
    --num_workers "${NUM_WORKERS}" \
    --loss "${LOSS_STR}" \
    --bce_weight "${BCE_WEIGHT}" \
    --dice_weight "${DICE_WEIGHT}" \
    --cldice_weight "${CLDICE_WEIGHT}" \
    --skelrec_weight "${SKELREC_WEIGHT}" \
    --cldice_iter "${CLDICE_ITER}" \
    --topo_warmup_epochs "${TOPO_WARMUP_EPOCHS}" \
    --pos_weight "${POS_WEIGHT}" \
    --select_metric "${SELECT_METRIC}" \
    --seed "${seed}" \
    --task "${eval_task}" > "${out_dir}/stdout.log" 2>&1
  echo "[DONE]  ${eval_task} at $(date -u +%H:%M:%S)"
}

run_eval_batch() {
  local dataset="$1"
  local data_path="$2"
  local crop_size="$3"

  echo "=================================================================="
  echo "Evaluating 3-Seed TTA Batch [${dataset}] (crop=${crop_size}, enhance=true, tta=4x)"
  echo "=================================================================="

  run_eval 0 "${dataset}" "${data_path}" "${crop_size}" &
  local pid_s0=$!

  run_eval 1 "${dataset}" "${data_path}" "${crop_size}" &
  local pid_s1=$!

  run_eval 2 "${dataset}" "${data_path}" "${crop_size}" &
  local pid_s2=$!

  local failed=0
  wait "${pid_s0}" || failed=1
  wait "${pid_s1}" || failed=1
  wait "${pid_s2}" || failed=1
  if [[ "${failed}" -ne 0 ]]; then
    echo "[ERROR] One or more TTA eval runs failed for dataset '${dataset}'."
    return 1
  fi
  echo "[BATCH DONE] ${dataset} TTA eval completed at $(date -u +%H:%M:%S)"
}

echo "=================================================================="
echo "Starting 3-Dataset 4x-Flip TTA Eval-Only Benchmark (DRIVE / STARE / CHASE-DB1)"
echo "  Adaptation: ${ADAPTATION} | Enhance: true (CLAHE) | Loss: ${LOSS_STR} | Select: ${SELECT_METRIC}"
echo "=================================================================="

run_eval_batch "drive" "${DRIVE_PATH}" "${DRIVE_CROP}"
run_eval_batch "stare" "${STARE_PATH}" "${STARE_CROP}"
run_eval_batch "chase" "${CHASE_PATH}" "${CHASE_CROP}"

echo "=================================================================="
echo "All 3 datasets x 3 seeds TTA eval completed! Aggregating 9-metric Mean ± Std..."
echo "=================================================================="

ADAPTATION="${ADAPTATION}" "${PYTHON}" -c "
import glob, json, os, numpy as np

adapt = os.environ['ADAPTATION']
datasets = [
    ('drive', 592),
    ('stare', 704),
    ('chase', 1008),
]

keys = [
    'pooled_dice',
    'dice_at_best_tau',
    'mean_cldice',
    'pooled_jaccard',
    'pooled_auc',
    'pooled_se',
    'pooled_sp',
    'pooled_acc',
    'pooled_mcc',
]

summary_all = {}
for ds, cs in datasets:
    pat = f'unified_retfound_mae_{adapt}_enh_cldice_{cs}_{ds}_seed[0-9]*_tta'
    files = sorted(glob.glob(f'./output_dir/{pat}/summary.json'))
    assert len(files) == 3, f'Expected 3 seed summary.json files for {ds} ({cs}), found {len(files)}: {files}'
    rows = [json.load(open(f))['test'] for f in files]
    label = f'Unified RETFound | {ds.upper()} ({cs}) | {adapt} | enhance | bce,dice,cldice | select=dice_cldice | TTA'
    summary_all[label] = {
        k: {'mean': float(np.mean([r[k] for r in rows])), 'std': float(np.std([r[k] for r in rows]))}
        for k in keys
    }
    m = summary_all[label]
    print(
        f'{label} (n={len(rows)}) | '
        f'Dice@0.5: {m[\"pooled_dice\"][\"mean\"]:.4f}±{m[\"pooled_dice\"][\"std\"]:.4f} | '
        f'Dice@tau*: {m[\"dice_at_best_tau\"][\"mean\"]:.4f}±{m[\"dice_at_best_tau\"][\"std\"]:.4f} | '
        f'clDice: {m[\"mean_cldice\"][\"mean\"]:.4f}±{m[\"mean_cldice\"][\"std\"]:.4f} | '
        f'IoU: {m[\"pooled_jaccard\"][\"mean\"]:.4f}±{m[\"pooled_jaccard\"][\"std\"]:.4f} | '
        f'AUC: {m[\"pooled_auc\"][\"mean\"]:.4f}±{m[\"pooled_auc\"][\"std\"]:.4f} | '
        f'Se: {m[\"pooled_se\"][\"mean\"]:.4f}±{m[\"pooled_se\"][\"std\"]:.4f} | '
        f'Sp: {m[\"pooled_sp\"][\"mean\"]:.4f}±{m[\"pooled_sp\"][\"std\"]:.4f} | '
        f'Acc: {m[\"pooled_acc\"][\"mean\"]:.4f}±{m[\"pooled_acc\"][\"std\"]:.4f} | '
        f'MCC: {m[\"pooled_mcc\"][\"mean\"]:.4f}±{m[\"pooled_mcc\"][\"std\"]:.4f}'
    )

out_json = f'./output_dir/unified_retfound_{adapt}_enh_cldice_all_datasets_tta_summary.json'
with open(out_json, 'w') as f:
    json.dump(summary_all, f, indent=2)
print(f'Saved aggregated 9-metric 3-dataset TTA summary to {out_json}')
"
