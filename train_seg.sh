#!/usr/bin/env bash
set -euo pipefail

# Dedicated retfound conda environment Python binary (per AGENTS.md / GEMINI.md rules)
PYTHON="/workspace/miniconda3/envs/retfound/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

DATASET="drive"
DATA_PATH="./data/drive_aug"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CROPS_PER_IMAGE="${CROPS_PER_IMAGE:-2}"
CROP_SIZE="${CROP_SIZE:-448}"
NUM_WORKERS="${NUM_WORKERS:-2}"

# Pre-trained RETFound_mae checkpoint on HuggingFace Hub
MAE_CKPT="${MAE_CKPT:-RETFound_mae_natureCFP}"

run_e0() {
  local seed="$1"
  local task="E0_unet_${DATASET}_seed${seed}"
  local out_dir="./output_dir/${task}"
  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/summary.json" ]]; then
    echo "[SKIP] ${task} already completed."
    return 0
  fi
  echo "[START] ${task} at $(date -u +%H:%M:%S)"
  "${PYTHON}" main_segment.py \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --model unet \
    --crop_size "${CROP_SIZE}" \
    --crops_per_image "${CROPS_PER_IMAGE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${seed}" \
    --task "${task}" > "${out_dir}/stdout.log" 2>&1
  echo "[DONE]  ${task} at $(date -u +%H:%M:%S)"
}

run_e1() {
  local seed="$1"
  local task="E1_retfound_mae_frozen_${DATASET}_seed${seed}"
  local out_dir="./output_dir/${task}"
  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/summary.json" ]]; then
    echo "[SKIP] ${task} already completed."
    return 0
  fi
  echo "[START] ${task} at $(date -u +%H:%M:%S)"
  "${PYTHON}" main_segment.py \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --model RETFound_mae \
    --finetune "${MAE_CKPT}" \
    --adaptation frozen \
    --crop_size "${CROP_SIZE}" \
    --crops_per_image "${CROPS_PER_IMAGE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${seed}" \
    --task "${task}" > "${out_dir}/stdout.log" 2>&1
  echo "[DONE]  ${task} at $(date -u +%H:%M:%S)"
}

run_e2() {
  local seed="$1"
  local task="E2_retfound_mae_lora_${DATASET}_seed${seed}"
  local out_dir="./output_dir/${task}"
  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/summary.json" ]]; then
    echo "[SKIP] ${task} already completed."
    return 0
  fi
  echo "[START] ${task} at $(date -u +%H:%M:%S)"
  "${PYTHON}" main_segment.py \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --model RETFound_mae \
    --finetune "${MAE_CKPT}" \
    --adaptation lora \
    --lora_rank 16 \
    --grad_checkpointing \
    --crop_size "${CROP_SIZE}" \
    --crops_per_image "${CROPS_PER_IMAGE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${seed}" \
    --task "${task}" > "${out_dir}/stdout.log" 2>&1
  echo "[DONE]  ${task} at $(date -u +%H:%M:%S)"
}

echo "=================================================================="
echo "Launching 3 Cyclic Parallel Training Streams on GPU (~15-18 GB total VRAM)"
echo "  Stream 0 (Seed 0): E0_s0 -> E1_s0 -> E2_s0"
echo "  Stream 1 (Seed 1): E1_s1 -> E2_s1 -> E0_s1"
echo "  Stream 2 (Seed 2): E2_s2 -> E0_s2 -> E1_s2"
echo "=================================================================="

( run_e0 0 && run_e1 0 && run_e2 0 ) &
PID_S0=$!

( run_e1 1 && run_e2 1 && run_e0 1 ) &
PID_S1=$!

( run_e2 2 && run_e0 2 && run_e1 2 ) &
PID_S2=$!

wait "${PID_S0}" "${PID_S1}" "${PID_S2}"

echo "=================================================================="
echo "All 9 Phase A runs completed! Aggregating 3-seed results..."
echo "=================================================================="

"${PYTHON}" -c "
import glob, json, numpy as np

exps = [
    ('E0: CNN-only U-Net', 'E0_unet_drive_seed*'),
    ('E1: RETFound_mae (Frozen)', 'E1_retfound_mae_frozen_drive_seed*'),
    ('E2: RETFound_mae (LoRA r=16)', 'E2_retfound_mae_lora_drive_seed*'),
]
keys = ['pooled_dice', 'pooled_jaccard', 'pooled_auc', 'pooled_se', 'pooled_sp', 'pooled_acc', 'pooled_mcc', 'dice_at_best_tau']
summary_all = {}
for label, pat in exps:
    files = sorted(glob.glob(f'./output_dir/{pat}/summary.json'))
    rows = [json.load(open(f))['test'] for f in files]
    summary_all[label] = {
        k: {'mean': float(np.mean([r[k] for r in rows])), 'std': float(np.std([r[k] for r in rows]))}
        for k in keys
    }
    m = summary_all[label]
    print(f'{label:28s} (n={len(rows)}) | '
          f'Dice: {m[\"pooled_dice\"][\"mean\"]:.4f}±{m[\"pooled_dice\"][\"std\"]:.4f} | '
          f'IoU: {m[\"pooled_jaccard\"][\"mean\"]:.4f}±{m[\"pooled_jaccard\"][\"std\"]:.4f} | '
          f'AUC: {m[\"pooled_auc\"][\"mean\"]:.4f}±{m[\"pooled_auc\"][\"std\"]:.4f} | '
          f'Se: {m[\"pooled_se\"][\"mean\"]:.4f}±{m[\"pooled_se\"][\"std\"]:.4f} | '
          f'Sp: {m[\"pooled_sp\"][\"mean\"]:.4f}±{m[\"pooled_sp\"][\"std\"]:.4f} | '
          f'MCC: {m[\"pooled_mcc\"][\"mean\"]:.4f}±{m[\"pooled_mcc\"][\"std\"]:.4f}')

with open('./output_dir/phaseA_drive_summary.json', 'w') as f:
    json.dump(summary_all, f, indent=2)
"
