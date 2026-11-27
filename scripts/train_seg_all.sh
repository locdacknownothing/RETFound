#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Dedicated retfound conda environment Python binary (per AGENTS.md / GEMINI.md rules)
PYTHON="/workspace/miniconda3/envs/retfound/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Unified RETFound 3-Dataset Multi-Seed Benchmark:
#   Datasets: DRIVE (592, bs=4, accum=1), STARE (704, bs=4, accum=1), CHASE-DB1 (1008, bs=2, accum=2)
#   Config: model=RETFound_mae, adaptation=frozen, enhance=true (CLAHE), loss=bce,dice,cldice, select_metric=dice_cldice
MODEL="RETFound_mae"
MAE_CKPT="${MAE_CKPT:-RETFound_mae_natureCFP}"
ADAPTATION="${ADAPTATION:-frozen}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
DROP_PATH="${DROP_PATH:-0.05}"

EPOCHS="${EPOCHS:-200}"
CROPS_PER_IMAGE="${CROPS_PER_IMAGE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

LR_DECODER="${LR_DECODER:-3e-4}"
LR_LORA="${LR_LORA:-1.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.08}"

LOSS_STR="bce,dice,cldice"
BCE_WEIGHT="${BCE_WEIGHT:-1.0}"
DICE_WEIGHT="${DICE_WEIGHT:-1.0}"
CLDICE_WEIGHT="${CLDICE_WEIGHT:-0.3}"
SKELREC_WEIGHT="${SKELREC_WEIGHT:-0.0}"
CLDICE_ITER="${CLDICE_ITER:-10}"
TOPO_WARMUP_EPOCHS="${TOPO_WARMUP_EPOCHS:-10}"
POS_WEIGHT="${POS_WEIGHT:-0.0}"
SELECT_METRIC="dice_cldice"
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-true}"

# Dataset paths, crop sizes, and per-dataset micro-batch / grad-accum (effective batch = 4)
DRIVE_PATH="${DRIVE_PATH:-./data/drive_aug}"
DRIVE_CROP=592
DRIVE_BS=4
DRIVE_ACCUM=1

STARE_PATH="${STARE_PATH:-./data/stare_aug}"
STARE_CROP=704
STARE_BS=4
STARE_ACCUM=1

CHASE_PATH="${CHASE_PATH:-./data/chase_aug}"
CHASE_CROP=1008
CHASE_BS=2
CHASE_ACCUM=2

run_exp() {
  local seed="$1"
  local dataset="$2"
  local data_path="$3"
  local crop_size="$4"
  local batch_size="$5"
  local accum_iter="$6"

  local task="unified_retfound_mae_${ADAPTATION}_enh_cldice_${crop_size}_${dataset}_seed${seed}"
  local out_dir="./output_dir/${task}"
  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/summary.json" ]]; then
    echo "[SKIP] ${task} already completed."
    return 0
  fi

  local -a extra_args=(--enhance)
  if [[ "${GRAD_CHECKPOINTING}" == "true" ]]; then
    extra_args+=(--grad_checkpointing)
  fi

  echo "[START] ${task} (data=${data_path}, crop=${crop_size}, bs=${batch_size}x${accum_iter}, enhance=true, loss=${LOSS_STR}, select=${SELECT_METRIC}) at $(date -u +%H:%M:%S)"
  "${PYTHON}" main_segment.py \
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
    --batch_size "${batch_size}" \
    --accum_iter "${accum_iter}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --lr_decoder "${LR_DECODER}" \
    --lr_lora "${LR_LORA}" \
    --weight_decay "${WEIGHT_DECAY}" \
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
    --task "${task}" \
    ${extra_args[@]+"${extra_args[@]}"} > "${out_dir}/stdout.log" 2>&1
  echo "[DONE]  ${task} at $(date -u +%H:%M:%S)"
}

run_pair() {
  local wave_name="$1"
  local s_a="$2" ds_a="$3" path_a="$4" cs_a="$5" bs_a="$6" acc_a="$7"
  local s_b="$8" ds_b="$9" path_b="${10}" cs_b="${11}" bs_b="${12}" acc_b="${13}"

  echo "=================================================================="
  echo "Launching 2-Slot Wave [${wave_name}] (epochs=${EPOCHS}, grad_ckpt=${GRAD_CHECKPOINTING})"
  echo "  Slot A: ${ds_a} seed${s_a} | crop=${cs_a} | bs=${bs_a} x accum=${acc_a} | path=${path_a}"
  echo "  Slot B: ${ds_b} seed${s_b} | crop=${cs_b} | bs=${bs_b} x accum=${acc_b} | path=${path_b}"
  echo "=================================================================="

  run_exp "${s_a}" "${ds_a}" "${path_a}" "${cs_a}" "${bs_a}" "${acc_a}" &
  local pid_a=$!

  run_exp "${s_b}" "${ds_b}" "${path_b}" "${cs_b}" "${bs_b}" "${acc_b}" &
  local pid_b=$!

  local failed=0
  wait "${pid_a}" || failed=1
  wait "${pid_b}" || failed=1
  if [[ "${failed}" -ne 0 ]]; then
    echo "[ERROR] One or more runs failed in wave '${wave_name}'."
    return 1
  fi
  echo "[WAVE DONE] ${wave_name} completed at $(date -u +%H:%M:%S)"
}

echo "=================================================================="
echo "Starting Unified RETFound 3-Dataset Training (DRIVE / STARE / CHASE-DB1)"
echo "  Adaptation: ${ADAPTATION} | Enhance: true (CLAHE) | Loss: ${LOSS_STR} | Select: ${SELECT_METRIC}"
echo "  Effective Batch Size: 4 | Grad Checkpointing: ${GRAD_CHECKPOINTING}"
echo "=================================================================="

# DRIVE (auto-skips if unified_retfound_mae_frozen_enh_cldice_592_drive_seed{0,1,2}/summary.json exist)
run_pair "DRIVE (seeds 0,1)" \
  0 "drive" "${DRIVE_PATH}" "${DRIVE_CROP}" "${DRIVE_BS}" "${DRIVE_ACCUM}" \
  1 "drive" "${DRIVE_PATH}" "${DRIVE_CROP}" "${DRIVE_BS}" "${DRIVE_ACCUM}"
run_exp 2 "drive" "${DRIVE_PATH}" "${DRIVE_CROP}" "${DRIVE_BS}" "${DRIVE_ACCUM}"

# 6 new runs (CHASE-DB1 + STARE) across 3 balanced 2-concurrent waves
run_pair "Wave 1 (CHASE seeds 0,1)" \
  0 "chase" "${CHASE_PATH}" "${CHASE_CROP}" "${CHASE_BS}" "${CHASE_ACCUM}" \
  1 "chase" "${CHASE_PATH}" "${CHASE_CROP}" "${CHASE_BS}" "${CHASE_ACCUM}"

run_pair "Wave 2 (CHASE seed 2 + STARE seed 0)" \
  2 "chase" "${CHASE_PATH}" "${CHASE_CROP}" "${CHASE_BS}" "${CHASE_ACCUM}" \
  0 "stare" "${STARE_PATH}" "${STARE_CROP}" "${STARE_BS}" "${STARE_ACCUM}"

run_pair "Wave 3 (STARE seeds 1,2)" \
  1 "stare" "${STARE_PATH}" "${STARE_CROP}" "${STARE_BS}" "${STARE_ACCUM}" \
  2 "stare" "${STARE_PATH}" "${STARE_CROP}" "${STARE_BS}" "${STARE_ACCUM}"

echo "=================================================================="
echo "All 3 datasets x 3 seeds completed! Aggregating 9-metric Mean ± Std..."
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
    pat = f'unified_retfound_mae_{adapt}_enh_cldice_{cs}_{ds}_seed[0-9]*'
    files = sorted(glob.glob(f'./output_dir/{pat}/summary.json'))
    assert len(files) == 3, f'Expected 3 seed summary.json files for {ds} ({cs}), found {len(files)}: {files}'
    rows = [json.load(open(f))['test'] for f in files]
    label = f'Unified RETFound | {ds.upper()} ({cs}) | {adapt} | enhance | bce,dice,cldice | select=dice_cldice'
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

out_json = f'./output_dir/unified_retfound_{adapt}_enh_cldice_all_datasets_summary.json'
with open(out_json, 'w') as f:
    json.dump(summary_all, f, indent=2)
print(f'Saved aggregated 9-metric 3-dataset summary to {out_json}')
"
