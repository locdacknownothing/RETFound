#!/usr/bin/env bash
set -euo pipefail

# Dedicated retfound conda environment Python binary (per AGENTS.md / GEMINI.md rules)
PYTHON="/workspace/miniconda3/envs/retfound/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Fixed experimental grid (12 runs total = Phase A/B x enhance {false,true} x seeds {0,1,2}):
#   dataset=drive, model=RETFound_mae, adaptation=frozen, crop_size=592, crops_per_image=1
# Uses distinct *592* task names so previous 448x448 Phase A outputs (E0_*, E1_*, E2_*) are untouched.
DATASET="drive"
DATA_PATH="./data/drive_aug"
MODEL="RETFound_mae"
MAE_CKPT="${MAE_CKPT:-RETFound_mae_natureCFP}"
ADAPTATION="frozen"

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CROPS_PER_IMAGE="${CROPS_PER_IMAGE:-1}"
CROP_SIZE="${CROP_SIZE:-592}"
NUM_WORKERS="${NUM_WORKERS:-2}"

BCE_WEIGHT="${BCE_WEIGHT:-1.0}"
DICE_WEIGHT="${DICE_WEIGHT:-1.0}"
CLDICE_WEIGHT="${CLDICE_WEIGHT:-0.3}"
SKELREC_WEIGHT="${SKELREC_WEIGHT:-0.2}"
CLDICE_ITER="${CLDICE_ITER:-10}"
TOPO_WARMUP_EPOCHS="${TOPO_WARMUP_EPOCHS:-10}"
SELECT_METRIC="${SELECT_METRIC:-dice_cldice}"

run_exp() {
  local phase="$1"      # phaseA | phaseB
  local enhance="$2"    # raw | enh
  local loss_str="$3"   # "bce,dice" | "bce,dice,cldice,skelrec"
  local seed="$4"

  local task="${phase}_retfound_mae_frozen_${enhance}_${CROP_SIZE}_${DATASET}_seed${seed}"
  local out_dir="./output_dir/${task}"
  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/summary.json" ]]; then
    echo "[SKIP] ${task} already completed."
    return 0
  fi

  local -a extra_flags=()
  if [[ "${enhance}" == "enh" ]]; then
    extra_flags+=(--enhance)
  fi

  echo "[START] ${task} (loss=${loss_str}, enhance=${enhance}) at $(date -u +%H:%M:%S)"
  "${PYTHON}" main_segment.py \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --model "${MODEL}" \
    --finetune "${MAE_CKPT}" \
    --adaptation "${ADAPTATION}" \
    --crop_size "${CROP_SIZE}" \
    --crops_per_image "${CROPS_PER_IMAGE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --loss "${loss_str}" \
    --bce_weight "${BCE_WEIGHT}" \
    --dice_weight "${DICE_WEIGHT}" \
    --cldice_weight "${CLDICE_WEIGHT}" \
    --skelrec_weight "${SKELREC_WEIGHT}" \
    --cldice_iter "${CLDICE_ITER}" \
    --topo_warmup_epochs "${TOPO_WARMUP_EPOCHS}" \
    --select_metric "${SELECT_METRIC}" \
    --seed "${seed}" \
    --task "${task}" \
    "${extra_flags[@]}" > "${out_dir}/stdout.log" 2>&1
  echo "[DONE]  ${task} at $(date -u +%H:%M:%S)"
}

echo "=================================================================="
echo "Launching 12 Phase A/B Runs (RETFound_mae frozen, 592x592) in 3 Seed Streams"
echo "  Stream 0 (Seed 0): phaseA(raw) -> phaseA(enh) -> phaseB(raw) -> phaseB(enh)"
echo "  Stream 1 (Seed 1): phaseA(raw) -> phaseA(enh) -> phaseB(raw) -> phaseB(enh)"
echo "  Stream 2 (Seed 2): phaseA(raw) -> phaseA(enh) -> phaseB(raw) -> phaseB(enh)"
echo "=================================================================="

run_stream() {
  local seed="$1"
  run_exp "phaseA" "raw" "bce,dice" "${seed}"
  run_exp "phaseA" "enh" "bce,dice" "${seed}"
  run_exp "phaseB" "raw" "bce,dice,cldice,skelrec" "${seed}"
  run_exp "phaseB" "enh" "bce,dice,cldice,skelrec" "${seed}"
}

run_stream 0 &
PID_S0=$!

run_stream 1 &
PID_S1=$!

run_stream 2 &
PID_S2=$!

wait "${PID_S0}" "${PID_S1}" "${PID_S2}"

echo "=================================================================="
echo "All 12 Phase A/B runs completed! Aggregating 3-seed results..."
echo "=================================================================="

CROP_SIZE="${CROP_SIZE}" DATASET="${DATASET}" "${PYTHON}" -c "
import glob, json, os, numpy as np

cs = os.environ['CROP_SIZE']
ds = os.environ['DATASET']
exps = [
    ('Phase A | raw | bce,dice', f'phaseA_retfound_mae_frozen_raw_{cs}_{ds}_seed*'),
    ('Phase A | enh | bce,dice', f'phaseA_retfound_mae_frozen_enh_{cs}_{ds}_seed*'),
    ('Phase B | raw | +cldice,skelrec', f'phaseB_retfound_mae_frozen_raw_{cs}_{ds}_seed*'),
    ('Phase B | enh | +cldice,skelrec', f'phaseB_retfound_mae_frozen_enh_{cs}_{ds}_seed*'),
]
keys = ['pooled_dice', 'mean_cldice', 'pooled_jaccard', 'pooled_auc', 'pooled_se', 'pooled_sp', 'pooled_mcc', 'dice_at_best_tau']
summary_all = {}
for label, pat in exps:
    files = sorted(glob.glob(f'./output_dir/{pat}/summary.json'))
    rows = [json.load(open(f))['test'] for f in files]
    summary_all[label] = {
        k: {'mean': float(np.mean([r[k] for r in rows])), 'std': float(np.std([r[k] for r in rows]))}
        for k in keys
    }
    m = summary_all[label]
    print(f'{label:34s} (n={len(rows)}) | '
          f'Dice: {m[\"pooled_dice\"][\"mean\"]:.4f}±{m[\"pooled_dice\"][\"std\"]:.4f} | '
          f'clDice: {m[\"mean_cldice\"][\"mean\"]:.4f}±{m[\"mean_cldice\"][\"std\"]:.4f} | '
          f'IoU: {m[\"pooled_jaccard\"][\"mean\"]:.4f}±{m[\"pooled_jaccard\"][\"std\"]:.4f} | '
          f'AUC: {m[\"pooled_auc\"][\"mean\"]:.4f}±{m[\"pooled_auc\"][\"std\"]:.4f} | '
          f'Se: {m[\"pooled_se\"][\"mean\"]:.4f}±{m[\"pooled_se\"][\"std\"]:.4f} | '
          f'Sp: {m[\"pooled_sp\"][\"mean\"]:.4f}±{m[\"pooled_sp\"][\"std\"]:.4f} | '
          f'MCC: {m[\"pooled_mcc\"][\"mean\"]:.4f}±{m[\"pooled_mcc\"][\"std\"]:.4f}')

with open(f'./output_dir/phaseAB_{ds}_{cs}_summary.json', 'w') as f:
    json.dump(summary_all, f, indent=2)
"
