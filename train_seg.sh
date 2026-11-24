#!/usr/bin/env bash
set -euo pipefail

# Dedicated retfound conda environment Python binary (per AGENTS.md / GEMINI.md rules)
PYTHON="/workspace/miniconda3/envs/retfound/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Unified RETFound DRIVE 3-seed benchmark:
#   dataset=drive, model=RETFound_mae, adaptation=lora (r=8 + unfrozen patch_embed/PatchEmbedConvStem + multi-scale adapters + MaskTransformerDecoder + 4-level upsampling CNN decoder)
#   crop_size=592, crops_per_image=1, batch_size=4, epochs=200, raw preprocessing, loss=bce,dice, select_metric=dice
DATASET="drive"
DATA_PATH="./data/drive_aug"
MODEL="RETFound_mae"
MAE_CKPT="${MAE_CKPT:-RETFound_mae_natureCFP}"
ADAPTATION="${ADAPTATION:-lora}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
DROP_PATH="${DROP_PATH:-0.05}"

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CROPS_PER_IMAGE="${CROPS_PER_IMAGE:-1}"
CROP_SIZE="${CROP_SIZE:-592}"
NUM_WORKERS="${NUM_WORKERS:-2}"

LR_DECODER="${LR_DECODER:-3e-4}"
LR_LORA="${LR_LORA:-1.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.08}"

LOSS_STR="${LOSS_STR:-bce,dice}"
BCE_WEIGHT="${BCE_WEIGHT:-1.0}"
DICE_WEIGHT="${DICE_WEIGHT:-1.0}"
CLDICE_WEIGHT="${CLDICE_WEIGHT:-0.3}"
SKELREC_WEIGHT="${SKELREC_WEIGHT:-0.2}"
CLDICE_ITER="${CLDICE_ITER:-10}"
TOPO_WARMUP_EPOCHS="${TOPO_WARMUP_EPOCHS:-10}"
POS_WEIGHT="${POS_WEIGHT:-1.4}"
SELECT_METRIC="${SELECT_METRIC:-dice}"

run_exp() {
  local seed="$1"
  local task="unified_retfound_mae_raw_${CROP_SIZE}_${DATASET}_seed${seed}"
  local out_dir="./output_dir/${task}"
  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/summary.json" ]]; then
    echo "[SKIP] ${task} already completed."
    return 0
  fi

  echo "[START] ${task} (loss=${LOSS_STR}, select_metric=${SELECT_METRIC}, raw) at $(date -u +%H:%M:%S)"
  "${PYTHON}" main_segment.py \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --model "${MODEL}" \
    --finetune "${MAE_CKPT}" \
    --adaptation "${ADAPTATION}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --drop_path "${DROP_PATH}" \
    --crop_size "${CROP_SIZE}" \
    --crops_per_image "${CROPS_PER_IMAGE}" \
    --batch_size "${BATCH_SIZE}" \
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
    --task "${task}" > "${out_dir}/stdout.log" 2>&1
  echo "[DONE]  ${task} at $(date -u +%H:%M:%S)"
}

echo "=================================================================="
echo "Launching 3-Seed Unified RETFound DRIVE Benchmark (592x592, 200 epochs)"
echo "  Model: ${MODEL} | Adaptation: ${ADAPTATION} | Preprocessing: raw"
echo "  Loss: ${LOSS_STR} | Select Metric: ${SELECT_METRIC} | Seeds: 0, 1, 2"
echo "=================================================================="

run_exp 0 &
PID_S0=$!

run_exp 1 &
PID_S1=$!

run_exp 2 &
PID_S2=$!

wait "${PID_S0}" "${PID_S1}" "${PID_S2}"

echo "=================================================================="
echo "All 3 seeds completed! Aggregating 9-metric Mean ± Std results..."
echo "=================================================================="

CROP_SIZE="${CROP_SIZE}" DATASET="${DATASET}" "${PYTHON}" -c "
import glob, json, os, numpy as np

cs = os.environ['CROP_SIZE']
ds = os.environ['DATASET']
pat = f'unified_retfound_mae_raw_{cs}_{ds}_seed*'
files = sorted(glob.glob(f'./output_dir/{pat}/summary.json'))
assert len(files) == 3, f'Expected 3 seed summary.json files, found {len(files)}: {files}'
rows = [json.load(open(f))['test'] for f in files]

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
label = 'Unified RETFound | raw | bce,dice | select=dice'
summary_all = {
    label: {
        k: {'mean': float(np.mean([r[k] for r in rows])), 'std': float(np.std([r[k] for r in rows]))}
        for k in keys
    }
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

out_json = f'./output_dir/unified_retfound_{ds}_{cs}_summary.json'
with open(out_json, 'w') as f:
    json.dump(summary_all, f, indent=2)
print(f'Saved aggregated 9-metric summary to {out_json}')
"
