#!/usr/bin/env bash
set -euo pipefail

# Dedicated retfound conda environment Python binary (per AGENTS.md / GEMINI.md rules)
PYTHON="/workspace/miniconda3/envs/retfound/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Unified RETFound DRIVE 2x3 topology-loss ablation benchmark (adaptation=frozen, enhance=true):
#   loss in {("bce,dice,cldice,skelrec" [fixed exact skeleton], "dice_cldice"), ("bce,dice,cldice", "dice_cldice")} x seeds {0, 1, 2}
#   crop_size=592, crops_per_image=1, batch_size=4, epochs=200
DATASET="drive"
DATA_PATH="./data/drive_aug"
MODEL="RETFound_mae"
MAE_CKPT="${MAE_CKPT:-RETFound_mae_natureCFP}"
ADAPTATION="${ADAPTATION:-frozen}"
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

BCE_WEIGHT="${BCE_WEIGHT:-1.0}"
DICE_WEIGHT="${DICE_WEIGHT:-1.0}"
CLDICE_WEIGHT="${CLDICE_WEIGHT:-0.3}"
SKELREC_WEIGHT="${SKELREC_WEIGHT:-0.2}"
CLDICE_ITER="${CLDICE_ITER:-10}"
TOPO_WARMUP_EPOCHS="${TOPO_WARMUP_EPOCHS:-10}"
POS_WEIGHT="${POS_WEIGHT:-0.0}"
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-true}"

run_exp() {
  local seed="$1"
  local enhance="$2"
  local loss_str="$3"
  local select_metric="$4"
  local tag="$5"

  local task="unified_retfound_mae_${ADAPTATION}_${tag}_${CROP_SIZE}_${DATASET}_seed${seed}"
  local out_dir="./output_dir/${task}"
  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/summary.json" ]]; then
    echo "[SKIP] ${task} already completed."
    return 0
  fi

  local -a extra_args=()
  if [[ "${enhance}" == "true" ]]; then
    extra_args+=(--enhance)
  fi
  if [[ "${GRAD_CHECKPOINTING}" == "true" ]]; then
    extra_args+=(--grad_checkpointing)
  fi

  echo "[START] ${task} (enhance=${enhance}, loss=${loss_str}, select_metric=${select_metric}) at $(date -u +%H:%M:%S)"
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
    --loss "${loss_str}" \
    --bce_weight "${BCE_WEIGHT}" \
    --dice_weight "${DICE_WEIGHT}" \
    --cldice_weight "${CLDICE_WEIGHT}" \
    --skelrec_weight "${SKELREC_WEIGHT}" \
    --cldice_iter "${CLDICE_ITER}" \
    --topo_warmup_epochs "${TOPO_WARMUP_EPOCHS}" \
    --pos_weight "${POS_WEIGHT}" \
    --select_metric "${select_metric}" \
    --seed "${seed}" \
    --task "${task}" \
    ${extra_args[@]+"${extra_args[@]}"} > "${out_dir}/stdout.log" 2>&1
  echo "[DONE]  ${task} at $(date -u +%H:%M:%S)"
}

run_pair() {
  local wave_name="$1"
  local s_a="$2" enh_a="$3" loss_a="$4" sel_a="$5" tag_a="$6"
  local s_b="$7" enh_b="$8" loss_b="$9" sel_b="${10}" tag_b="${11}"

  echo "=================================================================="
  echo "Launching 2-Seed Wave [${wave_name}] (${CROP_SIZE}x${CROP_SIZE}, ${EPOCHS} epochs, grad_ckpt=${GRAD_CHECKPOINTING})"
  echo "  Slot A: ${tag_a} seed${s_a} | enhance=${enh_a} | loss=${loss_a} | select=${sel_a}"
  echo "  Slot B: ${tag_b} seed${s_b} | enhance=${enh_b} | loss=${loss_b} | select=${sel_b}"
  echo "=================================================================="

  run_exp "${s_a}" "${enh_a}" "${loss_a}" "${sel_a}" "${tag_a}" &
  local pid_a=$!

  run_exp "${s_b}" "${enh_b}" "${loss_b}" "${sel_b}" "${tag_b}" &
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
echo "Starting 2x3 Unified RETFound DRIVE Topology Ablation in 3x2 Waves"
echo "  Adaptation: ${ADAPTATION} | Enhance: true | grad_checkpointing: ${GRAD_CHECKPOINTING}"
echo "  Wave 1: enh_topo_v2 (seed 0, 1)"
echo "  Wave 2: enh_topo_v2 (seed 2) + enh_cldice (seed 0)"
echo "  Wave 3: enh_cldice (seed 1, 2)"
echo "=================================================================="

run_pair "Wave 1" \
  0 "true" "bce,dice,cldice,skelrec" "dice_cldice" "enh_topo_v2" \
  1 "true" "bce,dice,cldice,skelrec" "dice_cldice" "enh_topo_v2"

run_pair "Wave 2" \
  2 "true" "bce,dice,cldice,skelrec" "dice_cldice" "enh_topo_v2" \
  0 "true" "bce,dice,cldice"         "dice_cldice" "enh_cldice"

run_pair "Wave 3" \
  1 "true" "bce,dice,cldice"         "dice_cldice" "enh_cldice" \
  2 "true" "bce,dice,cldice"         "dice_cldice" "enh_cldice"

echo "=================================================================="
echo "All 2x3 = 6 runs completed! Aggregating 9-metric Mean ± Std across enhance=true runs..."
echo "=================================================================="

ADAPTATION="${ADAPTATION}" CROP_SIZE="${CROP_SIZE}" DATASET="${DATASET}" "${PYTHON}" -c "
import glob, json, os, numpy as np

adapt = os.environ['ADAPTATION']
cs = os.environ['CROP_SIZE']
ds = os.environ['DATASET']

configs = [
    ('enh',         'enhance', 'bce,dice',                        'dice'),
    ('enh_topo',    'enhance', 'bce,dice,cldice,skelrec (dil)',   'dice_cldice'),
    ('enh_cldice',  'enhance', 'bce,dice,cldice',                 'dice_cldice'),
    ('enh_topo_v2', 'enhance', 'bce,dice,cldice,skelrec (exact)', 'dice_cldice'),
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
for tag, enh_lbl, loss_lbl, sel_lbl in configs:
    pat = f'unified_retfound_mae_{adapt}_{tag}_{cs}_{ds}_seed[0-9]*'
    files = sorted(glob.glob(f'./output_dir/{pat}/summary.json'))
    assert len(files) == 3, f'Expected 3 seed summary.json files for {tag}, found {len(files)}: {files}'
    rows = [json.load(open(f))['test'] for f in files]
    label = f'Unified RETFound | {adapt} | {enh_lbl} | {loss_lbl} | select={sel_lbl}'
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

out_json = f'./output_dir/unified_retfound_{adapt}_skelrec_fix_ablation_{ds}_{cs}_summary.json'
with open(out_json, 'w') as f:
    json.dump(summary_all, f, indent=2)
print(f'Saved aggregated 9-metric ablation summary to {out_json}')
"
