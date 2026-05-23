#!/usr/bin/env bash
# Run guided-action inference + future video for all StretchRobot tasks.
#
# Usage:
#   export INFER_DIFFUSION_CKPT=results/<run>/step_<N>/diffusion_pytorch_model.safetensors
#   bash optimize_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CFG_BASE="/home/wiss/chenh/storage/logs/egoasis4d-stretchrobot-vlawmvm-repre-ablation/ablation_repre_wo_geometric_frozenvla+wm+vm_stretchrobot"
INFER_DIFFUSION_CKPT="${INFER_DIFFUSION_CKPT:-results/PLACEHOLDER_RUN/step_0/diffusion_pytorch_model.safetensors}"

PYTHON="${PYTHON:-python}"

run_step() {
    local step_name="$1"
    shift
    echo "====> [START] ${step_name}"
    "$@"
    local exit_code=$?
    if [ "${exit_code}" -ne 0 ]; then
        echo "!!!! [FAILED] ${step_name} (exit_code=${exit_code})"
        echo "!!!! [FAILED_CMD] $*"
        exit "${exit_code}"
    fi
    echo "====> [DONE] ${step_name}"
}

run_guided_infer() {
    local task="$1"
    local cfg_suffix="$2"
    local cfg_path="${CFG_BASE}_${cfg_suffix}/config.yaml"

    if [ ! -f "${cfg_path}" ]; then
        echo "Error: config not found: ${cfg_path}"
        exit 1
    fi
    if [ ! -f "${INFER_DIFFUSION_CKPT}" ]; then
        echo "Error: diffusion ckpt not found: ${INFER_DIFFUSION_CKPT}"
        echo "Set INFER_DIFFUSION_CKPT to your finetuned checkpoint, e.g.:"
        echo "  export INFER_DIFFUSION_CKPT=results/2026_05_22_18_50_18/step_7000/diffusion_pytorch_model.safetensors"
        exit 1
    fi

    run_step "guided_${task}" \
        "${PYTHON}" rl_pipeline/rl_inference_stretchrobot_guided_action.py \
        --cfg "${cfg_path}" \
        --task "${task}" \
        --overwrite \
        --use_episode_correspondence \
        --infer_diffusion_ckpt "${INFER_DIFFUSION_CKPT}"
}

echo "========== Optimize: guided action + future video =========="
echo "INFER_DIFFUSION_CKPT=${INFER_DIFFUSION_CKPT}"
echo "INFER_SAVE_DIR=${INFER_SAVE_DIR}"
echo ""

run_guided_infer "pnp-ricecooker" "pnp-ricecooker"
run_guided_infer "pnp-microwave" "pnp-microwave"
run_guided_infer "pnp-socks" "pnp-socks"
run_guided_infer "pnp-basketball" "pnp-basketball"
run_guided_infer "wipe-table" "wipetable"

echo "========== ALL TASKS FINISHED SUCCESSFULLY =========="
