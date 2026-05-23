#!/usr/bin/env bash
# Run guided-action inference + future video for StretchRobot tasks.
#
# Usage:
#   export INFER_DIFFUSION_CKPT=results/<run>/step_<N>/diffusion_pytorch_model.safetensors
#   bash optimize_all.sh                              # all tasks
#   bash optimize_all.sh pnp-ricecooker wipe-table    # selected tasks
#   TASKS="pnp-socks pnp-basketball" bash optimize_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CFG_BASE="/home/wiss/chenh/storage/logs/egoasis4d-stretchrobot-vlawmvm-repre-ablation/ablation_repre_wo_geometric_frozenvla+wm+vm_stretchrobot"
INFER_DIFFUSION_CKPT="${INFER_DIFFUSION_CKPT:-results/2026_05_22_18_50_18/step_23000/diffusion_pytorch_model.safetensors}"

PYTHON="${PYTHON:-python}"

declare -A TASK_CFG_SUFFIX=(
    ["pnp-ricecooker"]="pnp-ricecooker"
    ["pnp-microwave"]="pnp-microwave"
    ["pnp-socks"]="pnp-socks"
    ["pnp-basketball"]="pnp-basketball"
    ["wipe-table"]="wipetable"
)
ALL_TASKS=(pnp-ricecooker pnp-microwave pnp-socks pnp-basketball wipe-table)

usage() {
    cat <<EOF
Usage: $(basename "$0") [task ...]

Run guided-action inference for StretchRobot tasks.
If no tasks are given, all tasks are run.

Available tasks:
  pnp-ricecooker
  pnp-microwave
  pnp-socks
  pnp-basketball
  wipe-table

Environment:
  INFER_DIFFUSION_CKPT  Diffusion checkpoint path
  TASKS                 Space- or comma-separated task list (used when no CLI args)
  PYTHON                Python executable (default: python)

Examples:
  bash $(basename "$0")
  bash $(basename "$0") pnp-ricecooker wipe-table
  TASKS="pnp-ricecooker,pnp-basketball,pnp-microwave" bash $(basename "$0")
EOF
}

resolve_tasks() {
    local -a selected=()
    local task

    if [ "$#" -gt 0 ]; then
        selected=("$@")
    elif [ -n "${TASKS:-}" ]; then
        local normalized="${TASKS//,/ }"
        read -r -a selected <<< "${normalized}"
    else
        selected=("${ALL_TASKS[@]}")
    fi

    for task in "${selected[@]}"; do
        if [ -z "${TASK_CFG_SUFFIX[${task}]+x}" ]; then
            echo "Error: unknown task '${task}'"
            echo ""
            usage
            exit 1
        fi
    done

    SELECTED_TASKS=("${selected[@]}")
}

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
        # --overwrite \
        --use_episode_correspondence \
        --infer_diffusion_ckpt "${INFER_DIFFUSION_CKPT}"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

resolve_tasks "$@"

echo "========== Optimize: guided action + future video =========="
echo "INFER_DIFFUSION_CKPT=${INFER_DIFFUSION_CKPT}"
echo "TASKS=${SELECTED_TASKS[*]}"
echo ""

for task in "${SELECTED_TASKS[@]}"; do
    run_guided_infer "${task}" "${TASK_CFG_SUFFIX[${task}]}"
done

echo "========== ALL TASKS FINISHED SUCCESSFULLY =========="

# pnp-ricecooker
# pnp-microwave
# pnp-socks
# pnp-basketball
# wipe-table

# bash optimize_all.sh pnp-basketball pnp-microwave wipe-table
# bash optimize_all.sh pnp-socks pnp-ricecooker