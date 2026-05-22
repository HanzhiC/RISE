#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/configs/ltx_model/finetune.yaml"
DATASET_BASE="${SCRIPT_DIR}/dataset"
DATASETS=(
    pnp_basketball
    pnp_microwave
    pnp_ricecooker
    pnp_socks
    wipe_table
)
NORM_FILE="${SCRIPT_DIR}/data/utils/action_norm.json"
CKPT_ROOT="${SCRIPT_DIR}/checkpoints"
DIFFUSION_CKPT="${CKPT_ROOT}/dynamics_model/pretrained/diffusion_pytorch_model.safetensors"

require_file() {
    local path=$1
    if [ ! -f "$path" ]; then
        echo "Error: required file not found: $path"
        exit 1
    fi
}

require_dir() {
    local path=$1
    if [ ! -d "$path" ]; then
        echo "Error: required directory not found: $path"
        exit 1
    fi
}

check_dataset() {
    local dataset_name=$1
    local dataset_dir="${DATASET_BASE}/${dataset_name}"
    local videos_dir="${dataset_dir}/videos_small/chunk-000"

    require_dir "$dataset_dir"
    require_dir "${videos_dir}/image"
    require_dir "${videos_dir}/wrist_image"

    if ! find "${videos_dir}/image" -maxdepth 1 -type f -name "*.mp4" | grep -q .; then
        echo "Error: no image camera mp4 files found under ${videos_dir}/image"
        exit 1
    fi

    if ! find "${videos_dir}/wrist_image" -maxdepth 1 -type f -name "*.mp4" | grep -q .; then
        echo "Error: no wrist_image camera mp4 files found under ${videos_dir}/wrist_image"
        exit 1
    fi
}

require_dir "$DATASET_BASE"
require_dir "$CKPT_ROOT"
require_file "$DIFFUSION_CKPT"
require_file "$NORM_FILE"

for dataset_name in "${DATASETS[@]}"; do
    check_dataset "$dataset_name"
done

echo "Starting finetune with datasets: ${DATASETS[*]}"
echo "Config: ${CONFIG_FILE}"

bash scripts/train.sh main.py configs/ltx_model/finetune.yaml
