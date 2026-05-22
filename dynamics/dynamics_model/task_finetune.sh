，#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/configs/ltx_model/finetune.yaml"
DATASET_NAME="open_the_ricecooker_pi_abs"
DATASET_DIR="${SCRIPT_DIR}/dataset/${DATASET_NAME}"
VIDEOS_DIR="${DATASET_DIR}/videos_small/chunk-000"
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

require_dir "$DATASET_DIR"
require_dir "$CKPT_ROOT"
require_file "$DIFFUSION_CKPT"
require_file "$NORM_FILE"

for cam in image wrist_image; do
    require_dir "${VIDEOS_DIR}/${cam}"
done

if ! find "${VIDEOS_DIR}/image" -maxdepth 1 -type f -name "*.mp4" | grep -q .; then
    echo "Error: no image camera mp4 files found under ${VIDEOS_DIR}/image"
    exit 1
fi

if ! find "${VIDEOS_DIR}/wrist_image" -maxdepth 1 -type f -name "*.mp4" | grep -q .; then
    echo "Error: no wrist_image camera mp4 files found under ${VIDEOS_DIR}/wrist_image"
    exit 1
fi

echo "Starting finetune with dataset: ${DATASET_NAME}"
echo "Config: ${CONFIG_FILE}"

bash scripts/train.sh main.py configs/ltx_model/finetune.yaml
