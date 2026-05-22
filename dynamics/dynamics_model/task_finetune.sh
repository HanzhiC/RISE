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
# finetune.yaml uses valid_cam: ['image'] only
REQUIRED_CAMERAS=(image)

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

count_mp4_files() {
    local dir=$1
    find "$dir" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l
}

check_dataset() {
    local dataset_name=$1
    local dataset_dir="${DATASET_BASE}/${dataset_name}"
    local videos_small_dir="${dataset_dir}/videos_small"

    require_dir "$dataset_dir"
    require_dir "$videos_small_dir"

    local chunk_dirs=()
    mapfile -t chunk_dirs < <(find "$videos_small_dir" -mindepth 1 -maxdepth 1 -type d -name 'chunk-*' | sort)
    if [ ${#chunk_dirs[@]} -eq 0 ]; then
        echo "Error: no chunk-* directories under ${videos_small_dir}"
        echo "Hint: run ./preprocess.sh ${dataset_name}"
        exit 1
    fi

    for cam in "${REQUIRED_CAMERAS[@]}"; do
        local found=0
        for chunk_dir in "${chunk_dirs[@]}"; do
            local cam_dir="${chunk_dir}/${cam}"
            if [ ! -d "$cam_dir" ]; then
                continue
            fi
            local n
            n="$(count_mp4_files "$cam_dir")"
            if [ "$n" -gt 0 ]; then
                found=1
                break
            fi
        done
        if [ "$found" -eq 0 ]; then
            echo "Error: no ${cam} camera mp4 files found under ${videos_small_dir}/chunk-*/${cam}"
            echo "Hint: run ./preprocess.sh ${dataset_name}"
            exit 1
        fi
    done
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
