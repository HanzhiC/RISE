#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$HOME/anaconda3/envs/rise/bin/python" ]]; then
        PYTHON_BIN="$HOME/anaconda3/envs/rise/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        PYTHON_BIN="$HOME/anaconda3/envs/rise/bin/python"
    fi
fi

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/dataset/open_the_ricecooker_pi_abs}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/checkpoints}"
DIFFUSION_CKPT="${DIFFUSION_CKPT:-${REPO_ROOT}/results/2026_05_19_18_25_28/step_7000/diffusion_pytorch_model.safetensors}"
NORM_CONSTANT="${NORM_CONSTANT:-FINETUNE_TASK}"
NORM_CONFIG_PATH="${NORM_CONFIG_PATH:-${REPO_ROOT}/data/utils/action_norm.json}"
DOMAIN_NAME="${DOMAIN_NAME:-$(basename "$DATASET_ROOT")}"
RETURN_ACTION="${RETURN_ACTION:-false}"
ACTION_CHUNK="${ACTION_CHUNK:-50}"

WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/tmp_infer_case}"
IMAGE_ROOT="${WORK_ROOT}/images"
OUTPUT_PATH="${WORK_ROOT}/outputs"
ACT_TOKENS_PATH="${WORK_ROOT}/act_tokens.pt"

VIDEO_DIR_IMAGE="${DATASET_ROOT}/videos_small/chunk-000/image"
PARQUET_DIR="${DATASET_ROOT}/data/chunk-000"

require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "Error: directory not found: $1"
        exit 1
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Error: file not found: $1"
        exit 1
    fi
}

require_dir "$DATASET_ROOT"
require_dir "$VIDEO_DIR_IMAGE"
require_dir "$PARQUET_DIR"
require_dir "$MODEL_ROOT"
require_file "$DIFFUSION_CKPT"

FIRST_IMAGE_MP4="$(find "$VIDEO_DIR_IMAGE" -maxdepth 1 -type f -name '*.mp4' | sort | head -n 1)"
if [[ -z "$FIRST_IMAGE_MP4" ]]; then
    echo "Error: no mp4 found under $VIDEO_DIR_IMAGE"
    exit 1
fi

EPISODE_STEM="$(basename "$FIRST_IMAGE_MP4" .mp4)"
IMAGE_MP4="$VIDEO_DIR_IMAGE/${EPISODE_STEM}.mp4"
PARQUET_PATH="$PARQUET_DIR/${EPISODE_STEM}.parquet"

if [[ ! -f "$PARQUET_PATH" ]]; then
    echo "Warning: matching parquet not found: $PARQUET_PATH"
    PARQUET_PATH="$(find "$PARQUET_DIR" -maxdepth 1 -type f -name '*.parquet' | sort | head -n 1)"
fi

require_file "$IMAGE_MP4"
require_file "$PARQUET_PATH"

mkdir -p "$IMAGE_ROOT/observation.images.top_head"
mkdir -p "$OUTPUT_PATH"

echo "return_action: $RETURN_ACTION"
echo "action_chunk: $ACTION_CHUNK"

"$PYTHON_BIN" - "$IMAGE_MP4" "$PARQUET_PATH" "$IMAGE_ROOT" "$ACT_TOKENS_PATH" <<'PY'
import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd

image_mp4, parquet_path, image_root, act_tokens_path = sys.argv[1:5]


def save_first_frame(video_path: str, out_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if (not ok) or frame is None:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    if not cv2.imwrite(out_path, frame):
        raise RuntimeError(f"Failed to write image to {out_path}")


save_first_frame(image_mp4, os.path.join(image_root, "observation.images.top_head", "0.png"))


def ensure_array(value):
    arr = np.asarray(value)
    return arr.astype(np.float32)


df = pd.read_parquet(parquet_path)
if "actions" in df.keys():
    actions = np.stack([ensure_array(df["actions"].iloc[i]) for i in range(len(df))])
else:
    cols = ["action.left_arm", "action.left_gripper", "action.right_arm", "action.right_gripper"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Unsupported parquet format, missing columns: {missing}")
    actions = np.stack([
        np.concatenate([ensure_array(df.at[i, c]) for c in cols])
        for i in range(len(df))
    ])

if actions.shape[0] < 50:
    raise RuntimeError(f"Need at least 50 action rows to match finetune action_chunk, but got {actions.shape[0]}")

# Match data/data_finetune.py: action_chunk=50, chunk=25 -> token indices 1,3,...,49
act_tokens = torch.from_numpy(actions[1:50:2]).unsqueeze(0)  # [1, 25, action_dim]
torch.save(act_tokens, act_tokens_path)
PY

cd "$REPO_ROOT"

cmd=(bash "$REPO_ROOT/infer.sh" \
    --model-root "$MODEL_ROOT" \
    --diffusion-ckpt "$DIFFUSION_CKPT" \
    --image-root "$IMAGE_ROOT" \
    --act-tokens "$ACT_TOKENS_PATH" \
    --output "$OUTPUT_PATH" \
    --norm "$NORM_CONSTANT" \
    --norm-config-path "$NORM_CONFIG_PATH" \
    --domain-name "$DOMAIN_NAME" \
    --n-chunk 1)

"${cmd[@]}"

echo "Done. Output video: ${OUTPUT_PATH}/video.mp4"
