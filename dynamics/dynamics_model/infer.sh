#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CONFIG="${SCRIPT_DIR}/configs/ltx_model/infer.yaml"
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

MODEL_ROOT="${SCRIPT_DIR}/checkpoints"
DIFFUSION_CKPT="${MODEL_ROOT}/dynamics_model/pretrained/diffusion_pytorch_model.safetensors"
IMAGE_ROOT=""
ACT_TOKENS_PATH=""
OUTPUT_PATH="${SCRIPT_DIR}/infer_outputs"
NORM_CONSTANT="FINETUNE_TASK"
N_CHUNK="1"
NORM_CONFIG_PATH=""
DOMAIN_NAME=""
RETURN_ACTION="false"
ACTION_CHUNK="50"

usage() {
    cat <<'EOF'
Usage:
  bash infer.sh \
    --image-root /abs/path/to/obs_images \
    --act-tokens /abs/path/to/act_tokens.pt \
    --diffusion-ckpt /abs/path/to/step_xxx/diffusion_pytorch_model.safetensors \
    [--model-root /abs/path/to/checkpoints_root] \
    [--output /abs/path/to/output_dir] \
    [--norm FINETUNE_TASK|PRETRAIN_1|PRETRAIN_2] \
    [--norm-config-path /abs/path/to/action_norm.json] \
    [--domain-name dataset_name] \
    [--return-action] \
    [--action-chunk 50] \
    [--n-chunk 1]

Notes:
  - --model-root should contain tokenizer/text_encoder/vae folders.
  - --diffusion-ckpt should be your trained model file.
  - image root should contain:
      observation.images.top_head/0.png
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-root)
            MODEL_ROOT="$2"
            shift 2
            ;;
        --diffusion-ckpt)
            DIFFUSION_CKPT="$2"
            shift 2
            ;;
        --image-root)
            IMAGE_ROOT="$2"
            shift 2
            ;;
        --act-tokens)
            ACT_TOKENS_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --norm)
            NORM_CONSTANT="$2"
            shift 2
            ;;
        --norm-config-path)
            NORM_CONFIG_PATH="$2"
            shift 2
            ;;
        --domain-name)
            DOMAIN_NAME="$2"
            shift 2
            ;;
        --return-action)
            RETURN_ACTION="true"
            shift 1
            ;;
        --action-chunk)
            ACTION_CHUNK="$2"
            shift 2
            ;;
        --n-chunk)
            N_CHUNK="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$IMAGE_ROOT" || -z "$ACT_TOKENS_PATH" ]]; then
    echo "Error: --image-root and --act-tokens are required."
    usage
    exit 1
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "Error: base config not found: $BASE_CONFIG"
    exit 1
fi

if [[ ! -d "$MODEL_ROOT" ]]; then
    echo "Error: model root not found: $MODEL_ROOT"
    exit 1
fi

if [[ ! -f "$DIFFUSION_CKPT" ]]; then
    echo "Error: diffusion checkpoint not found: $DIFFUSION_CKPT"
    exit 1
fi

if [[ ! -f "$ACT_TOKENS_PATH" ]]; then
    echo "Error: act tokens not found: $ACT_TOKENS_PATH"
    exit 1
fi

if [[ ! -f "$IMAGE_ROOT/observation.images.top_head/0.png" ]]; then
    echo "Error: expected image missing: $IMAGE_ROOT/observation.images.top_head/0.png"
    exit 1
fi

case "$NORM_CONSTANT" in
    FINETUNE_TASK|PRETRAIN_1|PRETRAIN_2) ;;
    *)
        echo "Error: --norm must be FINETUNE_TASK, PRETRAIN_1, or PRETRAIN_2"
        exit 1
        ;;
esac

mkdir -p "$OUTPUT_PATH"

TMP_CONFIG="$(mktemp /tmp/infer_config.XXXXXX.yaml)"

"$PYTHON_BIN" - "$BASE_CONFIG" "$TMP_CONFIG" "$MODEL_ROOT" "$DIFFUSION_CKPT" <<'PY'
import sys
import json
import os
import yaml

base_cfg, out_cfg, model_root, diffusion_ckpt = sys.argv[1:5]

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["pretrained_model_name_or_path"] = model_root
cfg["diffusion_model"]["model_path"] = diffusion_ckpt

with open(out_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "Running inference with:"
echo "  model_root:      $MODEL_ROOT"
echo "  diffusion_ckpt:  $DIFFUSION_CKPT"
echo "  image_root:      $IMAGE_ROOT"
echo "  act_tokens:      $ACT_TOKENS_PATH"
echo "  output:          $OUTPUT_PATH"
echo "  config:          $TMP_CONFIG"
echo "  norm_config:     ${NORM_CONFIG_PATH:-<default>}"
echo "  domain_name:     ${DOMAIN_NAME:-<default>}"
echo "  return_action:   $RETURN_ACTION"
echo "  action_chunk:    $ACTION_CHUNK"

if [[ "$RETURN_ACTION" == "true" ]]; then
    echo "Error: return-action is not supported by the current video-only checkpoint."
    echo "       Use the video-only path, or retrain/load a checkpoint with action_expert enabled."
    exit 1
fi

cmd=("$PYTHON_BIN" "$SCRIPT_DIR/infer.py" \
    --config_file "$TMP_CONFIG" \
    --image_root "$IMAGE_ROOT" \
    --output_path "$OUTPUT_PATH" \
    --n_chunk "$N_CHUNK" \
    --act_tokens_path "$ACT_TOKENS_PATH" \
    --norm_constant "$NORM_CONSTANT")

if [[ -n "$NORM_CONFIG_PATH" ]]; then
    cmd+=(--norm_config_path "$NORM_CONFIG_PATH")
fi

if [[ -n "$DOMAIN_NAME" ]]; then
    cmd+=(--domain_name "$DOMAIN_NAME")
fi

"${cmd[@]}"

echo "Inference done. Output video: $OUTPUT_PATH/video.mp4"