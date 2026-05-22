#!/usr/bin/env bash
# Download AnranZZ robot datasets from Hugging Face into ./dataset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$HOME/anaconda3/envs/rise/bin/huggingface-cli" ]]; then
        HF_CLI="$HOME/anaconda3/envs/rise/bin/huggingface-cli"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_CLI="$(command -v huggingface-cli)"
    else
        echo "Error: huggingface-cli not found. Install huggingface_hub or activate rise env."
        exit 1
    fi
else
    HF_CLI="$(dirname "$PYTHON_BIN")/huggingface-cli"
    if [[ ! -x "$HF_CLI" ]]; then
        HF_CLI="$(command -v huggingface-cli)"
    fi
fi

mkdir -p dataset

DATASETS=(
    wipe_table
    pnp_socks
    pnp_microwave
    pnp_basketball
    pnp_ricecooker
)

for name in "${DATASETS[@]}"; do
    echo "========== AnranZZ/${name} -> dataset/${name} =========="
    "$HF_CLI" download "AnranZZ/${name}" \
        --repo-type dataset \
        --local-dir "dataset/${name}"
done

echo "Done. Datasets under ${REPO_ROOT}/dataset:"
ls -la dataset/
