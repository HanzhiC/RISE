#!/bin/bash

# Build videos_small for LeRobot-style datasets.
# - Head cameras (e.g. observation.images.top_head): center-crop to square, then resize to 256x192.
# - Other cameras: center-crop to 256:192 aspect ratio, then resize to 256x192.
# - If a dataset has no videos/, encode from image columns stored in parquet.
#
# Usage:
#   ./preprocess.sh [dataset_name1] [dataset_name2] ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_BASE_DIR="${SCRIPT_DIR}/dataset"
TARGET_WIDTH=256
TARGET_HEIGHT=192

# ffmpeg: square center crop, then scale to TARGET_WIDTH x TARGET_HEIGHT
FFMPEG_VF_HEAD="crop=min(iw\\,ih):min(iw\\,ih),scale=${TARGET_WIDTH}:${TARGET_HEIGHT}"
# ffmpeg: center crop to target aspect ratio, then scale
FFMPEG_VF_DEFAULT="scale=${TARGET_WIDTH}:${TARGET_HEIGHT}:force_original_aspect_ratio=increase,crop=${TARGET_WIDTH}:${TARGET_HEIGHT}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg not found"
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found"
    exit 1
fi

is_head_camera() {
    local name=$1
    case "$name" in
        *top_head*|*images.head*|*/head/*|*head_rgb*|*head_color*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

process_dataset_from_videos() {
    local dataset_name=$1
    local dataset_dir="${DATASET_BASE_DIR}/${dataset_name}"
    local videos_dir=""
    local output_dir=""

    if [ ! -d "$dataset_dir" ]; then
        echo "Warning: dataset directory not found: $dataset_dir"
        return 1
    fi

    if [ -d "${dataset_dir}/videos" ]; then
        videos_dir="${dataset_dir}/videos"
    elif [ -d "${dataset_dir}/video" ]; then
        videos_dir="${dataset_dir}/video"
    else
        mapfile -t matched_video_dirs < <(find "$dataset_dir" -type d \( -name "videos" -o -name "video" \) | sort)
        if [ ${#matched_video_dirs[@]} -eq 0 ]; then
            echo "Warning: no video/videos directory found under: $dataset_dir"
            return 1
        fi
        videos_dir="${matched_video_dirs[0]}"
    fi

    local resolved_dataset_name
    resolved_dataset_name="$(dirname "${videos_dir#${DATASET_BASE_DIR}/}")"
    output_dir="${DATASET_BASE_DIR}/${resolved_dataset_name}/videos_small"

    mapfile -t video_files < <(find "$videos_dir" -type f -name "*.mp4" | sort)
    local total_videos=${#video_files[@]}

    if [ "$total_videos" -eq 0 ]; then
        echo "No video files found in $dataset_name"
        return 0
    fi

    echo "Processing dataset from existing videos: $resolved_dataset_name ($total_videos videos)"

    local processed=0
    local skipped=0
    local failed=0

    for i in "${!video_files[@]}"; do
        local video_path="${video_files[$i]}"
        local rel_path="${video_path#$videos_dir/}"
        local output_path="${output_dir}/${rel_path}"
        local output_dir_path
        output_dir_path="$(dirname "$output_path")"
        local vf_filter="$FFMPEG_VF_DEFAULT"

        if is_head_camera "$rel_path"; then
            vf_filter="$FFMPEG_VF_HEAD"
        fi

        mkdir -p "$output_dir_path"

        local current=$((i + 1))
        local percent=$((current * 100 / total_videos))
        local bar_length=40
        local filled=$((current * bar_length / total_videos))
        local bar=""

        for ((j=0; j<bar_length; j++)); do
            if [ "$j" -lt "$filled" ]; then
                bar="${bar}#"
            else
                bar="${bar}-"
            fi
        done

        if [ -f "$output_path" ]; then
            skipped=$((skipped + 1))
            printf "\r[%3d%%] [%s] %d/%d | processed:%d skipped:%d failed:%d" \
                "$percent" "$bar" "$current" "$total_videos" "$processed" "$skipped" "$failed"
            continue
        fi

        printf "\r[%3d%%] [%s] %d/%d | processed:%d skipped:%d failed:%d" \
            "$percent" "$bar" "$current" "$total_videos" "$processed" "$skipped" "$failed"

        if ffmpeg -i "$video_path" \
            -vf "$vf_filter" \
            -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p -y "$output_path" \
            -loglevel error 2>&1; then
            processed=$((processed + 1))
        else
            [ -f "$output_path" ] && rm -f "$output_path"
            failed=$((failed + 1))
        fi

        printf "\r[%3d%%] [%s] %d/%d | processed:%d skipped:%d failed:%d" \
            "$percent" "$bar" "$current" "$total_videos" "$processed" "$skipped" "$failed"
    done

    echo ""
    echo "Dataset $resolved_dataset_name: Total: $total_videos | Processed: $processed | Skipped: $skipped | Failed: $failed"
    echo ""
}

process_dataset_from_parquet() {
    local dataset_name=$1
    local dataset_dir="${DATASET_BASE_DIR}/${dataset_name}"

    if [ ! -d "$dataset_dir" ]; then
        echo "Warning: dataset directory not found: $dataset_dir"
        return 1
    fi

    echo "Processing dataset from parquet images: $dataset_name"

    python3 - "$dataset_dir" "$TARGET_WIDTH" "$TARGET_HEIGHT" <<'PY'
import io
import json
import os
import sys

dataset_dir = sys.argv[1]
target_width = int(sys.argv[2])
target_height = int(sys.argv[3])

try:
    import cv2
    import numpy as np
    import pandas as pd
    from PIL import Image
except ImportError as exc:
    print(f"Error: missing Python dependency for parquet video export: {exc}")
    print("Install requirements with pandas, pyarrow/fastparquet, pillow, numpy, and opencv-python.")
    sys.exit(1)


def is_head_camera(name: str) -> bool:
    markers = ("top_head", "images.head", "/head/", "head_rgb", "head_color")
    return any(marker in name for marker in markers)


def load_image_value(value):
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        frame = value
    elif isinstance(value, Image.Image):
        frame = np.array(value)
    elif isinstance(value, dict):
        if value.get("bytes") is not None:
            data = np.frombuffer(value["bytes"], dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif value.get("path"):
            frame = np.array(Image.open(value["path"]).convert("RGB"))
        else:
            return None
    elif isinstance(value, (bytes, bytearray)):
        data = np.frombuffer(value, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif isinstance(value, str):
        if not os.path.exists(value):
            return None
        frame = np.array(Image.open(value).convert("RGB"))
    else:
        return None

    if frame is None:
        return None

    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]

    return frame.astype(np.uint8)


def center_square_crop_and_resize(frame):
    """Head image: center crop to square, then resize to target_width x target_height."""
    h, w = frame.shape[:2]
    crop_size = min(h, w)
    x0 = (w - crop_size) // 2
    y0 = (h - crop_size) // 2
    cropped = frame[y0 : y0 + crop_size, x0 : x0 + crop_size]
    return cv2.resize(
        cropped, (target_width, target_height), interpolation=cv2.INTER_AREA
    )


def center_crop_aspect_and_resize(frame):
    """Non-head: center crop to target aspect ratio, then resize."""
    h, w = frame.shape[:2]
    src_aspect = w / h
    dst_aspect = target_width / target_height

    if src_aspect > dst_aspect:
        crop_h = h
        crop_w = int(round(h * dst_aspect))
        x0 = max(0, (w - crop_w) // 2)
        y0 = 0
    else:
        crop_w = w
        crop_h = int(round(w / dst_aspect))
        x0 = 0
        y0 = max(0, (h - crop_h) // 2)

    cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(
        cropped, (target_width, target_height), interpolation=cv2.INTER_AREA
    )


def preprocess_frame(frame, image_col: str):
    if is_head_camera(image_col):
        return center_square_crop_and_resize(frame)
    return center_crop_aspect_and_resize(frame)


meta_path = os.path.join(dataset_dir, "meta", "info.json")
if not os.path.exists(meta_path):
    print(f"Error: meta/info.json not found for {dataset_dir}")
    sys.exit(1)

with open(meta_path, "r", encoding="utf-8") as f:
    info = json.load(f)

fps = int(round(info.get("fps", 10)))
features = info.get("features", {})
image_columns = [name for name, spec in features.items() if spec.get("dtype") == "image"]

if not image_columns:
    print(f"Error: no image columns found in meta/info.json for {dataset_dir}")
    sys.exit(1)

parquet_files = []
data_root = os.path.join(dataset_dir, "data")
for root, _, files in os.walk(data_root):
    for file_name in sorted(files):
        if file_name.endswith(".parquet") and file_name.startswith("episode_"):
            parquet_files.append(os.path.join(root, file_name))
parquet_files.sort()

if not parquet_files:
    print(f"Error: no parquet episode files found under {data_root}")
    sys.exit(1)

output_root = os.path.join(dataset_dir, "videos_small")
os.makedirs(output_root, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
processed = 0
skipped = 0
failed = 0

print(f"Found {len(parquet_files)} parquet episodes")
print(f"Image columns: {', '.join(image_columns)}")

for idx, parquet_path in enumerate(parquet_files, start=1):
    rel_parent = os.path.basename(os.path.dirname(parquet_path))
    episode_name = os.path.splitext(os.path.basename(parquet_path))[0] + ".mp4"

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        print(f"[{idx}/{len(parquet_files)}] Failed to read {parquet_path}: {exc}")
        failed += 1
        continue

    current_failed = False
    wrote_any = False

    for image_col in image_columns:
        if image_col not in df.columns:
            continue

        cam_dir = os.path.join(output_root, rel_parent, image_col)
        os.makedirs(cam_dir, exist_ok=True)
        output_path = os.path.join(cam_dir, episode_name)

        if os.path.exists(output_path):
            continue

        writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        if not writer.isOpened():
            print(f"[{idx}/{len(parquet_files)}] Failed to open writer for {output_path}")
            current_failed = True
            continue

        frame_count = 0
        try:
            for value in df[image_col].tolist():
                frame = load_image_value(value)
                if frame is None:
                    continue
                frame = preprocess_frame(frame, image_col)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_count += 1
        finally:
            writer.release()

        if frame_count == 0:
            if os.path.exists(output_path):
                os.remove(output_path)
            print(f"[{idx}/{len(parquet_files)}] No valid frames for {parquet_path} column {image_col}")
            current_failed = True
            continue

        wrote_any = True

    if current_failed:
        failed += 1
    elif wrote_any:
        processed += 1
    else:
        skipped += 1

    print(f"[{idx}/{len(parquet_files)}] processed:{processed} skipped:{skipped} failed:{failed}", end="\r")

print()
print(f"Dataset {os.path.basename(dataset_dir)}: Total: {len(parquet_files)} | Processed: {processed} | Skipped: {skipped} | Failed: {failed}")
PY

    echo ""
}

process_dataset() {
    local dataset_name=$1
    local dataset_dir="${DATASET_BASE_DIR}/${dataset_name}"
    local info_json="${dataset_dir}/meta/info.json"

    if [ -d "${dataset_dir}/videos" ] || [ -d "${dataset_dir}/video" ]; then
        process_dataset_from_videos "$dataset_name"
        return
    fi

    if [ -f "$info_json" ]; then
        local total_videos
        total_videos="$(python3 - "$info_json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    info = json.load(f)
print(info.get("total_videos", 0))
PY
)"
        if [ "${total_videos}" = "0" ]; then
            process_dataset_from_parquet "$dataset_name"
            return
        fi
    fi

    process_dataset_from_videos "$dataset_name"
}

discover_dataset_names() {
    mapfile -t datasets < <(
        find "$DATASET_BASE_DIR" -mindepth 1 -maxdepth 1 -type d | xargs -r -n1 basename | sort
    )
}

if [ $# -eq 0 ]; then
    if [ ! -d "$DATASET_BASE_DIR" ]; then
        echo "Error: dataset directory not found: $DATASET_BASE_DIR"
        exit 1
    fi

    discover_dataset_names

    if [ ${#datasets[@]} -eq 0 ]; then
        echo "No datasets found in $DATASET_BASE_DIR"
        exit 0
    fi

    echo "Found ${#datasets[@]} dataset(s), processing all..."
    echo ""

    for dataset in "${datasets[@]}"; do
        process_dataset "$dataset"
    done
else
    for dataset_input in "$@"; do
        process_dataset "$dataset_input"
    done
fi
