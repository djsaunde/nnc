#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: download_imagenet.sh [options]

Downloads the ILSVRC2012 ImageNet tarballs (train, val, devkit) using the
credentials from https://image-net.org/ that you obtained after accepting the
terms of use. The script performs authenticated HTTP downloads via curl/wget
and resumes partial files when possible.

Options:
  --output DIR          Directory to store the downloaded files (default: datasets/imagenet/raw)
  --username USER       ImageNet username (or set IMAGENET_USERNAME env var)
  --password PASS       ImageNet password/token (or set IMAGENET_PASSWORD env var)
  --no-train            Skip downloading the training tarball
  --no-val              Skip downloading the validation tarball
  --no-devkit           Skip downloading the devkit archive
  --dry-run             Print the planned downloads without fetching
  -h, --help            Show this help message and exit

Examples:
  IMAGENET_USERNAME=alice IMAGENET_PASSWORD=secret \
    ./scripts/download_imagenet.sh --output ~/datasets/imagenet

  ./scripts/download_imagenet.sh --username alice --output datasets/imagenet/raw
  # Password will be requested interactively if not provided via env/flag.

Notes:
  * You must register with ImageNet and agree to their terms. This script does
    not bypass authentication; it simply automates the curl/wget calls.
  * Downloads are large (~150 GB total). Ensure you have sufficient disk space
    and a stable connection. Use --dry-run to verify paths beforehand.
  * The resulting tarballs remain untouched; downstream preprocessing (e.g.
    conversion to the project-specific .rec format) is handled separately.
USAGE
}

URL_BASE="https://image-net.org/data/ILSVRC/2012"
TRAIN_ARCHIVE="ILSVRC2012_img_train.tar"
VAL_ARCHIVE="ILSVRC2012_img_val.tar"
DEVKIT_ARCHIVE="ILSVRC2012_devkit_t12.tar.gz"

download_train=1
download_val=1
download_devkit=1
dry_run=0
output_dir="datasets/imagenet/raw"
username="${IMAGENET_USERNAME:-}"
password="${IMAGENET_PASSWORD:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --output)
        if [[ $# -lt 2 ]]; then
            echo "Error: --output requires a directory argument" >&2
            exit 1
        fi
        output_dir="$2"
        shift 2
        ;;
    --username)
        if [[ $# -lt 2 ]]; then
            echo "Error: --username requires a value" >&2
            exit 1
        fi
        username="$2"
        shift 2
        ;;
    --password)
        if [[ $# -lt 2 ]]; then
            echo "Error: --password requires a value" >&2
            exit 1
        fi
        password="$2"
        shift 2
        ;;
    --no-train)
        download_train=0
        shift
        ;;
    --no-val)
        download_val=0
        shift
        ;;
    --no-devkit)
        download_devkit=0
        shift
        ;;
    --dry-run)
        dry_run=1
        shift
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown argument: $1" >&2
        usage
        exit 1
        ;;
    esac
done

if [[ "$download_train" -eq 0 && "$download_val" -eq 0 && "$download_devkit" -eq 0 ]]; then
    echo "Nothing to download (all components skipped)." >&2
    exit 1
fi

ensure_tool() {
    if command -v curl >/dev/null 2>&1; then
        echo curl
    elif command -v wget >/dev/null 2>&1; then
        echo wget
    else
        echo "Error: curl or wget is required." >&2
        exit 1
    fi
}

downloader=$(ensure_tool)

prompt_if_needed() {
    local name="$1"
    local value="$2"
    local silent="$3"
    if [[ -n "$value" ]]; then
        echo "$value"
        return
    fi
    if [[ ! -t 0 ]]; then
        echo "Error: $name not set and stdin is not a TTY for prompting." >&2
        exit 1
    fi
    if [[ "$silent" == "yes" ]]; then
        read -r -s -p "Enter $name: " value
        echo
    else
        read -r -p "Enter $name: " value
    fi
    echo "$value"
}

if [[ "$dry_run" -eq 0 ]]; then
    username=$(prompt_if_needed "ImageNet username" "$username" "no")
    password=$(prompt_if_needed "ImageNet password" "$password" "yes")
else
    username="${username:-<prompted>}"
    password="${password:-<prompted>}"
fi

mkdir -p "$output_dir"

queue=()
if [[ "$download_train" -eq 1 ]]; then
    queue+=("$TRAIN_ARCHIVE")
fi
if [[ "$download_val" -eq 1 ]]; then
    queue+=("$VAL_ARCHIVE")
fi
if [[ "$download_devkit" -eq 1 ]]; then
    queue+=("$DEVKIT_ARCHIVE")
fi

echo "Preparing to download to: $output_dir"
for item in "${queue[@]}"; do
    echo " - $item"
    if [[ "$dry_run" -eq 1 ]]; then
        continue
    fi
    dest="$output_dir/$item"
    url="$URL_BASE/$item"
    if [[ -f "$dest" ]]; then
        echo "   File already exists, resuming: $dest"
    fi
    if [[ "$downloader" == curl ]]; then
        curl_opts=("-L" "-C" "-" "--fail" "--retry" "5" "--retry-delay" "5" \
                   "--user" "$username:$password" "-o" "$dest" "$url")
        echo "   Running: curl ${curl_opts[*]}"
        if ! curl "${curl_opts[@]}"; then
            echo "Error: download failed for $item" >&2
            exit 1
        fi
    else
        wget_opts=("--continue" "--tries=5" "--user=$username" "--password=$password" \
                   "--directory-prefix=$output_dir" "$url")
        echo "   Running: wget ${wget_opts[*]}"
        if ! wget "${wget_opts[@]}"; then
            echo "Error: download failed for $item" >&2
            exit 1
        fi
    fi
    echo "   Finished: $dest"
    echo
    if [[ "$item" == "$TRAIN_ARCHIVE" ]]; then
        echo "   Reminder: unpacking the training tarball yields per-class tar files."
    fi
done

echo "All requested downloads completed."
