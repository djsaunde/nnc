#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: download_tiny_imagenet.sh [options]

Fetches the Tiny ImageNet (200-class) dataset from the CS231n hosting site and
optionally unpacks it into your datasets directory.

Options:
  --output DIR      Directory where the archive and extracted data will live
                    (default: datasets)
  --archive PATH    Custom path for the downloaded zip (default: DIR/tiny-imagenet-200.zip)
  --no-extract      Skip extraction, leaving only the zip file
  --force           Re-download even if the archive already exists
  --dry-run         Print the planned actions without downloading/extracting
  -h, --help        Show this help message and exit

Example:
  ./scripts/download_tiny_imagenet.sh --output datasets

Notes:
  * The download is ~250 MB and extracts to ~1 GB. Ensure enough space.
  * Extraction produces a directory named tiny-imagenet-200 containing
    train/val/test subfolders.
USAGE
}

URL="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DEFAULT_ARCHIVE_NAME="tiny-imagenet-200.zip"

dest_root="datasets"
archive_path=""
extract=1
force=0
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
    --output)
        if [[ $# -lt 2 ]]; then
            echo "Error: --output requires a directory argument" >&2
            exit 1
        fi
        dest_root="$2"
        shift 2
        ;;
    --archive)
        if [[ $# -lt 2 ]]; then
            echo "Error: --archive requires a path" >&2
            exit 1
        fi
        archive_path="$2"
        shift 2
        ;;
    --no-extract)
        extract=0
        shift
        ;;
    --force)
        force=1
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

mkdir -p "$dest_root"
if [[ -z "$archive_path" ]]; then
    archive_path="$dest_root/$DEFAULT_ARCHIVE_NAME"
fi

ensure_downloader() {
    if command -v curl >/dev/null 2>&1; then
        echo curl
    elif command -v wget >/dev/null 2>&1; then
        echo wget
    else
        echo "Error: require curl or wget to download." >&2
        exit 1
    fi
}

downloader=$(ensure_downloader)

perform_download() {
    local url="$1"
    local dest="$2"
    local dl="$3"
    if [[ "$dl" == curl ]]; then
        curl -L -C - --fail --retry 5 --retry-delay 5 -o "$dest" "$url"
    else
        wget --continue --tries=5 -O "$dest" "$url"
    fi
}

if [[ "$dry_run" -eq 1 ]]; then
    echo "[dry-run] Would download $URL to $archive_path"
else
    if [[ -f "$archive_path" && "$force" -ne 1 ]]; then
        echo "Archive already exists: $archive_path (use --force to re-download)"
    else
        echo "Downloading Tiny ImageNet to $archive_path using $downloader..."
        perform_download "$URL" "$archive_path" "$downloader"
    fi
fi

if [[ "$extract" -eq 1 ]]; then
    if [[ "$dry_run" -eq 1 ]]; then
        echo "[dry-run] Would extract $archive_path into $dest_root"
    else
        if ! command -v unzip >/dev/null 2>&1; then
            echo "Error: unzip not found; install it or rerun with --no-extract." >&2
            exit 1
        fi
        echo "Extracting $archive_path into $dest_root ..."
        unzip -q "$archive_path" -d "$dest_root"
        echo "Extraction complete. Dataset available at $dest_root/tiny-imagenet-200"
    fi
fi

echo "Done."
