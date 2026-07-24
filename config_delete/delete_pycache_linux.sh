#!/usr/bin/env bash
# DecayCore — remove Python bytecode cache directories from the source tree
# (Linux).
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"

if [ ! -f "$repo_root/pyproject.toml" ] || [ ! -d "$repo_root/src/decaycore" ]; then
    echo "Refusing to continue: DecayCore source root was not found." >&2
    exit 2
fi

assume_yes=0
dry_run=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes) assume_yes=1 ;;
        -n|--dry-run) dry_run=1 ;;
        -h|--help)
            echo "Usage: $0 [-y|--yes] [-n|--dry-run]"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

# Keep the scan inside project-owned Python source directories. In particular,
# do not remove caches from .venv or generated build/output directories.
scan_roots=(
    "$repo_root/src"
    "$repo_root/tests"
    "$repo_root/scripts"
    "$repo_root/pyinstaller_hooks"
    "$repo_root/decaycore-dsp"
    "$repo_root/decaycore-scoring"
)

targets=()
[ -d "$repo_root/__pycache__" ] && targets+=("$repo_root/__pycache__")
for root in "${scan_roots[@]}"; do
    [ -d "$root" ] || continue
    while IFS= read -r -d '' directory; do
        targets+=("$directory")
    done < <(find "$root" -type d -name __pycache__ -prune -print0)
done

if [ "${#targets[@]}" -eq 0 ]; then
    echo "Nothing to delete — no project __pycache__ directories found."
    exit 0
fi

echo "The following Python cache directories will be deleted:"
for target in "${targets[@]}"; do
    echo "  $target"
done

if [ "$dry_run" -eq 1 ]; then
    echo "(dry run — nothing deleted)"
    exit 0
fi

if [ "$assume_yes" -ne 1 ]; then
    printf "Delete these %d directories? [y/N] " "${#targets[@]}"
    read -r reply
    case "$reply" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

rc=0
for target in "${targets[@]}"; do
    if rm -rf -- "$target"; then
        echo "Deleted: $target"
    else
        echo "Failed:  $target" >&2
        rc=1
    fi
done
exit "$rc"
