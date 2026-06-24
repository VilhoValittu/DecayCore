#!/usr/bin/env bash
# DecayCore — remove the Optuna disk cache and config.json (macOS).
#
# Mirrors src/decaycore/app_paths.py and
# src/decaycore/auto_mode/optuna_backend_storage.py.
set -u

PROG="DecayCore"

# On macOS data and config share the Application Support directory.
data_dir="$HOME/Library/Application Support/$PROG"
config_dir="$data_dir"
legacy_dir="$HOME/.camillafir"

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

# Collect targets: Optuna cache + other auto-mode disk caches + config.json.
targets=()
for base in "$data_dir" "$legacy_dir"; do
    for pat in "decaycore*optuna*" "decaycore_auto_mode_cache_*.json" "auto_mode_filter_priors.json"; do
        for f in "$base"/$pat; do
            [ -e "$f" ] && targets+=("$f")
        done
    done
done
[ -e "$config_dir/config.json" ] && targets+=("$config_dir/config.json")

if [ "${#targets[@]}" -eq 0 ]; then
    echo "Nothing to delete — no Optuna cache or config.json found."
    exit 0
fi

echo "The following files will be deleted:"
for t in "${targets[@]}"; do
    echo "  $t"
done

if [ "$dry_run" -eq 1 ]; then
    echo "(dry run — nothing deleted)"
    exit 0
fi

if [ "$assume_yes" -ne 1 ]; then
    printf "Delete these %d file(s)? [y/N] " "${#targets[@]}"
    read -r reply
    case "$reply" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

rc=0
for t in "${targets[@]}"; do
    if rm -f -- "$t"; then
        echo "Deleted: $t"
    else
        echo "Failed:  $t" >&2
        rc=1
    fi
done
exit "$rc"
