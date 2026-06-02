#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
parse_common_args "$@"
write_base_config "$OUT" "$LEFT_WAV" "$RIGHT_WAV" "minimum_phase" "Harman10" 20 250 5 15 "Minimum" 65536 400 true 50 0.05 82
run_headless_publish_exit "$OUT" "${WORKER_IN:-}" "$(basename "$OUT")"
