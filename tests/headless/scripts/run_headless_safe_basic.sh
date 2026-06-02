#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
parse_common_args "$@"
write_base_config "$OUT" "$LEFT_WAV" "$RIGHT_WAV" "safe_basic" "Harman6" 20 200 4 12 "Asymmetric" 65536 300 true 25 0.25 48
run_headless_publish_exit "$OUT" "${WORKER_IN:-}" "$(basename "$OUT")"
