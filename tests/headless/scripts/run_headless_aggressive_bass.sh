#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
parse_common_args "$@"
write_base_config "$OUT" "$LEFT_WAV" "$RIGHT_WAV" "aggressive_bass" "Harman10" 15 300 6 18 "Asymmetric" 65536 500 true 75 0.05 120
python - "$OUT/config.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
d["residual_peak_handling"] = {"enabled": True, "strength": "strong"}
json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
PY
run_headless_publish_exit "$OUT" "${WORKER_IN:-}" "$(basename "$OUT")"
