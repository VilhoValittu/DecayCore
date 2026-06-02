#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
parse_common_args "$@"
write_base_config "$OUT" "$LEFT_WAV" "$RIGHT_WAV" "linear_phase" "Harman10" 20 250 5 15 "Linear Phase" 131072 500 true 50 0.05 82
python - "$OUT/config.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
d.update({"ir_export_window_mode": "auto", "ir_export_window_shape": "tukey", "ir_export_tukey_alpha": 0.5})
d["windowing"] = {"mode": "auto", "window": "tukey", "tukey_alpha": 0.5}
json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
PY
run_headless_publish_exit "$OUT" "${WORKER_IN:-}" "$(basename "$OUT")"
