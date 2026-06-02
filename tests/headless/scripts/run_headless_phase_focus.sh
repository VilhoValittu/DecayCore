#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
parse_common_args "$@"
write_base_config "$OUT" "$LEFT_WAV" "$RIGHT_WAV" "phase_focus" "Harman10" 20 250 5 15 "Linear Phase" 65536 600 true 50 0.05 82
python - "$OUT/config.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
d.update({"gd_grad_limit_ms_per_oct": 1.5, "phase_safe_2058": True})
d["phase_options"] = {"enabled": True, "mode": "linear", "phase_limit_hz": 600, "phase_clamp_target_deg": 45, "gd_limiter_enabled": True}
json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
PY
run_headless_publish_exit "$OUT" "${WORKER_IN:-}" "$(basename "$OUT")"
