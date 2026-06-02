#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
parse_common_args "$@"
write_base_config "$OUT" "$LEFT_WAV" "$RIGHT_WAV" "mixed_phase" "Harman10" 20 250 5 15 "Mixed" 65536 600 true 50 0.05 82
python - "$OUT/config.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
d.update({"mixed_freq": 200, "trans_width": 140, "enable_ir_pre_energy_guard": True, "max_pre_ringing_db": -35})
d["phase_options"] = {"mode": "mixed", "split_frequency_hz": 200, "transition_width_hz": 140, "min_phase_above_split": True, "pre_ringing_guard_enabled": True}
json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
PY
run_headless_publish_exit "$OUT" "${WORKER_IN:-}" "$(basename "$OUT")"
