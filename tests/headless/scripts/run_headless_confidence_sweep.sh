#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
parse_common_args "$@"
BASE_OUT="$OUT/confidence_sweep"
for label_pull in off:0 025:0.25 050:0.50 075:0.75; do
  label="${label_pull%%:*}"
  pull="${label_pull##*:}"
  run_out="$BASE_OUT/confidence_$label"
  write_base_config "$run_out" "$LEFT_WAV" "$RIGHT_WAV" "confidence_$label" "Harman10" 20 250 5 15 "Asymmetric" 65536 400 true 50 "$pull" 82
  if [[ "$label" == "off" ]]; then
    python - "$run_out/config.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
d["confidence"] = {"enabled": False, "pull": 0}
json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
PY
  fi
  set +e
  run_headless "$run_out"
  code=$?
  set -e
  echo "$code" > "$run_out/exit_code.txt"
  publish_worker_session "$run_out" "${WORKER_IN:-}" "confidence_$label"
  write_influx_metrics "$run_out/metrics.json" "confidence_$label"
done
write_summary_json "$BASE_OUT" "$OUT/confidence_sweep_metrics.json"
echo "summary.json: $OUT/confidence_sweep_metrics.json"
