#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
parse_common_args "$@"
BASE_OUT="$OUT/tdc_sweep"
for label_strength in 025:25 050:50 075:75 100:100; do
  label="${label_strength%%:*}"
  strength="${label_strength##*:}"
  run_out="$BASE_OUT/tdc_$label"
  write_base_config "$run_out" "$LEFT_WAV" "$RIGHT_WAV" "tdc_$label" "Harman10" 20 250 5 15 "Asymmetric" 65536 400 true "$strength" 0.05 82
  set +e
  run_headless "$run_out"
  code=$?
  set -e
  echo "$code" > "$run_out/exit_code.txt"
  publish_worker_session "$run_out" "${WORKER_IN:-}" "tdc_$label"
  write_influx_metrics "$run_out/metrics.json" "tdc_$label"
done
write_summary_json "$BASE_OUT" "$OUT/tdc_sweep_metrics.json"
echo "summary.json: $OUT/tdc_sweep_metrics.json"
