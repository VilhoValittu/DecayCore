#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
parse_common_args "$@"
BASE_OUT="$OUT/residual_peaks"
# Sweep max_boost to observe how residual_peak_count and residual_peak_max_db
# change as the correction headroom increases.
for label_boost in 02:2 05:5 08:8 12:12; do
  label="${label_boost%%:*}"
  boost="${label_boost##*:}"
  run_out="$BASE_OUT/boost_$label"
  write_base_config "$run_out" "$LEFT_WAV" "$RIGHT_WAV" "boost_$label" "Harman10" 20 250 "$boost" 15 "Asymmetric" 65536 400 true 50 0.05 82
  set +e
  run_headless "$run_out"
  code=$?
  set -e
  echo "$code" > "$run_out/exit_code.txt"
  publish_worker_session "$run_out" "${WORKER_IN:-}" "residual_boost_$label"
  write_influx_metrics "$run_out/metrics.json" "residual_boost_$label"
done
write_summary_json "$BASE_OUT" "$OUT/residual_peaks_metrics.json"
echo "summary.json: $OUT/residual_peaks_metrics.json"
