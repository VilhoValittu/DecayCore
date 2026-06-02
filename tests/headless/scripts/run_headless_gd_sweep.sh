#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
parse_common_args "$@"
BASE_OUT="$OUT/gd_sweep"
# Sweep phase_limit to observe gd_max_ms and gd_spike_count behaviour.
# Tighter limits produce less phase correction; looser limits allow more.
for label_limit in 100:100 200:200 400:400 800:800; do
  label="${label_limit%%:*}"
  limit="${label_limit##*:}"
  run_out="$BASE_OUT/gd_phase_$label"
  write_base_config "$run_out" "$LEFT_WAV" "$RIGHT_WAV" "gd_phase_$label" "Harman10" 20 250 5 15 "Asymmetric" 65536 "$limit" true 50 0.05 82
  set +e
  run_headless "$run_out"
  code=$?
  set -e
  echo "$code" > "$run_out/exit_code.txt"
  publish_worker_session "$run_out" "${WORKER_IN:-}" "gd_phase_$label"
  write_influx_metrics "$run_out/metrics.json" "gd_phase_$label"
done
write_summary_json "$BASE_OUT" "$OUT/gd_sweep_metrics.json"
echo "summary.json: $OUT/gd_sweep_metrics.json"
