#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
parse_common_args "$@"
BASE_OUT="$OUT/optuna_sweep"
for trials in 24 48 82 120; do
  run_out="$BASE_OUT/trials_$trials"
  write_base_config "$run_out" "$LEFT_WAV" "$RIGHT_WAV" "optuna_trials_$trials" "Harman10" 20 250 5 15 "Asymmetric" 65536 400 true 50 0.05 "$trials"
  set +e
  run_headless "$run_out"
  code=$?
  set -e
  echo "$code" > "$run_out/exit_code.txt"
  publish_worker_session "$run_out" "${WORKER_IN:-}" "optuna_trials_$trials"
  write_influx_metrics "$run_out/metrics.json" "optuna_trials_$trials"
done
write_summary_json "$BASE_OUT" "$OUT/optuna_sweep_metrics.json"
echo "summary.json: $OUT/optuna_sweep_metrics.json"
