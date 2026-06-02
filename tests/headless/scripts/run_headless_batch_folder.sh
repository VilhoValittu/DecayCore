#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
INPUT_ROOT=""
OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT_ROOT="$2"; shift 2 ;;
    --output) OUT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $(basename "$0") --input DIR [--output DIR]"
      echo "Each child session directory must contain config.json."
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -n "$INPUT_ROOT" ]] || { echo "--input is required" >&2; exit 1; }
OUT="${OUT:-$DEFAULT_OUTPUT_ROOT/batch_folder}"
mkdir -p "$OUT"
printf "%-32s %-8s %-10s %-10s %-14s %-15s %-10s\n" session status runtime score mag_error_rms residual_peaks max_boost
for session_dir in "$INPUT_ROOT"/*; do
  [[ -d "$session_dir" && -f "$session_dir/config.json" ]] || continue
  session="$(basename "$session_dir")"
  run_out="$OUT/$session"
  mkdir -p "$run_out"
  cp "$session_dir/config.json" "$run_out/config.json"
  set +e
  run_headless "$run_out"
  code=$?
  set -e
  metrics="$run_out/metrics.json"
  status="$(read_metric_from_json "$metrics" status)"
  runtime="$(read_metric_from_json "$metrics" runtime_s)"
  score="$(read_metric_from_json "$metrics" auto_score)"
  rms="$(read_metric_from_json "$metrics" mag_error_rms_db)"
  peaks="$(read_metric_from_json "$metrics" residual_peak_count)"
  boost="$(read_metric_from_json "$metrics" max_boost_db)"
  [[ -n "$status" ]] || status="exit_$code"
  printf "%-32s %-8s %-10s %-10s %-14s %-15s %-10s\n" "$session" "$status" "$runtime" "$score" "$rms" "$peaks" "$boost"
  write_influx_metrics "$metrics" "$session"
done
write_summary_json "$OUT" "$OUT/batch_summary.json"
echo "summary.json: $OUT/batch_summary.json"
