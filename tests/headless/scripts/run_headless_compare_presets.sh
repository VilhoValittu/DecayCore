#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
parse_common_args "$@"
BASE_OUT="$OUT/compare"
mkdir -p "$BASE_OUT"
declare -a presets=(safe_basic default_auto aggressive_bass phase_focus mixed_phase)
for preset in "${presets[@]}"; do
  run_out="$BASE_OUT/$preset"
  case "$preset" in
    safe_basic) write_base_config "$run_out" "$LEFT_WAV" "$RIGHT_WAV" "$preset" "Harman6" 20 200 4 12 "Asymmetric" 65536 300 true 25 0.25 48 ;;
    default_auto) write_base_config "$run_out" "$LEFT_WAV" "$RIGHT_WAV" "$preset" "Harman10" 20 250 5 15 "Asymmetric" 65536 400 true 50 0.05 82 ;;
    aggressive_bass) write_base_config "$run_out" "$LEFT_WAV" "$RIGHT_WAV" "$preset" "Harman10" 15 300 6 18 "Asymmetric" 65536 500 true 75 0.05 120 ;;
    phase_focus) write_base_config "$run_out" "$LEFT_WAV" "$RIGHT_WAV" "$preset" "Harman10" 20 250 5 15 "Linear Phase" 65536 600 true 50 0.05 82 ;;
    mixed_phase) write_base_config "$run_out" "$LEFT_WAV" "$RIGHT_WAV" "$preset" "Harman10" 20 250 5 15 "Mixed" 65536 600 true 50 0.05 82 ;;
  esac
  set +e
  run_headless "$run_out"
  code=$?
  set -e
  echo "$code" > "$run_out/exit_code.txt"
  publish_worker_session "$run_out" "${WORKER_IN:-}" "compare_$preset"
  write_influx_metrics "$run_out/metrics.json" "compare_$preset"
done
write_summary_json "$BASE_OUT" "$OUT/comparison_metrics.json"
echo "comparison_metrics.json: $OUT/comparison_metrics.json"
