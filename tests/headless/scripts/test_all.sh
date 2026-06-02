#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

OUT=""
WORKER_IN="$REPO_ROOT/src/decaycore/worker/data/in"
LEFT_WAV="$DEFAULT_LEFT"
RIGHT_WAV="$DEFAULT_RIGHT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --left) LEFT_WAV="$2"; shift 2 ;;
    --right) RIGHT_WAV="$2"; shift 2 ;;
    --output) OUT="$2"; shift 2 ;;
    --worker-in|--publish-worker-in) WORKER_IN="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [--left PATH --right PATH --output DIR] [--worker-in DIR]

Runs all headless scripts once and prints a pass/fail summary.
EOF
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$OUT" ]]; then
  OUT="$DEFAULT_OUTPUT_ROOT/test_all"
fi
LEFT_WAV="$(resolve_user_path "$LEFT_WAV")"
RIGHT_WAV="$(resolve_user_path "$RIGHT_WAV")"
if [[ -n "$WORKER_IN" ]]; then WORKER_IN="$(resolve_user_path "$WORKER_IN")"; fi

mkdir -p "$OUT"

run_script() {
  local name="$1"; shift
  local run_out="$OUT/$name"
  local code=0
  set +e
  bash "$SCRIPT_DIR/$name.sh" "$@" --output "$run_out" > "$OUT/${name}.log" 2>&1
  code=$?
  set -e
  if [[ "$code" -eq 0 ]]; then
    echo "PASS  $name"
  else
    echo "FAIL  $name  (exit $code)"
  fi
  return "$code"
}

declare -a PASS=()
declare -a FAIL=()

try_script() {
  local name="$1"; shift
  local code=0
  run_script "$name" "$@" || code=$?
  if [[ "$code" -eq 0 ]]; then
    PASS+=("$name")
  else
    FAIL+=("$name")
  fi
}

COMMON_ARGS=(--left "$LEFT_WAV" --right "$RIGHT_WAV")
if [[ -n "$WORKER_IN" ]]; then COMMON_ARGS+=(--worker-in "$WORKER_IN"); fi

BASS_ARGS=(
  --main-left "$DEFAULT_BI_MAIN_LEFT"
  --main-right "$DEFAULT_BI_MAIN_RIGHT"
  --sub-left "$DEFAULT_BI_SUB_LEFT"
  --sub-right "$DEFAULT_BI_SUB_RIGHT"
)
if [[ -n "$WORKER_IN" ]]; then BASS_ARGS+=(--worker-in "$WORKER_IN"); fi

echo "=========================================="
echo "DecayCore headless test_all"
echo "Output: $OUT"
echo "=========================================="

try_script run_headless_basic           "${COMMON_ARGS[@]}"
try_script run_headless_safe_basic      "${COMMON_ARGS[@]}"
try_script run_headless_minimum_phase   "${COMMON_ARGS[@]}"
try_script run_headless_linear_phase    "${COMMON_ARGS[@]}"
try_script run_headless_mixed_phase     "${COMMON_ARGS[@]}"
try_script run_headless_phase_focus     "${COMMON_ARGS[@]}"
try_script run_headless_aggressive_bass "${COMMON_ARGS[@]}"
try_script run_headless_rt60_harmonics  "${COMMON_ARGS[@]}"
try_script run_headless_compare_presets "${COMMON_ARGS[@]}"
try_script run_headless_optuna_sweep    "${COMMON_ARGS[@]}"
try_script run_headless_tdc_sweep       "${COMMON_ARGS[@]}"
try_script run_headless_confidence_sweep "${COMMON_ARGS[@]}"
try_script run_headless_gd_sweep        "${COMMON_ARGS[@]}"
try_script run_headless_residual_peaks  "${COMMON_ARGS[@]}"

# batch_folder: use the basic run output as input
BATCH_INPUT="$OUT/run_headless_basic"
if [[ -d "$BATCH_INPUT" && -f "$BATCH_INPUT/config.json" ]]; then
  BATCH_ARGS=(--input "$OUT" --output "$OUT/run_headless_batch_folder")
  if [[ -n "$WORKER_IN" ]]; then BATCH_ARGS+=(--worker-in "$WORKER_IN"); fi
  name="run_headless_batch_folder"
  code=0
  set +e
  bash "$SCRIPT_DIR/$name.sh" "${BATCH_ARGS[@]}" > "$OUT/${name}.log" 2>&1
  code=$?
  set -e
  if [[ "$code" -eq 0 ]]; then
    echo "PASS  $name"
    PASS+=("$name")
  else
    echo "FAIL  $name  (exit $code)"
    FAIL+=("$name")
  fi
else
  echo "SKIP  run_headless_batch_folder  (no basic output to use as input)"
fi

echo "=========================================="
echo "Results: ${#PASS[@]} passed, ${#FAIL[@]} failed"
if [[ "${#FAIL[@]}" -gt 0 ]]; then
  echo "Failed:"
  for f in "${FAIL[@]}"; do
    echo "  $f  (log: $OUT/${f}.log)"
  done
fi
echo "=========================================="

python "$REPO_ROOT/tests/headless/compare_runs.py" --summary "$OUT" || true

[[ "${#FAIL[@]}" -eq 0 ]]
