#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
MEASUREMENT_METADATA=""
RT60_JSON=""
HARMONICS_JSON=""
LEFT_WAV="$DEFAULT_LEFT"
RIGHT_WAV="$DEFAULT_RIGHT"
OUT=""
WORKER_IN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --left) LEFT_WAV="$2"; shift 2 ;;
    --right) RIGHT_WAV="$2"; shift 2 ;;
    --output) OUT="$2"; shift 2 ;;
    --measurement-metadata) MEASUREMENT_METADATA="$2"; shift 2 ;;
    --rt60) RT60_JSON="$2"; shift 2 ;;
    --harmonics) HARMONICS_JSON="$2"; shift 2 ;;
    --worker-in|--publish-worker-in) WORKER_IN="$2"; shift 2 ;;
    -h|--help)
      usage_common "$(basename "$0")"
      echo "  --measurement-metadata PATH"
      echo "  --rt60 PATH"
      echo "  --harmonics PATH"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
OUT="${OUT:-$DEFAULT_OUTPUT_ROOT/run_headless_rt60_harmonics}"
LEFT_WAV="$(resolve_user_path "$LEFT_WAV")"
RIGHT_WAV="$(resolve_user_path "$RIGHT_WAV")"
[[ -n "$MEASUREMENT_METADATA" ]] && MEASUREMENT_METADATA="$(resolve_user_path "$MEASUREMENT_METADATA")"
[[ -n "$RT60_JSON" ]] && RT60_JSON="$(resolve_user_path "$RT60_JSON")"
[[ -n "$HARMONICS_JSON" ]] && HARMONICS_JSON="$(resolve_user_path "$HARMONICS_JSON")"
[[ -n "$WORKER_IN" ]] && WORKER_IN="$(resolve_user_path "$WORKER_IN")"
write_base_config "$OUT" "$LEFT_WAV" "$RIGHT_WAV" "rt60_harmonics" "Harman8" 20 250 4 12 "Asymmetric" 65536 350 true 35 0.20 82
if [[ -z "$MEASUREMENT_METADATA" && -f "${LEFT_WAV%__ir.wav}__metadata.json" ]]; then
  MEASUREMENT_METADATA="${LEFT_WAV%__ir.wav}__metadata.json"
fi
[[ -n "$MEASUREMENT_METADATA" && -f "$MEASUREMENT_METADATA" ]] && cp "$MEASUREMENT_METADATA" "$OUT/measurement_metadata.json"
[[ -n "$RT60_JSON" && -f "$RT60_JSON" ]] && cp "$RT60_JSON" "$OUT/rt60.json"
[[ -n "$HARMONICS_JSON" && -f "$HARMONICS_JSON" ]] && cp "$HARMONICS_JSON" "$OUT/harmonics.json"
python - "$OUT/config.json" "$MEASUREMENT_METADATA" "$RT60_JSON" "$HARMONICS_JSON" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
d["metadata"] = {"measurement_metadata": sys.argv[2], "rt60": sys.argv[3], "harmonics": sys.argv[4]}
d["exports"] = ["rt60_*", "harmonic_risk_*", "boost_risk_*"]
json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
PY
run_headless_publish_exit "$OUT" "${WORKER_IN:-}" "$(basename "$OUT")"
