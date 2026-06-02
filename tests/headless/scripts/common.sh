#!/usr/bin/env bash
set -euo pipefail

resolve_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  (cd "$script_dir/../../.." && pwd)
}

REPO_ROOT="$(resolve_repo_root)"
CALLER_CWD="$(pwd)"

export INFLUX_URL="${INFLUX_URL:-http://localhost:8086}"
export INFLUX_TOKEN="${INFLUX_TOKEN:-supertoken}"
export INFLUX_ORG="${INFLUX_ORG:-camillafir}"
export INFLUX_BUCKET="${INFLUX_BUCKET:-metrics}"

resolve_user_path() {
  local value="$1"
  if [[ -z "$value" ]]; then
    return 0
  fi
  case "$value" in
    /*|[A-Za-z]:*|\\\\*) printf '%s\n' "$value" ;;
    *) (cd "$CALLER_CWD" && realpath -m "$value") || (cd "$CALLER_CWD" && python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$value") ;;
  esac
}

path_for_python() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -w "$1"
  else
    printf '%s\n' "$1"
  fi
}

PYTHON_SRC="$(path_for_python "$REPO_ROOT/src")"
if command -v cygpath >/dev/null 2>&1; then
  PYTHONPATH_SEP=";"
else
  PYTHONPATH_SEP=":"
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$PYTHON_SRC$PYTHONPATH_SEP$PYTHONPATH"
else
  export PYTHONPATH="$PYTHON_SRC"
fi
DEFAULT_OUTPUT_ROOT="${CAMILLAFIR_HEADLESS_OUTPUT_ROOT:-/tmp/camillafir-headless-runs}"
DEFAULT_LEFT="$REPO_ROOT/tests/headless/speaker_data/measurement_left__20260422_154122__ir.wav"
DEFAULT_RIGHT="$REPO_ROOT/tests/headless/speaker_data/measurement_right__20260422_154153__ir.wav"
DEFAULT_BI_MAIN_LEFT="$REPO_ROOT/tests/headless/speaker_data/subs_inc/left_final__ir.wav"
DEFAULT_BI_MAIN_RIGHT="$REPO_ROOT/tests/headless/speaker_data/subs_inc/right_final__ir.wav"
DEFAULT_BI_SUB_LEFT="$REPO_ROOT/tests/headless/speaker_data/subs_inc/sub1_final__ir.wav"
DEFAULT_BI_SUB_RIGHT="$REPO_ROOT/tests/headless/speaker_data/subs_inc/sub2_final__ir.wav"

usage_common() {
  local name="$1"
  cat <<EOF
Usage: $name [--left PATH --right PATH --output DIR] [--worker-in DIR]

Defaults:
  --left   $DEFAULT_LEFT
  --right  $DEFAULT_RIGHT
  --output $DEFAULT_OUTPUT_ROOT/<script-name>

Use --worker-in src/decaycore/worker/data/in to publish metrics for Grafana.
EOF
}

usage_bass() {
  local name="$1"
  cat <<EOF
Usage: $name [--main-left PATH --main-right PATH --sub-left PATH --sub-right PATH --output DIR] [--worker-in DIR]

Defaults:
  --main-left  $DEFAULT_BI_MAIN_LEFT
  --main-right $DEFAULT_BI_MAIN_RIGHT
  --sub-left   $DEFAULT_BI_SUB_LEFT
  --sub-right  $DEFAULT_BI_SUB_RIGHT
  --output     $DEFAULT_OUTPUT_ROOT/<script-name>

Use --worker-in src/decaycore/worker/data/in to publish metrics for Grafana.
EOF
}

parse_common_args() {
  LEFT_WAV="$DEFAULT_LEFT"
  RIGHT_WAV="$DEFAULT_RIGHT"
  OUT="${OUT:-}"
  WORKER_IN="${WORKER_IN:-}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --left) LEFT_WAV="$2"; shift 2 ;;
      --right) RIGHT_WAV="$2"; shift 2 ;;
      --output) OUT="$2"; shift 2 ;;
      --worker-in|--publish-worker-in) WORKER_IN="$2"; shift 2 ;;
      -h|--help) usage_common "$(basename "$0")"; exit 0 ;;
      *) echo "Unknown argument: $1" >&2; usage_common "$(basename "$0")"; exit 1 ;;
    esac
  done
  if [[ -z "${OUT:-}" ]]; then
    OUT="$DEFAULT_OUTPUT_ROOT/$(basename "$0" .sh)"
  fi
  LEFT_WAV="$(resolve_user_path "$LEFT_WAV")"
  RIGHT_WAV="$(resolve_user_path "$RIGHT_WAV")"
  if [[ -n "$WORKER_IN" ]]; then WORKER_IN="$(resolve_user_path "$WORKER_IN")"; fi
}

parse_bass_args() {
  MAIN_LEFT="$DEFAULT_BI_MAIN_LEFT"
  MAIN_RIGHT="$DEFAULT_BI_MAIN_RIGHT"
  SUB_LEFT="$DEFAULT_BI_SUB_LEFT"
  SUB_RIGHT="$DEFAULT_BI_SUB_RIGHT"
  OUT="${OUT:-}"
  WORKER_IN="${WORKER_IN:-}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --main-left) MAIN_LEFT="$2"; shift 2 ;;
      --main-right) MAIN_RIGHT="$2"; shift 2 ;;
      --sub-left) SUB_LEFT="$2"; shift 2 ;;
      --sub-right) SUB_RIGHT="$2"; shift 2 ;;
      --output) OUT="$2"; shift 2 ;;
      --worker-in|--publish-worker-in) WORKER_IN="$2"; shift 2 ;;
      -h|--help) usage_bass "$(basename "$0")"; exit 0 ;;
      *) echo "Unknown argument: $1" >&2; usage_bass "$(basename "$0")"; exit 1 ;;
    esac
  done
  if [[ -z "${OUT:-}" ]]; then
    OUT="$DEFAULT_OUTPUT_ROOT/$(basename "$0" .sh)"
  fi
  MAIN_LEFT="$(resolve_user_path "$MAIN_LEFT")"
  MAIN_RIGHT="$(resolve_user_path "$MAIN_RIGHT")"
  SUB_LEFT="$(resolve_user_path "$SUB_LEFT")"
  SUB_RIGHT="$(resolve_user_path "$SUB_RIGHT")"
  if [[ -n "$WORKER_IN" ]]; then WORKER_IN="$(resolve_user_path "$WORKER_IN")"; fi
}

json_escape() {
  python - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

write_base_config() {
  local out="$1" left="$2" right="$3" preset="$4" house="$5" min_hz="$6" max_hz="$7" boost="$8" cut_abs="$9"
  local filter_type="${10:-Asymmetric}" taps="${11:-65536}" phase_limit="${12:-400}" tdc_enabled="${13:-true}" tdc_strength="${14:-50}" conf_pull="${15:-0.05}" optuna="${16:-82}"
  local left_py right_py
  left_py="$(path_for_python "$left")"
  right_py="$(path_for_python "$right")"
  mkdir -p "$out"
  cat > "$out/config.json" <<JSON
{
  "mode": "auto",
  "left_wav": $(json_escape "$left_py"),
  "right_wav": $(json_escape "$right_py"),
  "sample_rate": 48000,
  "house_curve": "$house",
  "correction_min_hz": $min_hz,
  "correction_max_hz": $max_hz,
  "mag_c_min": $min_hz,
  "mag_c_max": $max_hz,
  "max_boost": $boost,
  "max_cut_db": $cut_abs,
  "filter_type": "$filter_type",
  "taps": $taps,
  "phase_limit": $phase_limit,
  "enable_tdc": $tdc_enabled,
  "tdc_strength": $tdc_strength,
  "tdc_max_reduction_db": 9,
  "enable_afdw": true,
  "fdw_cycles": 14,
  "conf_pull_floor": $conf_pull,
  "auto_mode_trials": $optuna,
  "auto_options": {
    "preset": "$preset",
    "optuna_trials": $optuna
  },
  "correction": {
    "min_hz": $min_hz,
    "max_hz": $max_hz,
    "max_boost_db": $boost,
    "max_cut_db": -$cut_abs
  },
  "target": {
    "house_curve": "$house"
  },
  "tdc": {
    "enabled": $tdc_enabled,
    "strength": $tdc_strength,
    "max_reduction_db": 9
  },
  "confidence": {
    "enabled": true,
    "pull": $conf_pull
  },
  "bass_integration": {
    "enabled": false
  }
}
JSON
}

write_influx_metrics() {
  local metrics_path="$1"
  local session="$2"

  if [[ "${INFLUX_WRITE:-1}" == "0" ]]; then
    echo "InfluxDB write skipped for ${session}: INFLUX_WRITE=0."
    return 0
  fi

  CAMILLAFIR_SESSION="${session}" python - "${metrics_path}" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:
    print("InfluxDB write skipped: influxdb_client not installed.", file=sys.stderr)
    sys.exit(0)

metrics_path = Path(sys.argv[1])
doc = json.loads(metrics_path.read_text(encoding="utf-8"))


def normalize_key(value):
    out = []
    last_sep = False
    for ch in str(value or "").strip().lower():
        if ch.isalnum():
            out.append(ch)
            last_sep = False
        elif not last_sep:
            out.append("_")
            last_sep = True
    return "".join(out).strip("_")


def as_number(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            key_norm = normalize_key(key)
            joined = f"{prefix}_{key_norm}" if prefix else key_norm
            yield from flatten(item, joined)
    elif not isinstance(value, list) and prefix:
        yield prefix, value


session = str(os.environ.get("CAMILLAFIR_SESSION", metrics_path.parent.name) or metrics_path.parent.name)
status = str(doc.get("status", "unknown") or "unknown")
point = Point("decaycore").tag("session", session).tag("status", status)

for tag_key in ("target_curve_mode", "selected_house_curve"):
    value = str(doc.get(tag_key, "") or "").strip()
    if value:
        point = point.tag(tag_key, value[:128])

for key, value in flatten(doc.get("metrics", {}) or {}):
    if key in {"bi_mode", "bi_polarity"} and value is not None:
        point = point.tag(key, str(value)[:128])
        continue
    number = as_number(value)
    if number is not None:
        point = point.field(key, number)

for key, value in flatten(doc.get("rt60", {}) or {}, "rt60"):
    number = as_number(value)
    if number is not None:
        point = point.field(key, number)

for key, value in flatten(doc.get("harmonics", {}) or {}, "harmonic"):
    number = as_number(value)
    if number is not None:
        point = point.field(key, number)

runtime_s = as_number(doc.get("runtime_s"))
if runtime_s is not None:
    point = point.field("runtime_s", runtime_s)

client = InfluxDBClient(
    url=os.environ["INFLUX_URL"],
    token=os.environ["INFLUX_TOKEN"],
    org=os.environ["INFLUX_ORG"],
)
try:
    client.write_api(write_options=SYNCHRONOUS).write(bucket=os.environ["INFLUX_BUCKET"], record=point)
finally:
    client.close()

print(f"InfluxDB write ok: session={session} status={status}")
PY
}

run_headless() {
  local out="$1"
  local config="$out/config.json"
  local had_errexit=0
  [[ $- == *e* ]] && had_errexit=1
  local cmd=(python -m decaycore.headless --config "$config" --output "$out" --mode auto --no-plots --no-png --write-summary --write-metrics)
  echo "Output: $out"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  echo
  set +e
  "${cmd[@]}"
  local status=$?
  if [[ "$had_errexit" == "1" ]]; then
    set -e
  fi
  print_result_paths "$out"
  return "$status"
}

print_result_paths() {
  local out="$1"
  echo "metrics.json: $out/metrics.json"
  echo "summary.txt: $out/summary.txt"
  echo "run.log: $out/run.log"
}

publish_worker_session() {
  local out="$1" worker_in="${2:-}" session="${3:-}"
  [[ -n "$worker_in" ]] || return 0
  [[ -f "$out/metrics.json" ]] || return 0
  session="${session:-$(basename "$out")}"
  local dest="$worker_in/$session"
  mkdir -p "$dest"
  rm -f "$dest/.processing" "$dest/.done" "$dest/.failed"
  cp "$out/metrics.json" "$dest/metrics.json"
  if [[ -f "$out/summary.txt" ]]; then cp "$out/summary.txt" "$dest/summary.txt"; fi
  if [[ -f "$out/run.log" ]]; then cp "$out/run.log" "$dest/run.log"; fi
  echo "worker session: $dest"
}

run_headless_publish_exit() {
  local out="$1" worker_in="${2:-}" session="${3:-}"
  session="${session:-$(basename "$out")}"
  set +e
  run_headless "$out"
  local status=$?
  set -e
  publish_worker_session "$out" "$worker_in" "$session"
  write_influx_metrics "$out/metrics.json" "$session"
  exit "$status"
}

read_metric_from_json() {
  local file="$1" key="$2"
  if command -v jq >/dev/null 2>&1; then
    jq -r --arg key "$key" '.metrics[$key] // .[$key] // ""' "$file" 2>/dev/null || true
  else
    python - "$file" "$key" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    print((data.get("metrics") or {}).get(sys.argv[2], data.get(sys.argv[2], "")))
except Exception:
    print("")
PY
  fi
}

write_summary_json() {
  local root="$1" output_file="$2"
  python - "$root" "$output_file" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = {}
for metrics_path in sorted(root.glob("*/metrics.json")):
    try:
        doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        rows[metrics_path.parent.name] = {"status": "failed", "error": str(exc)}
        continue
    m = doc.get("metrics") or {}
    rows[metrics_path.parent.name] = {
        "status": doc.get("status"),
        "runtime_s": doc.get("runtime_s"),
        "auto_score": m.get("auto_score"),
        "filter_score": m.get("filter_score"),
        "mag_error_rms_db": m.get("mag_error_rms_db"),
        "mag_error_max_db": m.get("mag_error_max_db"),
        "residual_peak_count": m.get("residual_peak_count"),
        "residual_peak_max_db": m.get("residual_peak_max_db"),
        "gd_spike_count": m.get("gd_spike_count"),
        "gd_max_ms": m.get("gd_max_ms"),
        "max_boost_db": m.get("max_boost_db"),
        "max_cut_db": m.get("max_cut_db"),
        "bass_ripple_20_200_db": m.get("bass_ripple_20_200_db"),
        "harmonic_risk_max": m.get("harmonic_risk_max"),
        "optuna_trials": m.get("optuna_trials"),
    }
pathlib.Path(sys.argv[2]).write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
PY
}
