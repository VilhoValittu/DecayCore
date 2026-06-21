import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IN_DIR = Path("/data/in")
OUT_DIR = Path("/data/out")
SRC_DIR = Path("/src")
POLL_SECONDS = 3
STABLE_SECONDS = 10
PROCESS_TIMEOUT_SECONDS = int(os.environ.get("DECAYCORE_WORKER_TIMEOUT", os.environ.get("CAMILLAFIR_WORKER_TIMEOUT", "1800")) or "1800")

USEFUL_SUFFIXES = {".wav", ".json", ".txt", ".cfg", ".conf", ".yml", ".yaml", ".toml"}
STATE_FILES = {".processing", ".done", ".failed"}
LOW_CARDINALITY_TAGS = {
    "bi_mode",
    "bi_polarity",
    "target_curve_mode",
    "selected_house_curve",
}

RT60_BANDS = {
    "31": "rt60_31",
    "31.5": "rt60_31",
    "63": "rt60_63",
    "125": "rt60_125",
    "250": "rt60_250",
    "500": "rt60_500",
    "1k": "rt60_1000",
    "1000": "rt60_1000",
    "2k": "rt60_2000",
    "2000": "rt60_2000",
    "4k": "rt60_4000",
    "4000": "rt60_4000",
    "8k": "rt60_8000",
    "8000": "rt60_8000",
}

METRIC_ALIASES = {
    "score": "auto_score",
    "auto_score": "auto_score",
    "filter_score": "filter_score",
    "mag_error_rms_db": "mag_error_rms_db",
    "mag_error_max_db": "mag_error_max_db",
    "magnitude_error_rms_db": "mag_error_rms_db",
    "magnitude_error_max_db": "mag_error_max_db",
    "magnitude_rms_db": "mag_error_rms_db",
    "magnitude_max_db": "mag_error_max_db",
    "error_rms_db": "mag_error_rms_db",
    "error_max_db": "mag_error_max_db",
    "response_error_rms_db": "mag_error_rms_db",
    "response_error_max_db": "mag_error_max_db",
    "fr_error_rms_db": "mag_error_rms_db",
    "fr_error_max_db": "mag_error_max_db",
    "mag_rms": "mag_error_rms_db",
    "mag_max": "mag_error_max_db",
    "rms_db": "mag_error_rms_db",
    "max_error_db": "mag_error_max_db",
    "residual_peaks": "residual_peak_count",
    "gd_spikes": "gd_spike_count",
    "max_boost": "max_boost_db",
    "max_cut": "max_cut_db",
    "taps": "filter_taps",
    "trials": "optuna_trials",
    "bass_integration_crossover_hz": "bi_crossover_hz",
    "bass_integration_overlap_ripple_db": "bi_overlap_ripple_db",
    "bass_integration_cancellation_risk": "bi_cancellation_risk",
    "bass_integration_sub_dominance_db": "bi_sub_dominance_db",
    "bass_integration_delay_ms": "bi_delay_ms",
    "bass_integration_gain_db": "bi_gain_db",
    "bass_integration_mode": "bi_mode",
    "bass_integration_polarity": "bi_polarity",
}

SUMMARY_KEY_ALIASES = {
    "runtime": "runtime_s",
    "worker runtime": "runtime_worker_s",
    "decaycore runtime": "camillafir_runtime_s",
    "camillafir runtime": "camillafir_runtime_s",
    "auto score": "auto_score",
    "score": "score",
    "filter score": "filter_score",
    "rank score": "rank_score",
    "dsp penalty": "dsp_penalty",
    "excess penalty": "excess_penalty",
    "max net boost": "max_net_boost_db",
    "event count": "event_count",
    "event severity sum": "event_severity_sum",
    "magnitude error rms": "mag_error_rms_db",
    "mag error rms": "mag_error_rms_db",
    "fr error rms": "mag_error_rms_db",
    "response error rms": "mag_error_rms_db",
    "rms magnitude error": "mag_error_rms_db",
    "magnitude error max": "mag_error_max_db",
    "mag error max": "mag_error_max_db",
    "fr error max": "mag_error_max_db",
    "response error max": "mag_error_max_db",
    "max magnitude error": "mag_error_max_db",
    "magnitude error avg": "mag_error_avg_db",
    "mag error avg": "mag_error_avg_db",
    "bass ripple 20 200": "bass_ripple_20_200_db",
    "residual peak count": "residual_peak_count",
    "residual peak max": "residual_peak_max_db",
    "residual peak mean": "residual_peak_mean_db",
    "max boost": "max_boost_db",
    "max cut": "max_cut_db",
    "max correction": "max_correction_db",
    "net boost": "net_boost_db",
    "gd spike count": "gd_spike_count",
    "group delay spike count": "gd_spike_count",
    "gd max": "gd_max_ms",
    "group delay max": "gd_max_ms",
    "gd gradient max": "gd_gradient_max_ms_per_oct",
    "phase error rms": "phase_error_rms_deg",
    "phase error max": "phase_error_max_deg",
    "phase clamp target": "phase_clamp_target_deg",
    "phase clamp events": "phase_clamp_events",
    "tdc strength": "tdc_strength",
    "tdc max reduction": "tdc_max_reduction_db",
    "tdc events": "tdc_events",
    "confidence pull": "confidence_pull",
    "confidence low band count": "confidence_low_band_count",
    "filter taps": "filter_taps",
    "sample rate": "sample_rate",
    "filter length": "filter_length_s",
    "export peak": "export_peak_dbfs",
    "pre energy ratio": "pre_energy_ratio",
    "post energy ratio": "post_energy_ratio",
    "bi enabled": "bi_enabled",
    "bass integration enabled": "bi_enabled",
    "bi runtime": "bi_runtime_s",
    "bass integration runtime": "bi_runtime_s",
    "bi crossover": "bi_crossover_hz",
    "bass integration crossover": "bi_crossover_hz",
    "bi delay": "bi_delay_ms",
    "bi gain": "bi_gain_db",
    "bi overlap ripple": "bi_overlap_ripple_db",
    "bi cancellation risk": "bi_cancellation_risk",
    "bi sub dominance": "bi_sub_dominance_db",
    "bi xo gd delta": "bi_xo_gd_delta_ms",
    "bi phase delta": "bi_phase_delta_deg",
    "bi main sub corr score": "bi_main_sub_corr_score",
    "optuna trials": "optuna_trials",
    "optuna best value": "optuna_best_value",
    "optuna feasible trials": "optuna_feasible_trials",
    "pareto front size": "pareto_front_size",
    "phase2 pool size": "phase2_pool_size",
    "winner index": "winner_index",
    "winner polish used": "winner_polish_used",
    "bi mode": "bi_mode",
    "bi polarity": "bi_polarity",
    "target curve mode": "target_curve_mode",
    "selected house curve": "selected_house_curve",
}


































































if __name__ == "__main__":
    main()

def extract_harmonic_points(data: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def add(freq: Any, value: Any, key_hint: str = "") -> None:
        freq_num = to_number(freq)
        risk = harmonic_risk_from_value(value, key_hint)
        if freq_num is not None and risk is not None and float(freq_num) > 0:
            points.append((float(freq_num), float(risk)))

    def walk(value: Any, parent_key: str = "") -> None:
        if isinstance(value, dict):
            keys = {str(k).lower() for k in value}
            if {"freq", "frequency", "hz"} & keys:
                freq = value.get("freq", value.get("frequency", value.get("hz")))
                for val_key in ("h2_db", "h3_db", "h4_db", "db", "value", "percent", "pct"):
                    if val_key in value:
                        add(freq, value.get(val_key), val_key)
            for key, item in value.items():
                key_text = str(key)
                parent_norm = normalize_metric_key(parent_key)
                if parent_norm in {"h2", "h3", "h4", "harmonics_h2", "harmonics_h3"}:
                    add(key_text, item, parent_norm)
                elif normalize_metric_key(key_text) in {"h2", "h3", "h4"} and isinstance(item, dict):
                    for freq, mag in item.items():
                        add(freq, mag, key_text)
                elif "harmonic" in parent_norm and isinstance(item, (int, float, str)):
                    add(key_text, item, parent_norm)
                walk(item, key_text)
        elif isinstance(value, list):
            for item in value:
                walk(item, parent_key)

    walk(data)
    return points

def extract_harmonics(data: Any) -> dict[str, float]:
    points = extract_harmonic_points(data)
    if not points:
        return {}
    bands = {
        "20_80": [risk for freq, risk in points if 20 <= freq < 80],
        "80_200": [risk for freq, risk in points if 80 <= freq < 200],
        "200_500": [risk for freq, risk in points if 200 <= freq <= 500],
    }
    out: dict[str, float] = {}
    for band, risks in bands.items():
        if risks:
            risk = max(risks)
            out[f"harmonic_risk_{band}"] = risk
            out[f"boost_risk_{band}"] = risk
    freq_max, risk_max = max(points, key=lambda item: item[1])
    out["harmonic_risk_max"] = risk_max
    out["harmonic_risk_freq_hz"] = freq_max
    return out

def collect_artifact_metrics(session_dir: Path, output_dir: Path) -> tuple[dict[str, float | int], dict[str, str], list[str]]:
    raw: dict[str, Any] = {}
    warnings: list[str] = []
    for summary_path in discover_summary_files(session_dir, output_dir):
        try:
            raw.update(parse_summary_text(summary_path.read_text(encoding="utf-8", errors="replace")))
        except Exception as exc:
            warnings.append(f"Failed to parse summary {summary_path}: {exc}")
    for json_path in discover_json_files(session_dir, output_dir):
        data = parse_json_file(json_path)
        if data is None:
            warnings.append(f"Failed to parse JSON {json_path}")
            continue
        raw.update(parse_json_metrics(json_path))
        raw.update(extract_rt60(data))
        raw.update(extract_harmonics(data))
    return (*split_metrics_and_tags(raw), warnings)

def latest_mtime(path: Path) -> float:
    mtimes = [path.stat().st_mtime]
    for item in path.rglob("*"):
        try:
            mtimes.append(item.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes)

def session_has_useful_input(session_dir: Path) -> bool:
    for path in session_dir.rglob("*"):
        if path.is_file() and path.name not in STATE_FILES and path.suffix.lower() in USEFUL_SUFFIXES:
            return True
    return False

def is_session_ready(session_dir: Path) -> bool:
    if not session_dir.is_dir():
        return False
    if any((session_dir / name).exists() for name in STATE_FILES):
        return False
    if not session_has_useful_input(session_dir):
        return False
    try:
        first = latest_mtime(session_dir)
        if time.time() - first < STABLE_SECONDS:
            return False
        time.sleep(STABLE_SECONDS)
        second = latest_mtime(session_dir)
    except OSError:
        return False
    return first == second

def scan_sessions() -> list[Path]:
    IN_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(path for path in IN_DIR.iterdir() if is_session_ready(path))

def run_camillafir_session(session_dir: Path, output_dir: Path) -> dict[str, Any]:
    log_path = output_dir / "worker.log"
    config_path = session_dir / "config.json"

    result: dict[str, Any] = {
        "attempted": False,
        "success": False,
        "warnings": [],
        "errors": [],
    }

    if not config_path.is_file():
        result["warnings"].append("No config.json found; parsed metadata only.")
        log_line(log_path, "No config.json found; skipping DecayCore execution.")
        return result

    cmd = [
        "python",
        "-m",
        "decaycore.headless",
        "--config",
        str(config_path),
        "--output",
        str(output_dir),
        "--mode",
        "auto",
        "--no-plots",
        "--write-summary",
        "--write-metrics",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(SRC_DIR),
        str(SRC_DIR / "src"),
        env.get("PYTHONPATH", ""),
    ])

    result["attempted"] = True
    log_line(log_path, f"Running headless: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(SRC_DIR),
            env=env,
            text=True,
            capture_output=True,
            timeout=PROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        msg = "Headless run timed out"
        result["errors"].append(msg)
        log_line(log_path, msg)
        return result
    except Exception as exc:
        msg = f"Headless failed to start: {exc}"
        result["errors"].append(msg)
        log_line(log_path, msg)
        return result

    log_line(log_path, f"Return code: {proc.returncode}")

    if proc.stdout:
        log_line(log_path, f"stdout tail: {proc.stdout[-4000:]}")
    if proc.stderr:
        log_line(log_path, f"stderr tail: {proc.stderr[-4000:]}")

    if proc.returncode == 0:
        result["success"] = True
    elif proc.returncode == 2:
        result["warnings"].append("Partial success from headless")
    else:
        result["errors"].append(f"Headless returned {proc.returncode}")

    return result


__all__ = ['extract_harmonic_points', 'extract_harmonics', 'collect_artifact_metrics', 'latest_mtime', 'session_has_useful_input', 'is_session_ready', 'scan_sessions', 'run_camillafir_session']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['worker_01', 'worker_02', 'worker_03', 'worker_04']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
