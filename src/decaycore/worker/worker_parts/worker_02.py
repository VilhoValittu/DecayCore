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

def parse_json_file(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def parse_json_metrics(path: Path) -> dict[str, Any]:
    data = parse_json_file(path)
    if data is None:
        return {}
    flattened = flatten_metrics(data)
    return {normalize_metric_key(k): v for k, v in flattened.items()}

def parse_summary_text(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            label, value = line.split(":", 1)
        elif "=" in line:
            label, value = line.split("=", 1)
        else:
            continue
        label_norm = normalize_label(label)
        if "mag error rms max" in label_norm or "magnitude error rms max" in label_norm:
            numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value.replace(",", "."))
            if numbers:
                result["mag_error_rms_db"] = float(numbers[0])
            if len(numbers) > 1:
                result["mag_error_max_db"] = float(numbers[1])
            continue
        key = SUMMARY_KEY_ALIASES.get(label_norm)
        if key is None:
            for alias, mapped in SUMMARY_KEY_ALIASES.items():
                if alias in label_norm:
                    key = mapped
                    break
        if not key:
            continue
        if key in LOW_CARDINALITY_TAGS:
            cleaned = value.strip()
            if cleaned:
                result[key] = cleaned
            continue
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "enabled", "on"}:
            result[key] = 1
        elif lowered in {"false", "no", "disabled", "off"}:
            result[key] = 0
        else:
            number = to_number(value)
            if number is not None:
                result[key] = number
    return result

def discover_summary_files(session_dir: Path, output_dir: Path) -> list[Path]:
    files: list[Path] = []
    for base in (output_dir, session_dir):
        candidate = base / "summary.txt"
        if candidate.is_file():
            files.append(candidate)
        if base.is_dir():
            files.extend(path for path in base.rglob("summary.txt") if path.is_file())
    return sorted(set(files))

def discover_json_files(session_dir: Path, output_dir: Path) -> list[Path]:
    names = {
        "metrics.json",
        "metadata.json",
        "measurement_metadata.json",
        "rt60.json",
        "harmonics.json",
    }
    files: list[Path] = []
    for base in (output_dir, session_dir):
        if not base.is_dir():
            continue
        for path in base.rglob("*.json"):
            if path.name in names:
                files.append(path)
    return sorted(set(files))

def band_key_from_text(key: str) -> str | None:
    text = str(key or "").strip().lower().replace("hz", "")
    text = text.replace("_", "").replace("-", "").replace(" ", "")
    text = text.removeprefix("rt60")
    return RT60_BANDS.get(text)

def extract_rt60(data: Any) -> dict[str, float]:
    found: dict[str, float] = {}

    def walk(value: Any, parent_key: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key)
                direct = band_key_from_text(key_text)
                if direct is not None:
                    number = to_number(item)
                    if number is not None:
                        found[direct] = float(number)
                elif "rt60" in normalize_metric_key(parent_key) or "rt60" in normalize_metric_key(key_text):
                    mapped = band_key_from_text(key_text)
                    if mapped:
                        number = to_number(item)
                        if number is not None:
                            found[mapped] = float(number)
                walk(item, key_text)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    freq = item.get("freq", item.get("frequency", item.get("hz")))
                    rt = item.get("rt60", item.get("value", item.get("seconds")))
                    mapped = band_key_from_text(str(freq))
                    number = to_number(rt)
                    if mapped and number is not None:
                        found[mapped] = float(number)
                walk(item, parent_key)

    walk(data)
    bass_vals = [found[k] for k in ("rt60_31", "rt60_63", "rt60_125", "rt60_250") if k in found]
    mid_vals = [found[k] for k in ("rt60_500", "rt60_1000", "rt60_2000") if k in found]
    treble_vals = [found[k] for k in ("rt60_4000", "rt60_8000") if k in found]
    if bass_vals:
        found["rt60_bass_avg"] = sum(bass_vals) / len(bass_vals)
    if mid_vals:
        found["rt60_mid_avg"] = sum(mid_vals) / len(mid_vals)
    if treble_vals:
        found["rt60_treble_avg"] = sum(treble_vals) / len(treble_vals)
    if found.get("rt60_mid_avg", 0) > 0 and "rt60_bass_avg" in found:
        found["rt60_bass_to_mid_ratio"] = found["rt60_bass_avg"] / found["rt60_mid_avg"]
    return found

def harmonic_risk_from_value(value: Any, key_hint: str = "") -> float | None:
    number = to_number(value)
    if number is None:
        return None
    hint = key_hint.lower()
    if "percent" in hint or "pct" in hint or number >= 0:
        return max(0.0, float(number))
    return max(0.0, 60.0 + float(number))


__all__ = ['parse_json_file', 'parse_json_metrics', 'parse_summary_text', 'discover_summary_files', 'discover_json_files', 'band_key_from_text', 'extract_rt60', 'harmonic_risk_from_value']


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
