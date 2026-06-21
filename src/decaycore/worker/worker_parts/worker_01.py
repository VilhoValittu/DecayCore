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

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def log_line(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{utc_now()} {message}\n")

def to_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        val = float(value)
        if math.isfinite(val):
            return int(value) if isinstance(value, int) else val
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
        if match:
            try:
                val = float(match.group(0))
                if math.isfinite(val):
                    return val
            except ValueError:
                return None
    return None

def normalize_label(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[%()/:\[\],=]+", " ", text)
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_metric_key(key: str) -> str:
    raw = str(key or "").strip()
    if not raw:
        return ""
    norm = raw.lower()
    norm = norm.replace("%", "percent")
    norm = re.sub(r"[^a-z0-9]+", "_", norm).strip("_")
    norm = re.sub(r"_+", "_", norm)
    if norm.startswith("metrics_"):
        stripped = norm.removeprefix("metrics_")
        return METRIC_ALIASES.get(stripped, stripped)
    return METRIC_ALIASES.get(norm, norm)

def normalize_key_part(key: str) -> str:
    raw = str(key or "").strip().lower().replace("%", "percent")
    norm = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return re.sub(r"_+", "_", norm)

def flatten_metrics(data: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            key_norm = normalize_key_part(str(key))
            joined = f"{prefix}_{key_norm}" if prefix else key_norm
            out.update(flatten_metrics(value, joined))
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            if isinstance(value, (dict, list)):
                out.update(flatten_metrics(value, f"{prefix}_{idx}" if prefix else str(idx)))
    elif prefix:
        out[normalize_metric_key(prefix)] = data
    return out

def split_metrics_and_tags(raw: dict[str, Any]) -> tuple[dict[str, float | int], dict[str, str]]:
    metrics: dict[str, float | int] = {}
    tags: dict[str, str] = {}
    for key, value in raw.items():
        norm = normalize_metric_key(key)
        if not norm:
            continue
        if norm in LOW_CARDINALITY_TAGS and value is not None:
            text = str(value).strip()
            if text:
                tags[norm] = text[:128]
            continue
        number = to_number(value)
        if number is not None:
            metrics[norm] = number
    return metrics, tags


__all__ = ['utc_now', 'log_line', 'to_number', 'normalize_label', 'normalize_metric_key', 'normalize_key_part', 'flatten_metrics', 'split_metrics_and_tags']


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
