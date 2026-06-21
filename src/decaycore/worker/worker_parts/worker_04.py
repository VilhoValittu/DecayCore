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

def detect_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(SRC_DIR), "rev-parse", "--short", "HEAD"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        return ""
    return ""

def detect_version_from_text(text: str) -> str:
    match = re.search(r"\bversion\b\s*[:=]?\s*([vV]?[.]?\d+(?:[.\w-]*\d)?)", text or "", re.IGNORECASE)
    return match.group(1).strip() if match else ""

def detect_version(session_dir: Path | None = None, output_dir: Path | None = None) -> str:
    candidates = [
        SRC_DIR / "pyproject.toml",
        SRC_DIR / "__init__.py",
        SRC_DIR / "src" / "decaycore" / "__init__.py",
        SRC_DIR / "src" / "decaycore" / "version.py",
    ]
    if session_dir and output_dir:
        candidates.extend(discover_summary_files(session_dir, output_dir))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for pattern in (
            r"^version\s*=\s*[\"']([^\"']+)[\"']",
            r"VERSION\s*=\s*[\"']([^\"']+)[\"']",
            r"DEFAULT_VERSION\s*=\s*[\"']([^\"']+)[\"']",
        ):
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return match.group(1).strip()
        summary_version = detect_version_from_text(text)
        if summary_version:
            return summary_version
    return ""

def write_influx(session: str, metrics: dict[str, float | int], tags: dict[str, str], status: str) -> None:
    from influxdb_client import InfluxDBClient, Point

    client = InfluxDBClient(
        url=os.environ["INFLUX_URL"],
        token=os.environ["INFLUX_TOKEN"],
        org=os.environ["INFLUX_ORG"],
    )
    try:
        point = Point("decaycore").tag("session", session).tag("status", status)
        for key, value in sorted(tags.items()):
            if value:
                point = point.tag(key, str(value))
        for key, value in sorted(metrics.items()):
            number = to_number(value)
            if number is not None:
                point = point.field(key, number)
        client.write_api().write(bucket=os.environ["INFLUX_BUCKET"], record=point)
    finally:
        client.close()

def write_metrics_json(
    output_dir: Path,
    *,
    session: str,
    status: str,
    started_at: str,
    finished_at: str,
    runtime_worker_s: float,
    version: str,
    git_commit: str,
    tags: dict[str, str],
    metrics: dict[str, float | int],
    warnings: list[str],
    errors: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "session": session,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_worker_s": runtime_worker_s,
        "version": version,
        "git_commit": git_commit,
        "tags": tags,
        "metrics": metrics,
        "warnings": warnings,
        "errors": errors,
    }
    tmp = output_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(output_dir / "metrics.json")

def touch_state(path: Path, text: str = "") -> None:
    path.write_text(text or utc_now(), encoding="utf-8")

def process_session(session_dir: Path) -> dict[str, Any]:
    started = time.time()
    started_at = utc_now()
    output_dir = OUT_DIR / session_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "worker.log"
    processing = session_dir / ".processing"
    done = session_dir / ".done"
    failed = session_dir / ".failed"
    warnings: list[str] = []
    errors: list[str] = []
    metrics: dict[str, float | int] = {}
    tags: dict[str, str] = {}
    status = "failed"
    touch_state(processing)
    log_line(log_path, f"Processing session {session_dir}")
    try:
        run_info = run_camillafir_session(session_dir, output_dir)
        warnings.extend(run_info.get("warnings", []))
        errors.extend(run_info.get("errors", []))
        artifact_metrics, artifact_tags, parse_warnings = collect_artifact_metrics(session_dir, output_dir)
        warnings.extend(parse_warnings)
        metrics.update(artifact_metrics)
        tags.update(artifact_tags)

        useful_metrics = bool(metrics)
        if run_info.get("success"):
            status = "success"
        elif useful_metrics:
            status = "partial"
        else:
            status = "failed"
            if not errors:
                errors.append("No DecayCore invocation succeeded and no useful metrics were extracted.")

        runtime_worker_s = time.time() - started
        metrics["runtime_worker_s"] = runtime_worker_s
        metrics["worker_success"] = 1 if status == "success" else 0
        metrics["worker_partial"] = 1 if status == "partial" else 0
        metrics["worker_failed"] = 1 if status == "failed" else 0

        version = detect_version(session_dir, output_dir)
        git_commit = detect_git_commit()
        if version:
            tags["version"] = version
        if git_commit:
            tags["git_commit"] = git_commit

        try:
            write_influx(session_dir.name, metrics, tags, status)
        except Exception as exc:
            warnings.append(f"InfluxDB write failed: {exc}")
            log_line(log_path, f"InfluxDB write failed: {exc}")

        write_metrics_json(
            output_dir,
            session=session_dir.name,
            status=status,
            started_at=started_at,
            finished_at=utc_now(),
            runtime_worker_s=runtime_worker_s,
            version=version,
            git_commit=git_commit,
            tags=tags,
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        )
        return {"status": status, "metrics": metrics, "warnings": warnings, "errors": errors}
    except Exception as exc:
        runtime_worker_s = time.time() - started
        errors.append(str(exc))
        metrics["runtime_worker_s"] = runtime_worker_s
        metrics["worker_success"] = 0
        metrics["worker_partial"] = 0
        metrics["worker_failed"] = 1
        log_line(log_path, f"Session failed: {exc}")
        write_metrics_json(
            output_dir,
            session=session_dir.name,
            status="failed",
            started_at=started_at,
            finished_at=utc_now(),
            runtime_worker_s=runtime_worker_s,
            version=detect_version(session_dir, output_dir),
            git_commit=detect_git_commit(),
            tags=tags,
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        )
        return {"status": "failed", "metrics": metrics, "warnings": warnings, "errors": errors}
    finally:
        try:
            if processing.exists():
                processing.unlink()
            if status == "failed":
                touch_state(failed, "\n".join(errors) or utc_now())
            else:
                touch_state(done)
        except Exception as exc:
            log_line(log_path, f"State file update failed: {exc}")

def main() -> None:
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            for session_dir in scan_sessions():
                try:
                    process_session(session_dir)
                except Exception as exc:
                    output_dir = OUT_DIR / session_dir.name
                    log_line(output_dir / "worker.log", f"Unhandled session exception: {exc}")
        except Exception as exc:
            log_line(OUT_DIR / "worker.log", f"Worker loop exception: {exc}")
        time.sleep(POLL_SECONDS)


__all__ = ['detect_git_commit', 'detect_version_from_text', 'detect_version', 'write_influx', 'write_metrics_json', 'touch_state', 'process_session', 'main']


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
