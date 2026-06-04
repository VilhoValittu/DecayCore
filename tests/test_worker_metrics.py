import json
import time

import pytest
from decaycore.worker import worker


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def patch_worker_runtime(monkeypatch, tmp_path):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(worker, "OUT_DIR", out_dir)
    monkeypatch.setattr(worker, "write_influx", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "detect_version", lambda *args, **kwargs: "v.test")
    monkeypatch.setattr(worker, "detect_git_commit", lambda: "abc123")
    return out_dir


def test_metrics_only_session_is_partial_and_writes_done(monkeypatch, tmp_path):
    patch_worker_runtime(monkeypatch, tmp_path)
    session = tmp_path / "in" / "test_session"
    session.mkdir(parents=True)
    write_json(session / "metrics.json", {"auto": {"score": 82.1}, "taps": 4096})

    result = worker.process_session(session)

    assert result["status"] == "partial"
    assert (session / ".done").exists()
    assert not (session / ".failed").exists()
    output = json.loads((tmp_path / "out" / "test_session" / "metrics.json").read_text(encoding="utf-8"))
    assert output["status"] == "partial"
    assert output["metrics"]["auto_score"] == 82.1
    assert output["metrics"]["filter_taps"] == 4096
    assert output["metrics"]["worker_partial"] == 1


def test_rt60_json_extracts_bands_and_averages(monkeypatch, tmp_path):
    patch_worker_runtime(monkeypatch, tmp_path)
    session = tmp_path / "in" / "rt60_session"
    session.mkdir(parents=True)
    write_json(session / "rt60.json", {"rt60": {"31.5": 0.8, "63": 0.7, "125": 0.6, "250": 0.5, "500": 0.4, "1k": 0.3, "2k": 0.35, "4k": 0.25}})

    result = worker.process_session(session)

    metrics = result["metrics"]
    assert metrics["rt60_31"] == 0.8
    assert metrics["rt60_63"] == 0.7
    assert metrics["rt60_1000"] == 0.3
    assert metrics["rt60_bass_avg"] == 0.65
    assert metrics["rt60_mid_avg"] == pytest.approx(0.35)
    assert metrics["rt60_bass_to_mid_ratio"] > 1.8


def test_harmonics_json_extracts_risk_fields(monkeypatch, tmp_path):
    patch_worker_runtime(monkeypatch, tmp_path)
    session = tmp_path / "in" / "harmonics_session"
    session.mkdir(parents=True)
    write_json(
        session / "harmonics.json",
        {
            "harmonics": {
                "h2": {"50": -40, "100": -50, "250": -55},
                "h3": {"50": -45},
            }
        },
    )

    result = worker.process_session(session)

    metrics = result["metrics"]
    assert metrics["harmonic_risk_20_80"] == 20.0
    assert metrics["harmonic_risk_80_200"] == 10.0
    assert metrics["harmonic_risk_200_500"] == 5.0
    assert metrics["harmonic_risk_max"] == 20.0
    assert metrics["harmonic_risk_freq_hz"] == 50.0
    assert metrics["boost_risk_20_80"] == 20.0


def test_summary_text_parser_extracts_metrics_and_tags():
    parsed = worker.parse_summary_text(
        """
        Runtime: 12.5 s
        DecayCore runtime: 10.2 s
        Score: 91
        Filter score: 88.5
        Mag error RMS: 1.2 dB
        Mag error Max: 4.8 dB
        FR error RMS: 1.3 dB
        Response error Max: 5.1 dB
        Residual peak count: 4
        GD spike count: 2
        Max boost: 3.5 dB
        Bass integration crossover: 90 Hz
        BI mode: direct_dac
        BI polarity: inverted
        Selected house curve: harman
        """
    )

    assert parsed["runtime_s"] == 12.5
    assert parsed["camillafir_runtime_s"] == 10.2
    assert parsed["score"] == 91.0
    assert parsed["filter_score"] == 88.5
    assert parsed["mag_error_rms_db"] == 1.3
    assert parsed["mag_error_max_db"] == 5.1
    assert parsed["residual_peak_count"] == 4.0
    assert parsed["gd_spike_count"] == 2.0
    assert parsed["max_boost_db"] == 3.5
    assert parsed["bi_crossover_hz"] == 90.0
    assert parsed["bi_mode"] == "direct_dac"
    assert parsed["bi_polarity"] == "inverted"
    assert parsed["selected_house_curve"] == "harman"


def test_summary_text_parser_extracts_combined_mag_error_rms_max():
    parsed = worker.parse_summary_text("Mag error RMS/max: 1.2 / 4.8 dB\n")

    assert parsed["mag_error_rms_db"] == 1.2
    assert parsed["mag_error_max_db"] == 4.8


def test_nested_headless_metrics_normalize_mag_error_aliases(tmp_path):
    path = tmp_path / "metrics.json"
    write_json(
        path,
        {
            "metrics": {
                "magnitude_error_rms_db": 1.4,
                "response_error_max_db": 4.9,
                "auto_score": 81.0,
            }
        },
    )

    parsed = worker.parse_json_metrics(path)

    assert parsed["mag_error_rms_db"] == 1.4
    assert parsed["mag_error_max_db"] == 4.9
    assert parsed["auto_score"] == 81.0


def test_metric_aliases_cover_common_magnitude_error_names():
    rms_aliases = [
        "magnitude_error_rms_db",
        "magnitude_rms_db",
        "error_rms_db",
        "response_error_rms_db",
        "fr_error_rms_db",
        "mag_rms",
        "rms_db",
    ]
    max_aliases = [
        "magnitude_error_max_db",
        "magnitude_max_db",
        "error_max_db",
        "response_error_max_db",
        "fr_error_max_db",
        "mag_max",
        "max_error_db",
    ]

    for alias in rms_aliases:
        assert worker.normalize_metric_key(alias) == "mag_error_rms_db"
        assert worker.normalize_metric_key(f"metrics_{alias}") == "mag_error_rms_db"
    for alias in max_aliases:
        assert worker.normalize_metric_key(alias) == "mag_error_max_db"
        assert worker.normalize_metric_key(f"metrics_{alias}") == "mag_error_max_db"


def test_failed_invocation_marks_failed_without_raising(monkeypatch, tmp_path):
    patch_worker_runtime(monkeypatch, tmp_path)
    session = tmp_path / "in" / "failed_session"
    session.mkdir(parents=True)
    write_json(session / "config.json", {"run": True})

    monkeypatch.setattr(
        worker,
        "run_camillafir_session",
        lambda session_dir, output_dir: {
            "attempted": True,
            "success": False,
            "warnings": [],
            "errors": ["boom"],
        },
    )

    result = worker.process_session(session)

    assert result["status"] == "failed"
    assert (session / ".failed").exists()
    assert not (session / ".processing").exists()
    output = json.loads((tmp_path / "out" / "failed_session" / "metrics.json").read_text(encoding="utf-8"))
    assert output["status"] == "failed"
    assert output["metrics"]["worker_failed"] == 1


def test_session_ready_respects_state_and_stability(monkeypatch, tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "metrics.json").write_text("{}", encoding="utf-8")
    old = time.time() - 30
    monkeypatch.setattr(worker, "STABLE_SECONDS", 0)
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    for path in (session, session / "metrics.json"):
        path.touch()
        path_stat_time = old
        # os.utime avoids waiting while still exercising recursive mtime checks.
        import os

        os.utime(path, (path_stat_time, path_stat_time))

    assert worker.is_session_ready(session)
    (session / ".done").write_text("", encoding="utf-8")
    assert not worker.is_session_ready(session)
