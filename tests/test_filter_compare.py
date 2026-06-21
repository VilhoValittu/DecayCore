from __future__ import annotations

import numpy as np
import scipy.io.wavfile

from decaycore.filter_compare import (
    _resolve_filter_path,
    analyze_filter,
    compare_filters,
    write_html_report,
)


def test_compare_filters_wav_report_contains_metrics_and_names(tmp_path):
    fs_hz = 48_000
    n = 4096
    impulse_a = np.zeros(n, dtype=np.float32)
    impulse_b = np.zeros(n, dtype=np.float32)
    impulse_a[800] = 1.0
    impulse_a[820] = 0.22
    impulse_b[812] = 0.92
    impulse_b[840] = 0.30

    path_a = tmp_path / "filter_a.wav"
    path_b = tmp_path / "filter_b.wav"
    scipy.io.wavfile.write(path_a, fs_hz, impulse_a)
    scipy.io.wavfile.write(path_b, fs_hz, impulse_b)

    analysis_a = analyze_filter(path_a)
    analysis_b = analyze_filter(path_b)
    metrics = compare_filters(analysis_a, analysis_b)
    report_path = write_html_report(analysis_a, analysis_b, metrics, output_path=tmp_path / "report.html")

    html = report_path.read_text(encoding="utf-8")
    assert "DecayCore Filter Comparison" in html
    assert "filter_a" in html
    assert "filter_b" in html
    assert metrics.common_points >= 16
    assert metrics.max_abs_mag_diff_db >= metrics.mean_abs_mag_diff_db >= 0.0
    assert metrics.ir_peak_offset_ms is not None


def test_compare_filters_txt_supported(tmp_path):
    path_a = tmp_path / "a.txt"
    path_b = tmp_path / "b.txt"
    path_a.write_text("20 -3 0\n100 -1 10\n1000 0 30\n10000 -2 50\n", encoding="utf-8")
    path_b.write_text("20 -2 2\n100 -0.5 8\n1000 1 20\n10000 -1 40\n", encoding="utf-8")

    analysis_a = analyze_filter(path_a)
    analysis_b = analyze_filter(path_b)
    metrics = compare_filters(analysis_a, analysis_b)

    assert analysis_a.kind == "txt"
    assert analysis_b.kind == "txt"
    assert metrics.common_freq_min_hz >= 10.0
    assert metrics.mean_abs_phase_diff_deg >= 0.0


def test_resolve_filter_path_normalizes_backslashes_on_posix(tmp_path):
    wav_path = tmp_path / "demo.wav"
    scipy.io.wavfile.write(wav_path, 48_000, np.zeros(64, dtype=np.float32))

    raw = str(wav_path).replace("/", "\\")
    resolved = _resolve_filter_path(raw)

    assert resolved == wav_path.resolve()
