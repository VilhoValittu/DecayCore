import numpy as np

from decaycore.ui import decaycore_plot as plots
from decaycore.ui.export_summary.events import _append_acoustic_events


def test_format_band_rt60_summary_deduplicates_duplicate_display_band():
    bands = {
        63.0: 0.40,
        126.0: 0.20,
        252.0: 0.10,
        400.0: 0.07,
        2016.0: 0.13,
    }

    out = plots.format_band_rt60_summary(bands)

    assert out.count("400Hz:0.07s") == 1


def test_format_band_rt60_summary_preserves_first_occurrence_order():
    bands = {
        63.0: 0.40,
        126.0: 0.20,
        252.0: 0.10,
        400.0: 0.07,
        2016.0: 0.13,
    }

    out = plots.format_band_rt60_summary(bands)

    assert out == "63Hz:0.40s | 126Hz:0.20s | 252Hz:0.10s | 400Hz:0.07s | 2016Hz:0.13s"


def test_format_summary_content_uses_same_rt60_dedupe_for_left_and_right():
    stats = {
        "rt60_val": 0.30,
        "rt60_band_avg": 0.22,
        "rt60_bands": {
            63.0: 0.40,
            126.0: 0.20,
            252.0: 0.10,
            400.0: 0.07,
            2016.0: 0.13,
        },
        "cmp_avg_confidence": 80.0,
        "avg_confidence": 80.0,
        "reflections": [],
    }

    summary = plots.format_summary_content({}, stats, stats)

    assert "Band RT60 L: 63Hz:0.40s | 126Hz:0.20s | 252Hz:0.10s | 400Hz:0.07s | 2016Hz:0.13s" in summary
    assert "Band RT60 R: 63Hz:0.40s | 126Hz:0.20s | 252Hz:0.10s | 400Hz:0.07s | 2016Hz:0.13s" in summary


def test_dsp_quality_report_uses_aligned_measurement_axis_for_mag_error():
    freqs = np.geomspace(20.0, 20000.0, 32)
    measured_aligned = np.full(freqs.shape, -1.0)
    measured_raw = measured_aligned + 92.98
    target = np.zeros(freqs.shape)
    gain = np.ones(freqs.shape)
    stats = {
        "freq_axis": freqs.tolist(),
        "target_mags": target.tolist(),
        "measured_mags": measured_aligned.tolist(),
        "measured_mags_raw": measured_raw.tolist(),
        "predicted_filter_mags": gain.tolist(),
        "realized_filter_mags": gain.tolist(),
        "mag_mask": np.ones(freqs.shape).tolist(),
        "offset_db": 92.98,
        "phase_limit": 125.0,
    }

    report = "\n".join(plots.format_dsp_quality_report_block({}, stats, stats))

    assert "Predicted mag error RMS within mag_c band:      L 0.00 dB | R 0.00 dB" in report
    assert "Realized mag error RMS within mag_c band:       L 0.00 dB | R 0.00 dB" in report
    assert "186." not in report


def test_dsp_quality_report_includes_gd_abs_spread_metric():
    stats = {
        "gd_abs_max_20_500_ms": 17.25,
        "pre_energy_metric_valid": False,
        "pre_energy_metric_suspect": True,
        "pre_energy_metric_reason_code": "anchor_near_boundary",
        "reflections": [],
    }

    report = "\n".join(plots.format_dsp_quality_report_block({}, stats, stats))

    assert "GD abs spread 20-500 Hz (ms): L 17.25 | R 17.25" in report


def test_format_summary_content_warns_on_large_leveling_offset():
    stats = {
        "rt60_val": 0.30,
        "cmp_avg_confidence": 80.0,
        "avg_confidence": 80.0,
        "reflections": [],
        "offset_db": 45.0,
    }

    summary = plots.format_summary_content({}, stats, stats)

    assert "Warning: Very large leveling offset detected" in summary


def test_format_summary_content_reports_tdc_peak_details():
    stats = {
        "rt60_val": 0.30,
        "cmp_avg_confidence": 80.0,
        "avg_confidence": 80.0,
        "reflections": [],
        "tdc_peak_reduction_db": 2.5,
        "tdc_peak_reduction_hz": 52.0,
        "tdc_events_used": 1,
    }

    summary = plots.format_summary_content(
        {"enable_tdc": True, "tdc_strength": 50.0, "tdc_max_reduction_db": 9.0},
        stats,
        stats,
    )

    assert "TDC applied peak: L 2.50 dB @ 52.0 Hz | R 2.50 dB @ 52.0 Hz" in summary
    assert "TDC events used: L 1 | R 1" in summary


def test_format_summary_content_reports_magnitude_authority_trace():
    stats = {
        "rt60_val": 0.30,
        "cmp_avg_confidence": 80.0,
        "avg_confidence": 80.0,
        "reflections": [],
        "max_boost_db": 3.23,
        "boost_candidate_peak_db": 18.5,
        "boost_peak_db": 3.1,
        "clamp_summary": "soft_clip: boost=1 cut=0; hard_clamp: boost=1 cut=0",
        "stage_probes": {
            "post_hardclamp": {
                "stage": "post_hardclamp",
                "boost_peak_db": 1.0,
                "cut_peak_db": -2.0,
                "boost_bins": 3,
                "cut_bins": 4,
                "net_boost_peak_db": 1.0,
            }
        },
        "mag_authority_trace": [
            {
                "stage": "after_softclip",
                "reason_codes": ["softclip_boost", "user_boost_cap"],
                "changed_bins": 9,
                "max_delta_db": 12.0,
                "boost_peak_after_db": 6.0,
                "cut_peak_after_db": -2.0,
            },
            {
                "stage": "after_excursion_protection",
                "reason_codes": ["excursion_full_block", "excursion_soft_cap"],
                "changed_bins": 3,
                "max_delta_db": 6.0,
                "boost_peak_after_db": 3.0,
                "cut_peak_after_db": -2.0,
            },
            {
                "stage": "after_hardclamp",
                "reason_codes": ["hardclamp_boost", "user_boost_cap"],
                "changed_bins": 7,
                "max_delta_db": 3.0,
                "boost_peak_after_db": 1.0,
                "cut_peak_after_db": -2.0,
            }
        ],
    }

    summary = _append_acoustic_events("", stats, stats)

    assert "=== CLAMP DIAGNOSTICS (LEFT) ===" in summary
    assert "=== MAGNITUDE AUTHORITY TRACE (LEFT) ===" in summary
    assert "after_hardclamp" in summary
    assert "Large boost candidate warning: 18.50 dB requested before safety limits" in summary
    assert "Authority verdict: large boost candidate was safely reduced by softclip + hardclamp" in summary
    assert "excursion_full_block,excursion_soft_cap" in summary
    assert "hardclamp_boost,user_boost_cap" in summary
    assert "=== STAGE CHECKPOINTS (LEFT) ===" in summary
