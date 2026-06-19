from __future__ import annotations

from dataclasses import dataclass

from .results_formatters import safe_float


@dataclass(frozen=True)
class BassIntegrationDspSetting:
    label_key: str
    summary_label: str
    value: str


def _fmt_value(value: float, unit: str = "", *, digits: int = 3) -> str:
    if value != value or abs(value) == float("inf"):
        return "n/a"
    return f"{float(value):.{digits}f}{unit}"


def _safe_int(value: object, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _order_to_slope_db_per_oct(order: object, default: int) -> int:
    parsed_order = _safe_int(order, default)
    return max(1, parsed_order) * 6


def _fmt_filter_value(freq_hz: float, slope_db_per_oct: int) -> str:
    return f"{freq_hz:.1f} Hz / {int(slope_db_per_oct)} dB/oct"


def build_bass_integration_dsp_settings(data: dict | None) -> list[BassIntegrationDspSetting]:
    ui_data = dict(data or {})
    if not bool(ui_data.get("bass_integration_enable", False)):
        return []

    bi_meta = dict(ui_data.get("_bass_integration_meta", {}) or {})
    alignment = dict(bi_meta.get("alignment", {}) or {})
    allpass_meta = dict(bi_meta.get("recommended_allpass", {}) or {})
    diag = dict(bi_meta.get("diagnostics", {}) or {})
    combine_mode = str(
        diag.get(
            "sub_combine_mode",
            bi_meta.get("sub_combine_mode", ui_data.get("bass_integration_sub_combine_mode", "average")),
        )
        or "average"
    ).strip().lower()
    is_dual_sub_shared = bool(diag.get("dual_sub_preprocessing_applied", False)) or combine_mode == "dual_sub_peak_aligned_average"

    main_hpf_hz = float(bi_meta.get("avr_crossover_hz", ui_data.get("avr_crossover_hz", 80.0)) or 80.0)
    sub_lpf_hz = float(
        bi_meta.get(
            "direct_dac_sub_lpf_hz",
            ui_data.get("direct_dac_sub_lpf_hz", main_hpf_hz),
        )
        or main_hpf_hz
    )
    xo_slope_default = _safe_int(ui_data.get("sub_crossover_slope", 12), 12)
    main_hpf_slope_db_per_oct = _order_to_slope_db_per_oct(
        diag.get("direct_dac_main_hpf_order", xo_slope_default / 6.0),
        max(1, xo_slope_default // 6),
    )
    sub_lpf_slope_db_per_oct = _order_to_slope_db_per_oct(
        diag.get("direct_dac_sub_lpf_order", xo_slope_default / 6.0),
        max(1, xo_slope_default // 6),
    )

    sub_delay_value = safe_float(
        alignment.get("delay_ms", ui_data.get("bass_integration_sub_delay_ms", float("nan"))),
        float("nan"),
    )
    sub_array_delay_value = safe_float(
        alignment.get("sub_array_delay_ms", ui_data.get("bass_integration_sub_array_delay_ms", sub_delay_value)),
        sub_delay_value,
    )
    main_delay_default = max(0.0, -sub_delay_value) if sub_delay_value == sub_delay_value else 0.0
    main_l_delay_value = safe_float(
        alignment.get("main_l_delay_ms", ui_data.get("bass_integration_main_l_delay_ms", main_delay_default)),
        main_delay_default,
    )
    main_r_delay_value = safe_float(
        alignment.get("main_r_delay_ms", ui_data.get("bass_integration_main_r_delay_ms", main_delay_default)),
        main_delay_default,
    )
    polarity_invert = bool(alignment.get("polarity_invert", ui_data.get("bass_integration_sub_polarity_invert", False)))
    gain_trim_value = safe_float(
        alignment.get("gain_trim_db", ui_data.get("bass_integration_sub_gain_trim_db", float("nan"))),
        float("nan"),
    )
    allpass_on = bool(allpass_meta.get("enabled", False))
    delay_label_key = "results_metric_bass_shared_sub_array_delay" if is_dual_sub_shared else "results_metric_bass_alignment_delay"
    delay_label = "Shared sub-array delay" if is_dual_sub_shared else "Sub delay"
    delay_value = sub_array_delay_value if is_dual_sub_shared else sub_delay_value
    polarity_label_key = (
        "results_metric_bass_shared_sub_polarity" if is_dual_sub_shared else "results_metric_bass_alignment_polarity"
    )
    polarity_label = "Shared sub polarity" if is_dual_sub_shared else "Sub polarity"
    gain_label_key = "results_metric_bass_shared_sub_gain" if is_dual_sub_shared else "results_metric_bass_alignment_gain"
    gain_label = "Shared sub gain trim" if is_dual_sub_shared else "Sub gain trim"
    allpass_label_key = "results_metric_bass_shared_sub_allpass" if is_dual_sub_shared else "results_metric_bass_allpass"
    allpass_label = "Shared bass allpass" if is_dual_sub_shared else "Bass allpass"
    allpass_freq_label_key = (
        "results_metric_bass_shared_sub_allpass_freq" if is_dual_sub_shared else "results_metric_bass_allpass_freq"
    )
    allpass_freq_label = "Shared bass allpass freq" if is_dual_sub_shared else "Bass allpass freq"
    allpass_q_label_key = (
        "results_metric_bass_shared_sub_allpass_q" if is_dual_sub_shared else "results_metric_bass_allpass_q"
    )
    allpass_q_label = "Shared bass allpass Q" if is_dual_sub_shared else "Bass allpass Q"

    settings = [
        BassIntegrationDspSetting(
            "results_metric_main_hpf",
            "Main HPF",
            _fmt_filter_value(main_hpf_hz, main_hpf_slope_db_per_oct),
        ),
        BassIntegrationDspSetting(
            "results_metric_sub_lpf",
            "Sub LPF",
            _fmt_filter_value(sub_lpf_hz, sub_lpf_slope_db_per_oct),
        ),
        BassIntegrationDspSetting(delay_label_key, delay_label, _fmt_value(delay_value, " ms")),
        BassIntegrationDspSetting(
            "results_metric_bass_dsp_main_delay",
            "Main delay L/R",
            f"L {_fmt_value(main_l_delay_value, ' ms')}, R {_fmt_value(main_r_delay_value, ' ms')}",
        ),
        BassIntegrationDspSetting(
            polarity_label_key,
            polarity_label,
            "INVERTED" if polarity_invert else "NORMAL",
        ),
        BassIntegrationDspSetting(
            gain_label_key,
            gain_label,
            _fmt_value(gain_trim_value, " dB"),
        ),
        BassIntegrationDspSetting(
            allpass_label_key,
            allpass_label,
            "ON" if allpass_on else "OFF",
        ),
    ]

    if allpass_on:
        settings.extend(
            [
                BassIntegrationDspSetting(
                    allpass_freq_label_key,
                    allpass_freq_label,
                    f"{float(allpass_meta.get('freq_hz', 0.0) or 0.0):.1f} Hz",
                ),
                BassIntegrationDspSetting(
                    allpass_q_label_key,
                    allpass_q_label,
                    f"{float(allpass_meta.get('q', 0.707) or 0.707):.3f}",
                ),
            ]
        )

    return settings


__all__ = ["BassIntegrationDspSetting", "build_bass_integration_dsp_settings"]
