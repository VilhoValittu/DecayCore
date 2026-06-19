import json
from pathlib import Path

from decaycore.resources.i8n.decaycore_i18n import TRANSLATIONS, TRANSLATIONS_META
from decaycore.ui_i18n import (
    AFDW_PRESET_TIGHT,
    LAYOUT_STEREO,
    LVL_ALGO_AVERAGE,
    LVL_MODE_MANUAL,
    OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT,
    TDC_PRESET_SAFE,
    normalize_afdw_preset_key,
    normalize_layout_value,
    normalize_lvl_algo_value,
    normalize_lvl_mode_value,
    normalize_output_tilt_source_value,
    normalize_tdc_preset_key,
    tr_options,
)


def test_tr_options_preserves_stable_keys():
    options = tr_options(lambda key: f"tr:{key}", {"mono": "layout_mono", "stereo": "layout_stereo"})

    assert options == {
        "mono": "tr:layout_mono",
        "stereo": "tr:layout_stereo",
    }


def test_normalize_layout_value_accepts_legacy_and_translated_labels():
    assert normalize_layout_value("stereo") == LAYOUT_STEREO
    assert normalize_layout_value("Stereo") == LAYOUT_STEREO
    assert normalize_layout_value(TRANSLATIONS["fi"]["layout_stereo"]) == LAYOUT_STEREO


def test_normalize_level_mode_value_accepts_legacy_and_translated_labels():
    assert normalize_lvl_mode_value("manual") == LVL_MODE_MANUAL
    assert normalize_lvl_mode_value("Manual") == LVL_MODE_MANUAL
    assert normalize_lvl_mode_value(TRANSLATIONS["fi"]["lvl_mode_manual"]) == LVL_MODE_MANUAL


def test_normalize_level_algo_value_accepts_legacy_and_translated_labels():
    assert normalize_lvl_algo_value("average") == LVL_ALGO_AVERAGE
    assert normalize_lvl_algo_value("Average") == LVL_ALGO_AVERAGE
    assert normalize_lvl_algo_value(TRANSLATIONS["fi"]["lvl_algo_average"]) == LVL_ALGO_AVERAGE


def test_normalize_output_tilt_source_value_accepts_legacy_and_translated_labels():
    assert normalize_output_tilt_source_value("manual_target_tilt") == OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT
    assert normalize_output_tilt_source_value("Use Manual Tilt value") == OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT
    assert (
        normalize_output_tilt_source_value(TRANSLATIONS["fi"]["output_tilt_use_manual_target_tilt"])
        == OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT
    )


def test_normalize_preset_keys_accept_legacy_and_translated_labels():
    assert normalize_tdc_preset_key("Safe") == TDC_PRESET_SAFE
    assert normalize_tdc_preset_key(TRANSLATIONS["fi"]["tdc_preset_safe"]) == TDC_PRESET_SAFE
    assert normalize_afdw_preset_key("Tight") == AFDW_PRESET_TIGHT
    assert normalize_afdw_preset_key(TRANSLATIONS["fi"]["afdw_preset_tight"]) == AFDW_PRESET_TIGHT


def test_translation_meta_is_not_exposed_as_language_catalog():
    translations_path = Path(__file__).resolve().parents[1] / "src" / "decaycore" / "resources" / "i8n" / "translations.json"
    payload = json.loads(translations_path.read_text(encoding="utf-8"))

    assert payload["_meta"]["product"] == "DecayCore"
    assert "_meta" not in TRANSLATIONS
    assert TRANSLATIONS_META["product"] == payload["_meta"]["product"]


def test_hybrid_iir_translation_keys_exist():
    required = {
        "hybrid_iir_title",
        "hybrid_iir_help",
        "hybrid_iir_enabled",
        "hybrid_iir_tuning_title",
        "hybrid_iir_max_filters_per_channel",
        "hybrid_iir_min_freq_hz",
        "hybrid_iir_max_freq_hz",
        "hybrid_iir_min_peak_db",
        "hybrid_iir_min_q",
        "hybrid_iir_max_q",
        "hybrid_iir_max_cut_db",
        "hybrid_iir_min_confidence",
        "hybrid_iir_min_gd_excess_ms",
        "conf_pull_max_hz_label",
        "conf_pull_gamma_cut_label",
        "conf_pull_gamma_boost_label",
        "conf_pull_bass_boost_floor_min_label",
        "conf_pull_bass_boost_restore_label",
        "adv_summary_max_hz",
        "adv_summary_cut_gamma",
        "adv_summary_boost_gamma",
        "adv_summary_bass_boost_floor",
        "adv_summary_bass_restore",
    }
    for lang in ("en", "fi"):
        missing = required.difference(TRANSLATIONS[lang])
        assert not missing


def test_measurement_translation_keys_exist_for_linux_virtual_sub_support():
    required = {
        "measurement_refresh_devices",
        "measurement_sub_output_channel",
        "measurement_sub_output_channel_hint",
    }
    for lang in ("en", "fi"):
        missing = required.difference(TRANSLATIONS[lang])
        assert not missing


def test_bass_integration_dsp_settings_translation_keys_exist():
    required = {
        "results_section_bass_dsp_settings",
        "bass_integration_dsp_settings_note",
        "results_metric_bass_dsp_main_delay",
        "results_metric_bass_xo_main_sub_gd_assessment",
        "results_value_bass_xo_main_sub_gd_assessment_good",
        "results_value_bass_xo_main_sub_gd_assessment_average",
        "results_value_bass_xo_main_sub_gd_assessment_elevated",
        "results_value_bass_xo_main_sub_gd_assessment_reposition",
    }
    for lang in ("en", "fi"):
        missing = required.difference(TRANSLATIONS[lang])
        assert not missing
