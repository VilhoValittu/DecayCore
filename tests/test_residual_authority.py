from types import SimpleNamespace

import numpy as np
import pytest

from decaycore.auto_mode.cache_signature import _auto_signature_payload
from decaycore.config.decaycore_pipeline import build_filter_config
from decaycore.config.models import FilterConfig
from decaycore.dsp.phase_ir_residual import apply_residual_pass_if_enabled
from decaycore.dsp.residual_authority import (
    _apply_residual_authority_caps,
    build_residual_authority_caps,
)


class _NullLogger:
    def info(self, *_args, **_kwargs) -> None:
        return None

    def warning(self, *_args, **_kwargs) -> None:
        return None


def _cfg_float_allow_zero(cfg, name: str, default: float) -> float:
    try:
        value = float(getattr(cfg, name, default))
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        return float(default)
    return float(value) if np.isfinite(value) else float(default)


def _near(freq, hz, width=0.04):
    return np.abs(np.log2(freq / float(hz))) <= float(width)


def _base_data(**overrides):
    data = {
        "filter_type": "Asymmetric",
        "mixed_freq": 180.0,
        "mag_c_min": 10.0,
        "mag_c_max": 200.0,
        "max_boost": 5.0,
        "max_cut_db": 15.0,
        "phase_limit": 1000.0,
        "normalize_opt": False,
        "exc_prot": False,
        "exc_freq": 20.0,
        "ir_window_right": 500.0,
        "ir_window_left": 85.0,
        "fdw_cycles": 10.0,
        "lvl_manual_db": 0.0,
        "manual_target_tilt_db_per_oct": 0.0,
        "lvl_min": 300.0,
        "lvl_max": 3000.0,
    }
    data.update(overrides)
    return data


def test_off_mode_disables_residual_correction():
    freq = np.geomspace(10.0, 20000.0, 2048)

    caps = build_residual_authority_caps(freq, residual_pass_mode="off")

    assert float(caps["residual_boost_cap_db"].max()) == 0.0
    assert float(caps["residual_cut_cap_db"].max()) == 0.0
    assert not bool(caps["residual_allowed"].any())


def test_deep_null_blocks_boost():
    freq = np.geomspace(10.0, 20000.0, 2048)
    null_risk = np.zeros_like(freq)
    boost = np.ones_like(freq)
    cut = np.ones_like(freq)
    modal = np.zeros_like(freq)
    band = _near(freq, 80.0)
    null_risk[band] = 1.0
    boost[band] = 0.0

    caps = build_residual_authority_caps(
        freq,
        authority_null_risk=null_risk,
        authority_boost=boost,
        authority_cut=cut,
        authority_modal_support=modal,
        residual_pass_mode="modal_polish",
    )

    assert np.all(caps["residual_boost_cap_db"][band] <= 0.5)
    assert not bool(caps["residual_boost_allowed"][band].any())


def test_modal_peak_allows_cut_polish():
    freq = np.geomspace(10.0, 20000.0, 2048)
    modal = np.zeros_like(freq)
    cut = np.ones_like(freq)
    null_risk = np.zeros_like(freq)
    reflection = np.zeros_like(freq)
    band = _near(freq, 45.0)
    modal[band] = 1.0

    caps = build_residual_authority_caps(
        freq,
        authority_cut=cut,
        authority_modal_support=modal,
        authority_null_risk=null_risk,
        authority_reflection_risk=reflection,
        residual_pass_mode="modal_polish",
    )

    assert bool(caps["residual_cut_allowed"][band].any())
    assert float(caps["residual_cut_cap_db"][band].max()) > 2.0


def test_low_modal_support_blocks_modal_polish_cut():
    freq = np.geomspace(10.0, 20000.0, 2048)

    caps = build_residual_authority_caps(
        freq,
        authority_cut=np.ones_like(freq),
        authority_modal_support=np.zeros_like(freq),
        residual_pass_mode="modal_polish",
    )

    assert np.count_nonzero(caps["residual_cut_allowed"]) < int(0.05 * freq.size)


def test_reflection_risk_suppresses_residual_correction():
    freq = np.geomspace(10.0, 20000.0, 2048)
    reflection = np.zeros_like(freq)
    boost = np.ones_like(freq)
    cut = np.ones_like(freq)
    modal = np.ones_like(freq)
    band = _near(freq, 300.0)
    reflection[band] = 1.0

    caps = build_residual_authority_caps(
        freq,
        authority_reflection_risk=reflection,
        authority_boost=boost,
        authority_cut=cut,
        authority_modal_support=modal,
        residual_pass_mode="modal_polish",
    )

    assert not bool(caps["residual_boost_allowed"][band].any())
    assert not bool(caps["residual_cut_allowed"][band].any())


def test_general_fit_preserves_caps_but_respects_null_guard():
    freq = np.geomspace(10.0, 20000.0, 2048)
    null_risk = np.zeros_like(freq)

    clean = build_residual_authority_caps(
        freq,
        authority_boost=np.ones_like(freq),
        authority_cut=np.ones_like(freq),
        authority_null_risk=null_risk,
        residual_pass_mode="general_fit",
    )
    assert float(clean["residual_boost_cap_db"].max()) == pytest.approx(2.0)
    assert float(clean["residual_cut_cap_db"].max()) == pytest.approx(4.0)

    band = _near(freq, 100.0)
    null_risk[band] = 1.0
    guarded = build_residual_authority_caps(
        freq,
        authority_boost=np.ones_like(freq),
        authority_cut=np.ones_like(freq),
        authority_null_risk=null_risk,
        residual_pass_mode="general_fit",
    )
    assert float(guarded["residual_boost_cap_db"][band].max()) <= 0.5


def test_final_residual_correction_is_capped():
    freq = np.geomspace(10.0, 20000.0, 2048)
    residual = np.zeros_like(freq)
    band = _near(freq, 80.0)
    residual[band] = 5.0
    caps = {
        "residual_boost_cap_db": np.full_like(freq, 0.5),
        "residual_cut_cap_db": np.full_like(freq, 4.0),
        "residual_allowed": np.ones_like(freq, dtype=bool),
    }

    out = _apply_residual_authority_caps(residual, caps)

    assert float(out[band].max()) <= 0.5


def test_residual_pass_caps_authority_blocked_boost():
    freq = np.geomspace(10.0, 220.0, 512)
    gain = np.zeros_like(freq)
    target = np.full_like(freq, 6.0)
    band = _near(freq, 80.0, width=0.12)
    st = {
        "authority_null_risk": np.where(band, 1.0, 0.0),
        "authority_boost": np.where(band, 0.0, 1.0),
        "authority_cut": np.ones_like(freq),
        "authority_modal_support": np.zeros_like(freq),
        "authority_reflection_risk": np.zeros_like(freq),
    }
    cfg = SimpleNamespace(
        enable_residual_pass=True,
        enable_mag_correction=True,
        residual_pass_mode="modal_polish",
        residual_strength=1.0,
        residual_smoothing_mult=1.0,
        residual_conf_power=1.0,
        residual_max_delta_db=6.0,
        residual_authority_smooth_oct=1.0 / 192.0,
        mag_c_min=20.0,
        mag_c_max=180.0,
        trans_width=20.0,
        low_bass_cut_enable=False,
        low_bass_cut_hz=0.0,
        exc_prot=False,
        exc_freq=0.0,
        max_boost_db=12.0,
        max_cut_db=15.0,
    )

    out, telemetry = apply_residual_pass_if_enabled(
        cfg=cfg,
        freq_axis=freq,
        gain_db=gain,
        conf_mask=None,
        m_anal=np.zeros_like(freq),
        calc_offset_db=0.0,
        target_mags=target,
        st=st,
        mask_c=np.ones_like(freq, dtype=bool),
        base_sigma=6.0,
        filter_smooth=12.0,
        df_mode=False,
        raw_g=np.zeros_like(freq),
        final_g=np.zeros_like(freq),
        logger=_NullLogger(),
        cfg_float_allow_zero_fn=_cfg_float_allow_zero,
    )

    assert float(np.max(out[band] - gain[band])) <= 0.5
    assert telemetry is not None
    assert float(telemetry.residual_requested_boost_max_db) > float(telemetry.residual_applied_boost_max_db)


def test_residual_guard_config_is_built_and_affects_cache_signature():
    cfg = build_filter_config(
        FilterConfig_cls=FilterConfig,
        fs_v=44100,
        taps_v=1024,
        data=_base_data(
            residual_pass_mode="general_fit",
            residual_null_guard_strength=0.25,
            residual_max_boost_general_db=1.5,
        ),
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
    )
    assert cfg.residual_pass_mode == "general_fit"
    assert cfg.residual_null_guard_strength == pytest.approx(0.25)
    assert cfg.residual_max_boost_general_db == pytest.approx(1.5)

    base_payload = _auto_signature_payload(
        base_data=_base_data(residual_null_guard_strength=1.0),
        measurements={},
        fs_v=44100,
        taps_v=1024,
        xos=[],
        hpf=None,
    )
    changed_payload = _auto_signature_payload(
        base_data=_base_data(residual_null_guard_strength=0.5),
        measurements={},
        fs_v=44100,
        taps_v=1024,
        xos=[],
        hpf=None,
    )
    assert base_payload["mag_post_limits"]["residual_null_guard_strength"] == pytest.approx(1.0)
    assert changed_payload["mag_post_limits"]["residual_null_guard_strength"] == pytest.approx(0.5)
    assert base_payload != changed_payload
