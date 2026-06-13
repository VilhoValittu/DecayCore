from decaycore.config.decaycore_config import load_config
from decaycore.engine_build import build_config


def test_build_config_ir_anchor_defaults_survive_none_inputs():
    data = load_config()
    data.update(
        {
            "mode": "AUTO",
            "camillafir_automatic_mode": True,
            "filter_type": "Asymmetric",
            "min_causal_ms": None,
            "auto_asym_left_ratio": None,
            "auto_asym_left_max_ms": None,
            "pre_energy_ratio_max": None,
            "pre_energy_guard_strength": None,
        }
    )

    cfg = build_config(
        data,
        fs_v=44100,
        taps_v=4096,
        xos=[],
        hpf={"enabled": True, "freq": 25.0, "order": 2},
    )

    assert float(cfg.min_causal_ms) == 80.0
    assert float(cfg.auto_asym_left_ratio) == 0.35
    assert float(cfg.auto_asym_left_max_ms) == 25.0
    assert float(cfg.pre_energy_ratio_max) == 0.25
    assert float(cfg.pre_energy_guard_strength) == 0.8


def test_auto_flat_enables_unsafe_raw_dsp_for_prefer_bass():
    data = load_config()
    data.update(
        {
            "mode": "AUTO",
            "camillafir_automatic_mode": True,
            "auto_goal": "flat",
            "filter_type": "Asymmetric",
            "unsafe_raw_dsp": True,
            "max_boost": 6.0,
            "max_cut_db": 15.0,
            "max_slope_db_per_oct": 24.0,
            "reg_strength": 30.0,
            "low_bass_cut_enable": True,
            "exc_prot": True,
            "enable_residual_pass": True,
            "enable_afdw": False,
        }
    )

    cfg = build_config(
        data,
        fs_v=44100,
        taps_v=4096,
        xos=[],
        hpf={"enabled": True, "freq": 25.0, "order": 2},
    )

    assert bool(cfg.unsafe_raw_dsp) is True
    assert float(cfg.max_boost_db) == 6.0
    assert float(cfg.max_boost_db_user) == 6.0
    assert float(cfg.max_cut_db) >= 120.0
    assert float(cfg.max_slope_db_per_oct) == 0.0
    assert float(cfg.reg_strength) == 0.0
    assert bool(cfg.low_bass_cut_enable) is False
    assert bool(cfg.exc_prot) is False
    assert bool(cfg.acoustic_authority_limits_enable) is False
    assert bool(cfg.enable_residual_pass) is False
    assert bool(cfg.enable_afdw) is False


def test_auto_balanced_still_blocks_unsafe_raw_dsp():
    data = load_config()
    data.update(
        {
            "mode": "AUTO",
            "camillafir_automatic_mode": True,
            "auto_goal": "balanced",
            "filter_type": "Asymmetric",
            "unsafe_raw_dsp": True,
            "max_boost": 18.0,
        }
    )

    cfg = build_config(
        data,
        fs_v=44100,
        taps_v=4096,
        xos=[],
        hpf={"enabled": True, "freq": 25.0, "order": 2},
    )

    assert bool(cfg.unsafe_raw_dsp) is False
    assert float(cfg.max_boost_db) <= 12.0
