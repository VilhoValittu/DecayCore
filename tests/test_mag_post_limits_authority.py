import logging
from types import SimpleNamespace

import numpy as np

from decaycore.dsp.dsp_config import CfgReader
from decaycore.dsp.mag_post_limits import _apply_acoustic_authority_caps


def _run_caps(st, cfg=None, *, boost_cap=None, max_cut_db=15.0):
    freq = np.geomspace(10.0, 1000.0, 128)
    mask = (freq >= 20.0) & (freq <= 300.0)
    if boost_cap is None:
        boost_cap = np.full_like(freq, 6.0, dtype=float)
    if cfg is None:
        cfg = SimpleNamespace()
    return freq, mask, _apply_acoustic_authority_caps(
        cfg=cfg,
        cfg_reader=CfgReader(cfg),
        st=st,
        logger=logging.getLogger(__name__),
        freq_axis=freq,
        mask_c=mask,
        boost_cap_db=np.asarray(boost_cap, dtype=float),
        max_cut_db=float(max_cut_db),
    )


def test_acoustic_authority_caps_disabled_leaves_caps_full():
    st = {
        "authority_boost": np.zeros(128, dtype=float),
        "authority_cut": np.zeros(128, dtype=float),
    }
    cfg = SimpleNamespace(acoustic_authority_limits_enable=False)

    freq, _, result = _run_caps(st, cfg=cfg)

    assert bool(result["enabled"]) is False
    assert result["source"] == "disabled"
    assert np.allclose(result["boost_cap_db"], np.full_like(freq, 6.0))
    assert np.allclose(result["cut_cap_db"], np.full_like(freq, 15.0))
    assert st["acoustic_authority_limits_enabled"] is False
    assert st["authority_boost_cap_reduced_bins"] == 0
    assert st["authority_cut_cap_reduced_bins"] == 0


def test_acoustic_authority_caps_missing_arrays_leave_caps_full():
    st = {}

    freq, _, result = _run_caps(st)

    assert bool(result["enabled"]) is False
    assert result["source"] == "missing"
    assert np.allclose(result["boost_cap_db"], np.full_like(freq, 6.0))
    assert np.allclose(result["cut_cap_db"], np.full_like(freq, 15.0))
    assert st["acoustic_authority_limits_source"] == "missing"
    assert st["authority_boost_cap_reduced_bins"] == 0
    assert st["authority_cut_cap_reduced_bins"] == 0


def test_acoustic_authority_caps_low_boost_authority_reduces_boost_cap():
    freq = np.geomspace(10.0, 1000.0, 128)
    st = {
        "authority_boost": np.ones_like(freq),
        "authority_cut": np.ones_like(freq),
    }
    idx = int(np.argmin(np.abs(freq - 80.0)))
    st["authority_boost"][idx] = 0.1

    _, _, result = _run_caps(st)

    cap_80 = float(np.asarray(result["boost_cap_db"])[idx])
    assert bool(result["enabled"]) is True
    assert 0.3 < cap_80 < 1.0
    assert float(result["boost_reduction_max_db"]) > 5.0
    assert int(result["boost_reduced_bins"]) > 0
    assert st["authority_boost_cap_reduced_bins"] > 0


def test_acoustic_authority_caps_high_cut_authority_preserves_cut_depth():
    freq = np.geomspace(10.0, 1000.0, 128)
    st = {
        "authority_boost": np.ones_like(freq),
        "authority_cut": np.zeros_like(freq),
    }
    idx = int(np.argmin(np.abs(freq - 45.0)))
    st["authority_cut"][idx] = 0.95

    _, _, result = _run_caps(st)

    cut_cap = float(np.asarray(result["cut_cap_db"])[idx])
    assert 14.0 < cut_cap <= 15.0


def test_acoustic_authority_caps_low_cut_authority_respects_minimum_cut_cap():
    freq = np.geomspace(10.0, 1000.0, 128)
    st = {
        "authority_boost": np.ones_like(freq),
        "authority_cut": np.zeros_like(freq),
    }
    cfg = SimpleNamespace(authority_cut_min_frac=0.0, authority_cut_min_cap_db=3.0)
    idx = int(np.argmin(np.abs(freq - 90.0)))

    _, _, result = _run_caps(st, cfg=cfg)

    assert float(np.asarray(result["cut_cap_db"])[idx]) == 3.0


def test_acoustic_authority_caps_sanitizes_nan_and_inf():
    freq = np.geomspace(10.0, 1000.0, 128)
    boost_authority = np.ones_like(freq)
    cut_authority = np.ones_like(freq)
    nan_idx = int(np.argmin(np.abs(freq - 80.0)))
    inf_idx = int(np.argmin(np.abs(freq - 100.0)))
    cut_idx = int(np.argmin(np.abs(freq - 120.0)))
    boost_authority[nan_idx] = np.nan
    boost_authority[inf_idx] = np.inf
    cut_authority[cut_idx] = -np.inf
    st = {
        "authority_boost": boost_authority,
        "authority_cut": cut_authority,
    }

    _, _, result = _run_caps(st)

    assert np.all(np.isfinite(result["boost_cap_db"]))
    assert np.all(np.isfinite(result["cut_cap_db"]))
    assert np.isclose(float(np.asarray(result["boost_cap_db"])[nan_idx]), 0.3)
    assert np.isclose(float(np.asarray(result["boost_cap_db"])[inf_idx]), 6.0)
    assert np.isfinite(float(result["boost_reduction_max_db"]))
    assert np.isfinite(float(result["cut_reduction_max_db"]))
