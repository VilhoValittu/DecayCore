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


def test_acoustic_authority_caps_disabled_preserves_user_boost_cap_with_low_authority():
    st = {
        "authority_boost": np.zeros(128, dtype=float),
        "authority_cut": np.zeros(128, dtype=float),
    }
    cfg = SimpleNamespace(acoustic_authority_limits_enable=False)
    base_boost_cap = np.full(128, 12.0, dtype=float)

    freq, _, result = _run_caps(st, cfg=cfg, boost_cap=base_boost_cap)

    assert freq.shape == base_boost_cap.shape
    assert np.allclose(result["boost_cap_db"], base_boost_cap)
    assert int(result["boost_reduced_bins"]) == 0


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
    # 1/4 oktaavin dippi 80 Hz ympärillä: leveämpi kuin 1/9 okt caps-tasoitus,
    # joten dipin keskikohta säilyttää syvyytensä.
    idx = int(np.argmin(np.abs(freq - 80.0)))
    dip_band = (freq >= 80.0 * 2 ** -0.125) & (freq <= 80.0 * 2 ** 0.125)
    st["authority_boost"][dip_band] = 0.1

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
    # 1/4 oktaavin kaista 45 Hz ympärillä, leveämpi kuin 1/9 okt caps-tasoitus.
    idx = int(np.argmin(np.abs(freq - 45.0)))
    peak_band = (freq >= 45.0 * 2 ** -0.125) & (freq <= 45.0 * 2 ** 0.125)
    st["authority_cut"][peak_band] = 0.95

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
    # Tasoitus pois, jotta sanitoinnin per-bin-vaikutus on tarkasti assertoitavissa.
    cfg = SimpleNamespace(authority_caps_smooth_oct=0.0)

    _, _, result = _run_caps(st, cfg=cfg)

    assert np.all(np.isfinite(result["boost_cap_db"]))
    assert np.all(np.isfinite(result["cut_cap_db"]))
    assert np.isclose(float(np.asarray(result["boost_cap_db"])[nan_idx]), 0.3)
    assert np.isclose(float(np.asarray(result["boost_cap_db"])[inf_idx]), 6.0)
    assert np.isfinite(float(result["boost_reduction_max_db"]))
    assert np.isfinite(float(result["cut_reduction_max_db"]))


def _run_caps_hf(st, cfg):
    """Aja caps-laskenta täydellä kaistalla (20 Hz - 20 kHz), mask_c koko kaista."""
    freq = np.geomspace(20.0, 20000.0, 1024)
    mask = np.ones_like(freq, dtype=bool)
    result = _apply_acoustic_authority_caps(
        cfg=cfg,
        cfg_reader=CfgReader(cfg),
        st=st,
        logger=logging.getLogger(__name__),
        freq_axis=freq,
        mask_c=mask,
        boost_cap_db=np.full_like(freq, 5.0, dtype=float),
        max_cut_db=24.0,
    )
    return freq, result


def test_acoustic_authority_caps_smooths_jagged_hf_authority():
    """Regressio: per-bin rosoinen authority ei saa tuottaa sahalaitaisia kattoja.

    Bugi: authority_caps_smooth_oct luettiin mutta tasoitusta ei sovellettu,
    jolloin HF:n kampasuodinrakenteesta syntyvä rosoinen authority leikkasi
    sileän korjauksen sahalaitaiseksi, kun mag_c_max nostettiin 20 kHz:iin.
    """
    freq = np.geomspace(20.0, 20000.0, 1024)
    jagged = np.where(np.arange(freq.size) % 2 == 0, 0.9, 0.2)

    st_smooth = {"authority_boost": jagged.copy(), "authority_cut": jagged.copy()}
    _, result_smooth = _run_caps_hf(st_smooth, SimpleNamespace())

    st_raw = {"authority_boost": jagged.copy(), "authority_cut": jagged.copy()}
    _, result_raw = _run_caps_hf(st_raw, SimpleNamespace(authority_caps_smooth_oct=0.0))

    assert bool(result_smooth["enabled"]) is True
    assert bool(result_raw["enabled"]) is True

    boost_smooth = np.asarray(result_smooth["boost_cap_db"], dtype=float)
    boost_raw = np.asarray(result_raw["boost_cap_db"], dtype=float)
    cut_smooth = np.asarray(result_smooth["cut_cap_db"], dtype=float)
    cut_raw = np.asarray(result_raw["cut_cap_db"], dtype=float)

    # Escape hatch (smooth_oct=0) säilyttää bin-tason sahan, oletus tasoittaa sen.
    # Reunabinit jätetään pois: tasoituksen edge-pad jättää niihin loivan rampin.
    inner = slice(16, -16)
    assert float(np.max(np.abs(np.diff(boost_raw[inner])))) > 2.0
    assert float(np.max(np.abs(np.diff(boost_smooth[inner])))) < 0.1
    assert float(np.max(np.abs(np.diff(cut_raw[inner])))) > 2.0
    assert float(np.max(np.abs(np.diff(cut_smooth[inner])))) < 0.5

    # Tasoitus ei saa nostaa kattoja yli perustason eikä rikkoa rajoja.
    assert np.all(boost_smooth <= 5.0 + 1e-9)
    assert np.all(boost_smooth >= 0.0)
    assert np.all(cut_smooth <= 24.0 + 1e-9)
