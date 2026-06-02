# DecayCore
# Copyright (c) 2026 Vilho Valittu
# All rights reserved.
#
# This file is part of the proprietary DecayCore codebase.
# No copying, redistribution, commercial reuse, or removal of attribution
# is permitted without prior written permission.

"""Tests for P4 phase authority gating (phase_authority.py)."""

from __future__ import annotations

import numpy as np
import pytest

from decaycore.dsp.phase_authority import (
    apply_phase_authority_gating,
    build_phase_authority_gain,
)


FREQ = np.geomspace(10.0, 20000.0, 2048)


# ---------------------------------------------------------------------------
# 1. Missing authority preserves old behaviour
# ---------------------------------------------------------------------------

def test_missing_authority_preserves_behaviour():
    gain = build_phase_authority_gain(
        FREQ,
        authority_phase=None,
        phase_limit_hz=600.0,
        disable_above_hz=1200.0,
    )
    assert gain[FREQ < 500].mean() > 0.95
    assert gain[FREQ > 1500].max() <= 0.05


# ---------------------------------------------------------------------------
# 2. Low phase authority suppresses phase correction
# ---------------------------------------------------------------------------

def test_low_authority_suppresses():
    authority = np.ones_like(FREQ)
    authority[(FREQ > 90) & (FREQ < 130)] = 0.0

    gain = build_phase_authority_gain(
        FREQ,
        authority_phase=authority,
        phase_limit_hz=600.0,
        disable_above_hz=1200.0,
        gamma=1.20,
        smooth_oct=None,
    )
    near_110 = gain[(FREQ > 100) & (FREQ < 120)]
    near_40 = gain[(FREQ > 35) & (FREQ < 50)]
    assert near_110.mean() < 0.15
    assert near_40.mean() > 0.8


# ---------------------------------------------------------------------------
# 3. Modal support rescues low authority partly
# ---------------------------------------------------------------------------

def test_modal_support_rescues():
    authority = np.zeros_like(FREQ)
    modal = np.zeros_like(FREQ)
    modal[(FREQ > 40) & (FREQ < 50)] = 1.0

    gain = build_phase_authority_gain(
        FREQ,
        authority_phase=authority,
        authority_modal_support=modal,
        phase_limit_hz=600.0,
        disable_above_hz=1200.0,
        smooth_oct=None,
    )
    near_45 = gain[(FREQ > 42) & (FREQ < 48)]
    assert near_45.mean() > 0.10
    assert near_45.mean() < 0.60


# ---------------------------------------------------------------------------
# 4. Reflection risk suppresses phase correction
# ---------------------------------------------------------------------------

def test_reflection_risk_suppresses():
    authority = np.ones_like(FREQ)
    reflection = np.zeros_like(FREQ)
    reflection[(FREQ > 250) & (FREQ < 350)] = 1.0

    gain = build_phase_authority_gain(
        FREQ,
        authority_phase=authority,
        authority_reflection_risk=reflection,
        phase_limit_hz=600.0,
        disable_above_hz=1200.0,
        smooth_oct=None,
    )
    near_300 = gain[(FREQ > 280) & (FREQ < 320)].mean()
    near_150 = gain[(FREQ > 130) & (FREQ < 170)].mean()
    assert near_300 < near_150


# ---------------------------------------------------------------------------
# 5. Null risk suppresses phase correction
# ---------------------------------------------------------------------------

def test_null_risk_suppresses():
    authority = np.ones_like(FREQ)
    null = np.zeros_like(FREQ)
    null[(FREQ > 100) & (FREQ < 140)] = 1.0

    gain = build_phase_authority_gain(
        FREQ,
        authority_phase=authority,
        authority_null_risk=null,
        phase_limit_hz=600.0,
        disable_above_hz=1200.0,
        smooth_oct=None,
    )
    near_120 = gain[(FREQ > 110) & (FREQ < 130)].mean()
    near_60 = gain[(FREQ > 55) & (FREQ < 65)].mean()
    assert near_120 < near_60


# ---------------------------------------------------------------------------
# 6. Frequency ceiling works
# ---------------------------------------------------------------------------

def test_frequency_ceiling():
    authority = np.ones_like(FREQ)

    gain = build_phase_authority_gain(
        FREQ,
        authority_phase=authority,
        phase_limit_hz=600.0,
        disable_above_hz=1200.0,
        smooth_oct=None,
    )
    assert gain[FREQ < 500].mean() > 0.95
    assert gain[FREQ > 1300].max() <= 0.05
    mid_gain = gain[(FREQ > 700) & (FREQ < 1100)]
    assert mid_gain.min() >= 0.0
    assert mid_gain.max() <= 1.0


# ---------------------------------------------------------------------------
# 7. Integration: authority suppresses extra_phase energy in low-authority band
# ---------------------------------------------------------------------------

def test_integration_suppresses_extra_phase_in_low_authority_band():
    rng = np.random.default_rng(42)
    extra_phase = rng.uniform(-0.5, 0.5, size=len(FREQ))

    authority = np.ones_like(FREQ)
    authority[(FREQ > 90) & (FREQ < 130)] = 0.0

    class _FakeCfg:
        phase_authority_enable = True
        phase_authority_gamma = 1.20
        phase_authority_min_gain = 0.0
        phase_authority_soft_floor = 0.20
        phase_authority_smooth_oct = None
        phase_authority_disable_above_hz = 1200.0
        phase_limit = 600.0
        phase_c_max = None

    st = {"authority_phase": authority.tolist()}

    gated = apply_phase_authority_gating(
        FREQ, extra_phase, _FakeCfg(), st, logger=None
    )

    mask = (FREQ > 100) & (FREQ < 120)
    energy_before = float(np.mean(np.abs(extra_phase[mask])))
    energy_after = float(np.mean(np.abs(gated[mask])))
    assert energy_after < energy_before * 0.25

    # outside the low-authority band correction should be mostly preserved
    mask_outside = (FREQ > 35) & (FREQ < 50)
    energy_out_before = float(np.mean(np.abs(extra_phase[mask_outside])))
    energy_out_after = float(np.mean(np.abs(gated[mask_outside])))
    assert energy_out_after > energy_out_before * 0.7

    # telemetry written
    assert "phase_authority_enabled" in st
    assert st["phase_authority_enabled"] is True
    assert st["phase_authority_suppressed_bins"] >= 0


# ---------------------------------------------------------------------------
# 8. Disabled mode returns extra_phase unchanged
# ---------------------------------------------------------------------------

def test_disabled_returns_unchanged():
    rng = np.random.default_rng(7)
    extra_phase = rng.uniform(-1.0, 1.0, size=len(FREQ))

    class _FakeCfgOff:
        phase_authority_enable = False

    st: dict = {}
    result = apply_phase_authority_gating(FREQ, extra_phase, _FakeCfgOff(), st, logger=None)
    np.testing.assert_array_equal(result, extra_phase)
    assert st.get("phase_authority_enabled") is False
