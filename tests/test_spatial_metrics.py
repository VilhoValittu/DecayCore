# DecayCore
# Copyright (c) 2026 Vilho Valittu
# All rights reserved.
#
# This file is part of the proprietary DecayCore codebase.
# No copying, redistribution, commercial reuse, or removal of attribution
# is permitted without prior written permission.

"""Tests for dsp/spatial_metrics.py (IACC computation)."""

import math

import numpy as np
import pytest

from decaycore.dsp.spatial_metrics import IACCResult, compute_iacc


FS = 44100


def _dirac(n: int, pos: int = 0) -> np.ndarray:
    ir = np.zeros(n)
    ir[pos] = 1.0
    return ir


def _white_noise(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(float)


class TestComputeIACC:
    def test_identical_signals_returns_one(self):
        """IACC_A of identical signals should be ~1.0."""
        ir = _white_noise(4096)
        result = compute_iacc(ir, ir.copy(), FS)
        assert math.isfinite(result.all)
        assert result.all == pytest.approx(1.0, abs=1e-6)

    def test_inverted_signal_returns_one(self):
        """Inverted signal: cross-corr peak is still 1 (we take abs)."""
        ir = _white_noise(4096)
        result = compute_iacc(ir, -ir, FS)
        assert math.isfinite(result.all)
        assert result.all == pytest.approx(1.0, abs=1e-6)

    def test_independent_signals_near_zero(self):
        """Independent white noise channels should produce low IACC."""
        ir_l = _white_noise(65536, seed=1)
        ir_r = _white_noise(65536, seed=2)
        result = compute_iacc(ir_l, ir_r, FS)
        assert math.isfinite(result.all)
        assert result.all < 0.15

    def test_tau_ms_for_delayed_dirac(self):
        """Delayed dirac: tau_ms should match the delay."""
        delay_samples = 441  # 10 ms @ 44100 Hz
        ir_l = _dirac(4096, pos=1000)
        ir_r = _dirac(4096, pos=1000 + delay_samples)
        result = compute_iacc(ir_l, ir_r, FS)
        assert math.isfinite(result.tau_ms)
        assert result.tau_ms == pytest.approx(-10.0, abs=0.5)

    def test_returns_iacc_result_type(self):
        ir = _white_noise(2048)
        result = compute_iacc(ir, ir, FS)
        assert isinstance(result, IACCResult)

    def test_short_input_returns_nan(self):
        """Arrays shorter than 4 samples should yield NaN fields."""
        result = compute_iacc(np.array([1.0, 0.5]), np.array([1.0, 0.5]), FS)
        assert not math.isfinite(result.all)
        assert not math.isfinite(result.early)

    def test_empty_input_returns_nan(self):
        result = compute_iacc(np.array([]), np.array([]), FS)
        assert not math.isfinite(result.all)

    def test_zero_energy_returns_nan(self):
        result = compute_iacc(np.zeros(1024), np.zeros(1024), FS)
        assert not math.isfinite(result.all)

    def test_iacc_early_and_late_finite_for_long_ir(self):
        """For an IR longer than 80 ms, early and late should both be finite."""
        n = int(0.2 * FS)   # 200 ms
        ir_l = _white_noise(n, seed=10)
        ir_r = _white_noise(n, seed=11)
        result = compute_iacc(ir_l, ir_r, FS)
        assert math.isfinite(result.early)
        assert math.isfinite(result.late)

    def test_all_values_in_range(self):
        """IACC values must be in [0, 1]."""
        ir_l = _white_noise(8192, seed=20)
        ir_r = _white_noise(8192, seed=21)
        result = compute_iacc(ir_l, ir_r, FS)
        for attr in ("early", "late", "all"):
            val = getattr(result, attr)
            if math.isfinite(val):
                assert 0.0 <= val <= 1.0, f"{attr}={val} out of range"

    def test_asymmetric_length_truncates_to_shorter(self):
        """If IRs have different lengths, should still run without error."""
        ir_l = _white_noise(4096, seed=30)
        ir_r = _white_noise(2048, seed=31)
        result = compute_iacc(ir_l, ir_r, FS)
        assert isinstance(result, IACCResult)
