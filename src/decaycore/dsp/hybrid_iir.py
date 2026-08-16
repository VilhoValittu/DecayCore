# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import scipy.optimize

from .modal_analysis_parts import RoomModeEvent

HYBRID_IIR_POLICY_VERSION = 7

# Vahva modaalinen naytto saa loysentaa luottamuskynnysta, mutta vain naiden
# rajojen sisalla - kyseessa on turvallisuuden ohituspolku, ei viritysnuppi.
_STRONG_MODAL_CUT_PRIORITY = 0.70
_STRONG_MODAL_CONFIDENCE_FLOOR = 0.25


@dataclass(frozen=True)
class HybridIIRPolicy:
    enabled: bool = False
    max_filters_per_channel: int = 3
    min_freq_hz: float = 20.0
    max_freq_hz: float = 200.0
    min_peak_db: float = 4.0
    min_q: float = 3.0
    max_q: float = 12.0
    max_cut_db: float = 6.0
    min_confidence: float = 0.30
    min_gd_excess_ms: float = 10.0
    min_cut_priority: float = 0.0
    max_voice_clarity_risk: float = 0.45

    @classmethod
    def from_config(cls, cfg: Any) -> HybridIIRPolicy:
        return cls(
            enabled=bool(getattr(cfg, "hybrid_iir_enabled", False)),
            max_filters_per_channel=_safe_int(getattr(cfg, "hybrid_iir_max_filters_per_channel", 3), 3),
            min_freq_hz=_safe_float(getattr(cfg, "hybrid_iir_min_freq_hz", 20.0), 20.0),
            max_freq_hz=_safe_float(getattr(cfg, "hybrid_iir_max_freq_hz", 200.0), 200.0),
            min_peak_db=_safe_float(getattr(cfg, "hybrid_iir_min_peak_db", 4.0), 4.0),
            min_q=_safe_float(getattr(cfg, "hybrid_iir_min_q", 3.0), 3.0),
            max_q=_safe_float(getattr(cfg, "hybrid_iir_max_q", 12.0), 12.0),
            max_cut_db=_safe_float(getattr(cfg, "hybrid_iir_max_cut_db", 6.0), 6.0),
            min_confidence=_safe_float(getattr(cfg, "hybrid_iir_min_confidence", 0.30), 0.30),
            min_gd_excess_ms=_safe_float(getattr(cfg, "hybrid_iir_min_gd_excess_ms", 10.0), 10.0),
            min_cut_priority=_safe_float(getattr(cfg, "hybrid_iir_min_cut_priority", 0.0), 0.0),
        ).normalized()

    def normalized(self) -> HybridIIRPolicy:
        min_f = max(1.0, float(self.min_freq_hz))
        max_f = max(min_f + 1.0, float(self.max_freq_hz))
        min_q = max(0.2, float(self.min_q))
        max_q = max(min_q, float(self.max_q))
        return HybridIIRPolicy(
            enabled=bool(self.enabled),
            max_filters_per_channel=max(0, int(self.max_filters_per_channel)),
            min_freq_hz=min_f,
            max_freq_hz=max_f,
            min_peak_db=max(0.0, float(self.min_peak_db)),
            min_q=min_q,
            max_q=max_q,
            max_cut_db=max(0.0, float(self.max_cut_db)),
            min_confidence=float(np.clip(self.min_confidence, 0.0, 1.0)),
            min_gd_excess_ms=max(0.0, float(self.min_gd_excess_ms)),
            min_cut_priority=float(np.clip(self.min_cut_priority, 0.0, 1.0)),
            max_voice_clarity_risk=float(np.clip(self.max_voice_clarity_risk, 0.0, 1.0)),
        )

    def to_signature_dict(self) -> dict[str, Any]:
        p = self.normalized()
        return {
            "policy_v": int(HYBRID_IIR_POLICY_VERSION),
            "enabled": bool(p.enabled),
            "max_filters_per_channel": int(p.max_filters_per_channel),
            "min_freq_hz": float(p.min_freq_hz),
            "max_freq_hz": float(p.max_freq_hz),
            "min_peak_db": float(p.min_peak_db),
            "min_q": float(p.min_q),
            "max_q": float(p.max_q),
            "max_cut_db": float(p.max_cut_db),
            "min_confidence": float(p.min_confidence),
            "min_gd_excess_ms": float(p.min_gd_excess_ms),
            "min_cut_priority": float(p.min_cut_priority),
            "max_voice_clarity_risk": float(p.max_voice_clarity_risk),
        }


@dataclass(frozen=True)
class HybridBiquad:
    freq_hz: float
    q: float
    gain_db: float
    source_peak_db: float
    safe_cut_db: float
    confidence: float
    gd_excess_ms: float
    cut_priority: float
    transfer_cut_db: float = 0.0
    residual_cut_db: float = 0.0
    residual_authority_cap_db: float = 0.0
    fitted_peak_db: float = 0.0
    fit_refined: bool = False

    def to_camilladsp(self) -> dict[str, float | str | bool]:
        return {
            "type": "Peaking",
            "freq": float(self.freq_hz),
            "q": float(self.q),
            "gain": float(self.gain_db),
            "confidence": float(self.confidence),
            "safe_cut_db": float(self.safe_cut_db),
            "source_peak_db": float(self.source_peak_db),
            "fitted_peak_db": float(self.fitted_peak_db),
            "transfer_cut_db": float(self.transfer_cut_db),
            "residual_cut_db": float(self.residual_cut_db),
            "residual_authority_cap_db": float(self.residual_authority_cap_db),
            "cut_priority": float(self.cut_priority),
            "gd_excess_ms": float(self.gd_excess_ms),
            "fit_refined": bool(self.fit_refined),
        }


@dataclass(frozen=True)
class HybridIIRResult:
    enabled: bool
    biquads: tuple[HybridBiquad, ...] = ()
    rejected: tuple[dict[str, Any], ...] = ()
    response: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=complex))
    mag_db: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    phase_rad: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    transfer_mag_db: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    residual_extra_mag_db: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    policy_version: int = HYBRID_IIR_POLICY_VERSION
    mode: str = "residual_aware_magnitude_preconditioning"
    residual_cut_cap_source: str = "policy_max"
    aggregate_residual_depth_db: float = 0.0
    aggregate_trim_factor: float = 1.0

    def to_stats(self) -> dict[str, Any]:
        return {
            "hybrid_iir_enabled": bool(self.enabled),
            "hybrid_iir_policy_version": int(self.policy_version),
            "hybrid_iir_mode": str(self.mode),
            "hybrid_iir_biquads": [b.to_camilladsp() for b in self.biquads],
            "hybrid_iir_rejected": [dict(item) for item in self.rejected],
            "hybrid_iir_filter_count": int(len(self.biquads)),
            "hybrid_iir_fit_refined_count": int(sum(1 for b in self.biquads if b.fit_refined)),
            "hybrid_iir_mag_db": np.asarray(self.mag_db, dtype=float).tolist(),
            "hybrid_iir_phase_rad": np.asarray(self.phase_rad, dtype=float).tolist(),
            "hybrid_iir_transfer_mag_db": np.asarray(self.transfer_mag_db, dtype=float).tolist(),
            "hybrid_iir_residual_extra_mag_db": np.asarray(
                self.residual_extra_mag_db,
                dtype=float,
            ).tolist(),
            "hybrid_iir_transfer_filter_count": int(sum(1 for b in self.biquads if b.transfer_cut_db > 1e-6)),
            "hybrid_iir_residual_cut_count": int(sum(1 for b in self.biquads if b.residual_cut_db > 1e-6)),
            "hybrid_iir_residual_cut_cap_source": str(self.residual_cut_cap_source),
            "hybrid_iir_aggregate_residual_depth_db": float(self.aggregate_residual_depth_db),
            "hybrid_iir_aggregate_trim_factor": float(self.aggregate_trim_factor),
        }


def design_hybrid_iir(
    events: Sequence[RoomModeEvent],
    freq_axis: np.ndarray,
    fs: float,
    policy: HybridIIRPolicy,
    measured_mag_db: np.ndarray | None = None,
    *,
    residual_events: Sequence[RoomModeEvent] | None = None,
    fir_gain_db: np.ndarray | None = None,
    residual_cut_cap_db: np.ndarray | None = None,
    response_freq_axis: np.ndarray | None = None,
) -> HybridIIRResult:
    """Suunnittelee modaaliset peaking-leikkaukset `freq_axis`-akselilla.

    `response_freq_axis` antaa kutsujalle mahdollisuuden syntetisoida
    tulostaulukot suoraan lopulliselle akselille, jolloin samaa vastetta ei
    lasketa kahdesti.
    """
    policy = policy.normalized()
    freq = np.asarray(freq_axis, dtype=float).reshape(-1)
    if not policy.enabled or policy.max_filters_per_channel <= 0 or freq.size == 0:
        return HybridIIRResult(enabled=bool(policy.enabled))

    mag_meas = None
    if measured_mag_db is not None:
        mm = np.asarray(measured_mag_db, dtype=float).reshape(-1)
        if mm.size == freq.size:
            mag_meas = mm
    residual_plan_requested = residual_events is not None or fir_gain_db is not None
    fir_gain = None
    if fir_gain_db is not None:
        fg = np.asarray(fir_gain_db, dtype=float).reshape(-1)
        if fg.size == freq.size and np.all(np.isfinite(fg)):
            fir_gain = fg
    residual_cut_cap = None
    residual_cut_cap_source = "policy_max"
    if residual_cut_cap_db is not None:
        rc = np.asarray(residual_cut_cap_db, dtype=float).reshape(-1)
        if rc.size == freq.size:
            residual_cut_cap = np.clip(
                np.nan_to_num(rc, nan=0.0, posinf=policy.max_cut_db, neginf=0.0),
                0.0,
                float(policy.max_cut_db),
            )
            residual_cut_cap_source = "provided"
        else:
            # Konservatiivinen fallback: tuntematon auktoriteetti -> ei ylimaaraista
            # leikkausta. Merkitaan lahde, jotta tama erottuu telemetriassa
            # tilanteesta jossa data itse ei tue leikkausta.
            residual_cut_cap = np.zeros_like(freq, dtype=float)
            residual_cut_cap_source = "invalid_shape_zero"
    use_residual_plan = fir_gain is not None and residual_events is not None
    if residual_plan_requested and not use_residual_plan:
        return HybridIIRResult(
            enabled=True,
            rejected=tuple(_rejection(event, "invalid_residual_plan") for event in tuple(events or ())),
        )

    selected: list[HybridBiquad] = []
    rejected: list[dict[str, Any]] = []
    ordered = sorted(
        tuple(events or ()), key=lambda ev: (-_event_float(ev, "cut_priority"), -_event_float(ev, "peak_db"))
    )
    for position, event in enumerate(ordered):
        biquad, reason = _candidate_to_biquad(event, policy)
        if biquad is None:
            rejected.append(_rejection(event, reason))
            continue
        if mag_meas is not None:
            biquad = _refine_biquad_against_measurement(biquad, event, freq, mag_meas, float(fs), policy)
        if use_residual_plan:
            biquad = _apply_residual_cut_plan(
                biquad,
                event,
                residual_events=tuple(residual_events or ()),
                freq_axis=freq,
                fir_gain_db=fir_gain,
                residual_cut_cap_db=residual_cut_cap,
                policy=policy,
            )
            if biquad is None:
                rejected.append(_rejection(event, "no_transfer_or_residual_authority"))
                continue
        else:
            biquad = replace(
                biquad,
                residual_cut_db=float(biquad.safe_cut_db),
            )
        selected.append(biquad)
        if len(selected) >= int(policy.max_filters_per_channel):
            # Loput ehdokkaat eivat mahdu budjettiin; kirjataan ne nakyviin sen
            # sijaan etta ne katoaisivat raportista jaljettomiin.
            rejected.extend(_rejection(rest, "max_filters_reached") for rest in ordered[position + 1 :])
            break

    selected, aggregate_depth_db, aggregate_trim_factor = _limit_aggregate_residual_cut(
        selected,
        freq,
        float(fs),
        policy,
    )
    response_axis = freq if response_freq_axis is None else np.asarray(response_freq_axis, dtype=float).reshape(-1)
    return _result_with_response(
        HybridIIRResult(
            enabled=True,
            biquads=tuple(selected),
            rejected=tuple(rejected),
            residual_cut_cap_source=residual_cut_cap_source,
            aggregate_residual_depth_db=float(aggregate_depth_db),
            aggregate_trim_factor=float(aggregate_trim_factor),
        ),
        response_axis,
        float(fs),
    )


def synthesize_hybrid_response(
    biquads: Sequence[HybridBiquad],
    freq_axis: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kokoaa valittujen biquadien vasteen: (response, mag_db, transfer_mag_db, phase_rad).

    `transfer_mag_db` sisaltaa vain FIR:sta siirretyn osuuden, joten
    `mag_db - transfer_mag_db` on se lisavaimennus jonka hybridivaihe tuo
    FIR-suunnitelman paalle. Kaikki taulukot ovat muotoa (bins,).
    """
    freq = np.asarray(freq_axis, dtype=float).reshape(-1)
    response = np.ones(freq.size, dtype=complex)
    transfer_response = np.ones(freq.size, dtype=complex)
    for biquad in biquads:
        response *= peaking_eq_response(freq, float(fs), biquad.freq_hz, biquad.q, biquad.gain_db)
        if float(biquad.transfer_cut_db) > 0.0:
            transfer_response *= peaking_eq_response(
                freq,
                float(fs),
                biquad.freq_hz,
                biquad.q,
                -float(biquad.transfer_cut_db),
            )
    mag_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
    transfer_mag_db = 20.0 * np.log10(np.maximum(np.abs(transfer_response), 1e-12))
    phase_rad = np.unwrap(np.angle(response)) if response.size else np.asarray([], dtype=float)
    return response, mag_db, transfer_mag_db, phase_rad


def _result_with_response(result: HybridIIRResult, freq_axis: np.ndarray, fs: float) -> HybridIIRResult:
    """Taeyttaa tulokseen vastetaulukot annetulla taajuusakselilla."""
    response, mag_db, transfer_mag_db, phase_rad = synthesize_hybrid_response(result.biquads, freq_axis, fs)
    return replace(
        result,
        response=response,
        mag_db=np.nan_to_num(mag_db, nan=0.0, posinf=0.0, neginf=0.0),
        phase_rad=np.nan_to_num(phase_rad, nan=0.0, posinf=0.0, neginf=0.0),
        transfer_mag_db=np.nan_to_num(
            transfer_mag_db,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        residual_extra_mag_db=np.nan_to_num(
            mag_db - transfer_mag_db,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
    )


# Paallekkaiset moodit summautuvat: yksittainen `max_cut_db` ei riita takaamaan
# etta hybridivaiheen tuoma *lisa*vaimennus pysyy katon alla. Trimmauskerroin
# ratkaistaan bisektiolla, koska syvyys on monotoninen kertoimen suhteen.
_AGGREGATE_CUT_TOLERANCE_DB = 1e-6
_AGGREGATE_TRIM_BISECTION_STEPS = 12
_MIN_EFFECTIVE_CUT_DB = 1e-6


def _residual_extra_depth_db(biquads: Sequence[HybridBiquad], freq: np.ndarray, fs: float) -> float:
    """Syvin lisavaimennus (dB, positiivinen) jonka hybridivaihe tuo FIR:n paalle.

    FIR:sta siirretty `transfer_cut_db` on nollasummainen - `filter_pipeline`
    vahentaa saman verran FIR-kayrasta - joten se ei kuulu tahan mittaan.
    """
    freq = np.asarray(freq, dtype=float).reshape(-1)
    if not biquads or freq.size == 0:
        return 0.0
    extra_db = np.zeros(freq.size, dtype=float)
    for biquad in biquads:
        total_mag_db = 20.0 * np.log10(
            np.maximum(
                np.abs(peaking_eq_response(freq, fs, biquad.freq_hz, biquad.q, biquad.gain_db)),
                1e-12,
            )
        )
        if float(biquad.transfer_cut_db) > 0.0:
            transfer_mag_db = 20.0 * np.log10(
                np.maximum(
                    np.abs(peaking_eq_response(freq, fs, biquad.freq_hz, biquad.q, -float(biquad.transfer_cut_db))),
                    1e-12,
                )
            )
            total_mag_db = total_mag_db - transfer_mag_db
        extra_db += total_mag_db
    depth = -float(np.min(np.nan_to_num(extra_db, nan=0.0, posinf=0.0, neginf=0.0)))
    return max(0.0, depth)


def _scaled_residual_biquads(biquads: Sequence[HybridBiquad], scale: float) -> list[HybridBiquad]:
    """Skaalaa vain residual-osuuden; transfer-osuus sailyy sellaisenaan."""
    scaled: list[HybridBiquad] = []
    for biquad in biquads:
        residual_cut_db = float(biquad.residual_cut_db) * float(scale)
        total_cut_db = float(biquad.transfer_cut_db) + residual_cut_db
        scaled.append(
            replace(
                biquad,
                gain_db=-total_cut_db,
                safe_cut_db=total_cut_db,
                residual_cut_db=residual_cut_db,
            )
        )
    return scaled


def _limit_aggregate_residual_cut(
    selected: Sequence[HybridBiquad],
    freq: np.ndarray,
    fs: float,
    policy: HybridIIRPolicy,
) -> tuple[list[HybridBiquad], float, float]:
    """Pitaa yhteenlasketun lisavaimennuksen `max_cut_db`-katon alla.

    Palauttaa (biquadit, mitattu syvyys ennen trimmausta, kaytetty kerroin).
    """
    biquads = list(selected)
    cap_db = float(policy.max_cut_db)
    depth_db = _residual_extra_depth_db(biquads, freq, float(fs))
    if not biquads or depth_db <= cap_db + _AGGREGATE_CUT_TOLERANCE_DB:
        return biquads, depth_db, 1.0

    # depth(scale) on kasvava ja depth(0) == 0, joten bisektio loytaa suurimman
    # sallitun kertoimen deterministisesti ja rajatussa ajassa.
    lo = 0.0
    hi = 1.0
    for _ in range(_AGGREGATE_TRIM_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        if _residual_extra_depth_db(_scaled_residual_biquads(biquads, mid), freq, float(fs)) <= cap_db:
            lo = mid
        else:
            hi = mid
    trimmed = [
        biquad
        for biquad in _scaled_residual_biquads(biquads, lo)
        if float(biquad.transfer_cut_db) + float(biquad.residual_cut_db) > _MIN_EFFECTIVE_CUT_DB
    ]
    return trimmed, depth_db, float(lo)


_MODE_FIT_MIN_POINTS = 9
# Sovitus hyvaksytaan vain, jos loydetty piikki on vahintaan puolet eventin
# huipusta ja jaannos on pieni suhteessa piikkiin - muuten data ei tue mallia.
_MODE_FIT_MIN_GAIN_RATIO = 0.50
_MODE_FIT_MAX_RESIDUAL_PER_GAIN = 0.30


def _peaking_mag_db_model(freq: np.ndarray, fs: float, f0: float, q: float, gain_db: float) -> np.ndarray:
    """RBJ-peaking-magnitudi (dB) mallifunktiona; sallii positiivisen gainin.

    Kayttaa samaa RBJ-ydinta kuin `peaking_eq_coefficients()`, mutta ilman
    gainin leikkausrajausta - moodi sovitetaan boostina mitattuun ylimaaraan.
    """
    response = _biquad_response(freq, fs, _rbj_peaking_coefficients(fs, f0, q, gain_db))
    return 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))


def _refine_biquad_against_measurement(
    biquad: HybridBiquad,
    event: RoomModeEvent,
    freq: np.ndarray,
    mag_db: np.ndarray,
    fs: float,
    policy: HybridIIRPolicy,
) -> HybridBiquad:
    """Tarkentaa biquadin f0:n ja Q:n sovittamalla peaking-mallin mitattuun
    lokaaliin poikkeamaan pienimman neliosumman menetelmalla.

    Leikkaussyvyys (safe_cut) ei muutu. Sovitusrajat pitavat f0:n tapahtuman
    leveyden sisalla ja Q:n policyn rajoissa. Epaonnistunut tai riittamaton
    sovitus palauttaa alkuperaisen biquadin sellaisenaan.
    """
    try:
        f0 = float(biquad.freq_hz)
        if not np.isfinite(f0) or f0 <= 0.0 or fs < 4.0 * f0:
            return biquad
        width_oct = max(_event_float(event, "width_oct"), 1.0 / 24.0)
        half_oct = float(np.clip(1.5 * width_oct, 1.0 / 12.0, 0.5))
        with np.errstate(divide="ignore", invalid="ignore"):
            dist_oct = np.abs(np.log2(np.maximum(freq, 1e-9) / f0))
        window = (dist_oct <= half_oct) & np.isfinite(mag_db) & (freq > 0.0)
        if int(np.count_nonzero(window)) < _MODE_FIT_MIN_POINTS:
            return biquad
        f_w = freq[window]
        y_w = mag_db[window]
        lx = np.log2(f_w)
        lx_c = lx - float(np.median(lx))

        peak_db = max(float(biquad.source_peak_db), 0.5)
        q_init = float(np.clip(float(biquad.q), policy.min_q, policy.max_q))
        f_bound_oct = float(min(width_oct, 1.0 / 6.0))
        # Sovitus ei saa siirtaa korjausta politiikan kaistan ulkopuolelle:
        # `max_freq_hz` on kaventunut huoneen Schroeder-estimaattiin.
        f_lo = max(f0 * 2.0 ** (-f_bound_oct), float(policy.min_freq_hz))
        f_hi = min(f0 * 2.0**f_bound_oct, float(policy.max_freq_hz))
        if not (f_hi > f_lo):
            return biquad
        # Parametrit: (f0, Q, gain_db, offset_db, kallistuma_db_per_oct).
        # Lineaarinen pohja sovitetaan yhdessa piikin kanssa, jotta moodin
        # hannat eivat vuoda pohjaan ja vaarista Q:ta.
        lo = np.asarray([f_lo, policy.min_q, 0.25 * peak_db, -6.0, -12.0], dtype=float)
        hi = np.asarray([f_hi, policy.max_q, 2.0 * peak_db, 6.0, 12.0], dtype=float)

        y_base = float(np.median(np.concatenate([y_w[:3], y_w[-3:]])))

        def residuals(params: np.ndarray) -> np.ndarray:
            model = (
                _peaking_mag_db_model(f_w, fs, params[0], params[1], params[2]) + y_base + params[3] + params[4] * lx_c
            )
            return model - y_w

        x0 = np.clip(np.asarray([f0, q_init, peak_db, 0.0, 0.0], dtype=float), lo, hi)
        result = scipy.optimize.least_squares(residuals, x0, bounds=(lo, hi), max_nfev=200)
        if not result.success or not np.all(np.isfinite(result.x)):
            return biquad
        g_fit = float(result.x[2])
        cost = float(np.sqrt(np.mean(residuals(result.x) ** 2)))
        if g_fit < _MODE_FIT_MIN_GAIN_RATIO * peak_db:
            return biquad
        if not np.isfinite(cost) or cost > _MODE_FIT_MAX_RESIDUAL_PER_GAIN * g_fit:
            return biquad
        return replace(
            biquad,
            freq_hz=float(result.x[0]),
            q=float(np.clip(result.x[1], policy.min_q, policy.max_q)),
            fitted_peak_db=float(g_fit),
            fit_refined=True,
        )
    except (TypeError, ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError):
        return biquad


def _matching_residual_event(
    event: RoomModeEvent,
    residual_events: Sequence[RoomModeEvent],
) -> RoomModeEvent | None:
    """Return the closest residual mode inside the source mode's safe width."""
    source_freq_hz = _event_float(event, "freq_hz")
    if source_freq_hz <= 0.0:
        return None
    safe_width_oct = _event_float(
        event,
        "safe_width_oct",
        _event_float(event, "width_oct"),
    )
    tolerance_oct = float(np.clip(0.5 * safe_width_oct, 1.0 / 24.0, 0.12))
    matched: list[tuple[float, RoomModeEvent]] = []
    for residual in residual_events or ():
        if str(getattr(residual, "kind", "") or "").strip().lower() != "room_mode":
            continue
        residual_freq_hz = _event_float(residual, "freq_hz")
        if residual_freq_hz <= 0.0:
            continue
        distance_oct = abs(np.log2(residual_freq_hz / source_freq_hz))
        if np.isfinite(distance_oct) and distance_oct <= tolerance_oct:
            matched.append((float(distance_oct), residual))
    if not matched:
        return None
    matched.sort(
        key=lambda item: (
            item[0],
            -_event_float(item[1], "cut_priority"),
            -_event_float(item[1], "peak_db"),
        )
    )
    return matched[0][1]


def _apply_residual_cut_plan(
    biquad: HybridBiquad,
    event: RoomModeEvent,
    *,
    residual_events: Sequence[RoomModeEvent],
    freq_axis: np.ndarray,
    fir_gain_db: np.ndarray,
    residual_cut_cap_db: np.ndarray | None,
    policy: HybridIIRPolicy,
) -> HybridBiquad | None:
    """Split the IIR cut into FIR transfer and residual-reduction authority."""
    freq = np.asarray(freq_axis, dtype=float).reshape(-1)
    fir_gain = np.asarray(fir_gain_db, dtype=float).reshape(-1)
    if freq.size < 2 or fir_gain.size != freq.size:
        return None

    fir_gain_at_mode_db = float(
        np.interp(
            float(biquad.freq_hz),
            freq,
            fir_gain,
        )
    )
    transferable_cut_db = max(0.0, -fir_gain_at_mode_db)
    transfer_cut_db = float(
        min(
            float(biquad.safe_cut_db),
            transferable_cut_db,
        )
    )

    residual_cut_db = 0.0
    residual_authority_cap_db = float(policy.max_cut_db)
    if residual_cut_cap_db is not None:
        residual_authority_cap_db = float(
            np.interp(
                float(biquad.freq_hz),
                freq,
                np.asarray(residual_cut_cap_db, dtype=float),
            )
        )
    residual_event = _matching_residual_event(event, residual_events)
    if residual_event is not None:
        residual_biquad, _reason = _candidate_to_biquad(residual_event, policy)
        if residual_biquad is not None:
            supported_peak_db = (
                float(biquad.fitted_peak_db)
                if biquad.fit_refined and biquad.fitted_peak_db > 0.0
                else float(biquad.source_peak_db)
            )
            total_cut_cap_db = float(
                min(
                    float(policy.max_cut_db),
                    supported_peak_db,
                )
            )
            remaining_cut_db = max(0.0, total_cut_cap_db - transfer_cut_db)
            residual_cut_db = float(
                min(
                    float(residual_biquad.safe_cut_db),
                    remaining_cut_db,
                    residual_authority_cap_db,
                )
            )

    total_cut_db = float(transfer_cut_db + residual_cut_db)
    if total_cut_db <= 1e-6:
        return None
    return replace(
        biquad,
        gain_db=-total_cut_db,
        safe_cut_db=total_cut_db,
        transfer_cut_db=transfer_cut_db,
        residual_cut_db=residual_cut_db,
        residual_authority_cap_db=residual_authority_cap_db,
    )


def peaking_eq_response(freq_axis: np.ndarray, fs: float, freq_hz: float, q: float, gain_db: float) -> np.ndarray:
    return _biquad_response(
        freq_axis,
        fs,
        peaking_eq_coefficients(float(fs), float(freq_hz), float(q), float(gain_db)),
    )


def peaking_eq_coefficients(
    fs: float, freq_hz: float, q: float, gain_db: float
) -> tuple[float, float, float, float, float, float]:
    """RBJ-peaking leikkaukseksi rajattuna - hybrid-IIR ei koskaan boostaa."""
    return _rbj_peaking_coefficients(fs, freq_hz, q, min(0.0, _safe_float(gain_db, 0.0)))


def _rbj_peaking_coefficients(
    fs: float, freq_hz: float, q: float, gain_db: float
) -> tuple[float, float, float, float, float, float]:
    """Normalisoidut RBJ-peaking-kertoimet ilman gainin merkkirajausta."""
    fs = max(1.0, float(fs))
    f0 = float(np.clip(freq_hz, 1e-6, fs * 0.499))
    q = max(1e-6, float(q))
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * q)
    a_amp = 10.0 ** (float(gain_db) / 40.0)
    b0 = 1.0 + alpha * a_amp
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * a_amp
    a0 = 1.0 + alpha / max(a_amp, 1e-12)
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / max(a_amp, 1e-12)
    return (float(b0 / a0), float(b1 / a0), float(b2 / a0), 1.0, float(a1 / a0), float(a2 / a0))


def _biquad_response(
    freq_axis: np.ndarray,
    fs: float,
    coefficients: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Biquadin kompleksivaste taajuusakselilla; (bins,) sisaan, (bins,) ulos."""
    b0, b1, b2, a0, a1, a2 = coefficients
    freq = np.asarray(freq_axis, dtype=float).reshape(-1)
    omega = 2.0 * np.pi * np.clip(freq, 0.0, max(1.0, float(fs) / 2.0)) / max(float(fs), 1.0)
    z1 = np.exp(-1j * omega)
    z2 = np.exp(-2j * omega)
    den = a0 + a1 * z1 + a2 * z2
    num = b0 + b1 * z1 + b2 * z2
    with np.errstate(divide="ignore", invalid="ignore"):
        h = num / np.where(np.abs(den) > 1e-18, den, 1.0)
    return np.nan_to_num(h, nan=1.0, posinf=1.0, neginf=1.0)


def _candidate_to_biquad(event: RoomModeEvent, policy: HybridIIRPolicy) -> tuple[HybridBiquad | None, str]:
    freq = _event_float(event, "freq_hz")
    peak = _event_float(event, "peak_db")
    confidence = _event_float(event, "confidence")
    gd_excess = _event_float(event, "gd_excess_ms")
    cut_priority = _event_float(event, "cut_priority", _event_float(event, "correction_priority"))
    voice_risk = _event_float(event, "voice_clarity_risk")
    kind = str(getattr(event, "kind", "") or "").strip().lower()
    if not (policy.min_freq_hz <= freq <= policy.max_freq_hz):
        return None, "outside_frequency_range"
    if peak < policy.min_peak_db:
        return None, "peak_below_threshold"
    modal_conf_floor = float(policy.min_confidence)
    strong_modal_evidence = (
        kind == "room_mode"
        and cut_priority >= _STRONG_MODAL_CUT_PRIORITY
        and gd_excess >= policy.min_gd_excess_ms
        and peak >= policy.min_peak_db
    )
    if strong_modal_evidence:
        modal_conf_floor = min(modal_conf_floor, _STRONG_MODAL_CONFIDENCE_FLOOR)
    if confidence < modal_conf_floor:
        return None, "confidence_below_threshold"
    if gd_excess < policy.min_gd_excess_ms:
        return None, "gd_evidence_below_threshold"
    if voice_risk > policy.max_voice_clarity_risk:
        return None, "voice_clarity_risk"
    if cut_priority < policy.min_cut_priority:
        return None, "modal_priority_below_threshold"

    q_raw = _event_float(event, "q_estimate")
    if not np.isfinite(q_raw) or q_raw <= 0.0:
        width_hz = _event_float(event, "width_hz")
        q_raw = freq / max(width_hz, 1e-9)
    if q_raw < policy.min_q:
        return None, "q_below_threshold"
    q = float(np.clip(q_raw, policy.min_q, policy.max_q))

    safe_cut = min(_event_float(event, "safe_cut_db"), policy.max_cut_db, peak)
    if safe_cut <= 0.0:
        return None, "safe_cut_unavailable"
    gain_db = -float(safe_cut)
    return (
        HybridBiquad(
            freq_hz=float(freq),
            q=float(q),
            gain_db=float(gain_db),
            source_peak_db=float(peak),
            safe_cut_db=float(safe_cut),
            confidence=float(confidence),
            gd_excess_ms=float(gd_excess),
            cut_priority=float(cut_priority),
        ),
        "",
    )


def _rejection(event: RoomModeEvent, reason: str) -> dict[str, Any]:
    return {
        "freq_hz": _event_float(event, "freq_hz"),
        "peak_db": _event_float(event, "peak_db"),
        "q_estimate": _event_float(event, "q_estimate"),
        "confidence": _event_float(event, "confidence"),
        "gd_excess_ms": _event_float(event, "gd_excess_ms"),
        "reason": str(reason),
    }


def _event_float(event: Any, key: str, default: float = 0.0) -> float:
    return _safe_float(getattr(event, key, default), default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        out = float(default)
    return out if np.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)
