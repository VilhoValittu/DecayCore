---
title: DecayCore Stability and Reproducibility
description: Explain DecayCore's deterministic pipeline, bounded correction, input robustness, comparable evaluation, phase safeguards, and audit trail.
hide_page_heading: true
permalink: /how-it-works/stability/
---

# Stability and Reproducibility

## In brief

Repeatable results require identical measurements, settings, application behavior, and comparison context. DecayCore combines deterministic processing, bounded correction, versioned compatibility, and exported diagnostics to make differences inspectable.

The limits reduce the chance that a small measurement or setting change produces a large filter change.

## 1) Deterministic pipeline and versioned behavior

Given the same:
- DecayCore version
- input files
- settings
- base fs/taps or comparison-grid settings

DecayCore produces repeatable outputs.

The pipeline keeps its behavior deterministic for fixed inputs and settings.
Automatic mode may use seeded Optuna-backed search, but the seed is derived deterministically from the measurement/signature context so identical runs remain reproducible.
Cache reuse in automatic-mode target and preset search does not change this.

## 2) Input robustness: TXT and WAV/IR

DecayCore accepts REW text exports and WAV impulse responses. WAV handling includes:
- deterministic WAV parsing policy aligned to TXT baseline behavior
- path sanitation and local WAV existence validation
- higher minimum FFT robustness for WAV IR -> FR conversion
- alignment guard between delay estimate and impulse-peak alignment

## 3) Bounded correction prevents inverse-filter runaway

DecayCore keeps each stage bounded:

Magnitude bounds:
- `max_boost_db`, `max_cut_db`
- low-bass safety controls
- smoothing and regularization (`filter_smooth`, `reg_strength`)

Shape bounds:
- slope limits (`max_slope_db_per_oct`)
- optional asymmetric slope limits for boost vs cut

Decay bounds:
- TDC strength and max reduction limits
- optional TDC slope limit

## 4) Confidence-guided correction improves repeatability

Room data confidence is frequency-dependent.
DecayCore uses confidence-aware behavior to avoid overreacting to artifacts:
- Confidence Pull logic
- A-FDW support
- smoothing that favors robust trends over narrow features

Practical effect: small mic-position changes usually alter details, not the whole correction profile.

## 5) Phase-domain safeguards

Phase processing is constrained by design:
- `phase_limit` bounds correction bandwidth
- Mixed-phase excess correction is faded from LF to HF
- Mixed-only limits cap excess delay and pre-ringing risk
- adaptive excess-phase clamping and conditional GD spike guards

## 6) Headroom reproducibility and stereo consistency

Level handling is deterministic and tied to realized correction:
- auto-headroom model computes attenuation from realized max boost + margin
- stereo-linked runs share the leveling anchor and window, and both channels get the lower of the two broadband gains
- plot-only compensation keeps analysis visuals readable while preserving output safety

## 7) Comparable evaluation across fs/taps

Changing fs/taps can change analysis grids even when audible behavior is similar.
DecayCore addresses this with:
- comparison mode (fixed reference grid for scoring/match)
- multi-rate export with time-length-aware tap mapping

Scores then stay comparable when only the grid changes.

## 8) Observability and audit trail

DecayCore records:
- Summary version stamp and key effective parameters
- run timing breakdown (Read, DSP, ZIP/PNG, Render, Total)
- expanded System Health warnings for source completeness and risky settings

If two runs differ, diagnostics help explain why.

## Reproducible-run checklist

1. Use the same single pinned app version.
2. Use identical measurement sources and source type (TXT vs WAV) per comparison.
3. Keep base fs/taps fixed, or enable comparison mode.
4. Keep correction guardrails fixed (boost/cut/slope/regularization/phase limit).
5. Keep phase strategy fixed.
6. Keep headroom margin and stereo-link policy fixed.
7. Verify System Health is clean before exporting.
8. Compare Summary outputs and timing data when validating repeated runs.
