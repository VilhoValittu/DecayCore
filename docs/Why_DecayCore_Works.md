---
title: Why DecayCore Works
description: Practical explanation of DecayCore's time-domain-first alignment, confidence weighting, DSP guardrails, phase safety, and temporal decay control.
hide_page_heading: true
---

# Why DecayCore Works

## In brief

DecayCore corrects broad, repeatable problems and limits correction where the measurement is weak. It treats magnitude, timing, phase, and low-frequency decay as related but separate tasks.

Controlled cuts and bounded shaping do most of the work. Boost is limited to regions where the measurement supports it.

## 1) Time-domain first, then frequency-domain shaping

In-room measurements combine multiple phenomena:
- propagation delay (time-of-flight)
- loudspeaker/crossover behavior
- room reflections and comb filtering
- modal energy storage (ringing)

A pure inverse-EQ approach mixes these into one problem and tends to overcorrect.
DecayCore separates them:
- TOF is aligned before phase analysis
- magnitude correction is bounded
- phase is reconstructed with explicit strategy and limits
- decay is treated separately via TDC

## 2) Confidence-weighted correction instead of blind inversion

Not every dip is correctable. Reflection cancellations and low-confidence bins can move with tiny mic-position changes.

DecayCore reduces overfit using:
- confidence-aware shaping
- smoothing and regularization
- optional Adaptive FDW (A-FDW)
- Confidence Pull behavior in uncertain regions

Together, these controls reduce correction demand in regions that change with microphone position.

## 3) Explicit guardrails keep filters physically sane

Most failures in room correction come from unbounded boosts, deep-null chasing, and steep local corrections.
DecayCore uses explicit limits, including:
- `max_boost_db`, `max_cut_db`
- `max_slope_db_per_oct` and optional split slope limits for boost vs cut
- `low_bass_cut_hz` / excursion-oriented low-bass safety
- `reg_strength` to avoid null-filling behavior

These limits contain gain and slope demand, especially around deep nulls.

## 4) Phase reconstruction with layered safety

Phase correction can improve transients, but only if it is constrained.

DecayCore provides multiple safety layers:
- `phase_limit` to bound correction bandwidth
- FDW / A-FDW to reduce reflection-driven phase noise
- Mixed-phase excess correction fade (LF full correction -> HF no correction)
- Mixed-only guards: `max_excess_delay_ms` and `max_pre_ringing_db`
- adaptive excess-phase clamp behavior for robust operation
- conditional GD spike guarding in bass-focused high-risk cases

## 5) TDC targets ringing that EQ alone cannot fix

Room modes are not only amplitude peaks; they are time-domain energy storage.

Temporal Decay Control (TDC):
- shapes decay behavior directly
- is independent from static magnitude EQ
- includes bounded controls (`tdc_strength`, `tdc_max_reduction_db`, optional `tdc_slope_db_per_oct`)

## 6) Headroom and channel consistency are part of DSP safety

A "good" filter that clips is still a bad filter.

DecayCore uses an auto-headroom model:
- output attenuation follows realized max boost plus margin
- normalization remains optional as extra safety
- stereo-link shares the leveling anchor and, in shared mode, the window; hybrid mode can retain channel-specific windows, while final auto-gain remains channel-specific

## 7) Reproducibility across fs/taps and input formats

For comparisons across settings, DecayCore provides:
- comparison mode with fixed analysis grid
- multi-rate generation with auto-taps time-length mapping
- deterministic WAV policy aligned with TXT baseline behavior

## 8) Operational transparency

To inspect a run, use:

- System Health checks for risky settings and missing sources
- Summary for the version stamp and effective parameters
- run timing breakdown for read, DSP, export, and render stages

When comparing or adjusting filters:

1. Start with bounded magnitude correction and sane headroom.
2. Add phase only within clean confidence and frequency limits.
3. Apply TDC carefully for modal decay problems.
4. Use comparison mode when doing A/B decisions.
5. Re-check output diagnostics and Summary before final deployment.

See the [TDC impulse example]({{ '/pics/tdc_impulse_example.png' | relative_url }}).
