---
title: IR Export Windowing vs DSP Correction
description: Explain how DecayCore separates FIR correction design from final impulse-response export windowing for reproducible comparisons.
hide_page_heading: true
---

# IR Export Windowing vs DSP Correction

## In brief

DSP correction designs the filter before export windowing places and tapers its impulse in time. Changing the window does not rerun target matching, but it can change the response of the exported FIR.

## DSP Correction (what is corrected)

Analysis and optimization define the intended correction before the FIR is written to disk. This stage includes:

- target curve fitting
- magnitude correction
- phase / excess-phase reconstruction
- FDW / Adaptive FDW (A-FDW)
- Temporal Decay Control (TDC)
- auto-leveling and safety limits

Changing the export window leaves these design decisions unchanged.

## IR Export Windowing (how the FIR is written)

IR export windowing changes the impulse's onset, symmetry, or tail. Tapering or trimming nonzero samples can also change the exported filter's frequency and phase response. The target and correction settings remain the same.

Supported modes (UI):

- `auto` - automatic window selection (default)
- `rew_asym` - REW-style asymmetric (causal) window

Legacy config values (`off`, `rew_sym`) are still accepted when set directly in config files,
but are no longer exposed in the UI.

## Why this distinction matters

The same DSP design can produce FIR files with different time alignment, impulse symmetry, and practical latency. Their realized responses may also differ if the window changes nonzero samples.

This is useful for:

- matching different convolver requirements
- minimizing audible latency
- comparing results directly with REW
- controlled listening tests and A/B evaluation

DecayCore records the windowing type in exported filenames so you can tell which FIR you loaded. Compare the exported filters when evaluating different window settings.
