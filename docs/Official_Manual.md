---
title: DecayCore Technical Reference
description: Technical reference for the DecayCore DSP pipeline, correction controls, phase strategies, safety policies, and exported filters.
hide_page_heading: true
---

# DecayCore Technical Reference

## In brief

DecayCore separates measurement alignment, target matching, phase construction, decay control, safety limiting, FIR synthesis, and export. This page describes that pipeline and the controls that affect it. For operating instructions, use the [User Manual]({{ '/User_Manual.html' | relative_url }}).

## 1. Processing model

DecayCore treats a room measurement as several related problems rather than one response to invert:

- propagation delay and channel alignment
- broad magnitude error relative to a target
- unreliable reflection cancellations and narrow detail
- minimum- and excess-phase behavior
- low-frequency energy decay
- playback headroom and FIR realization

The correction remains cuts-first. Boost is allowed only within configured limits and acoustic-confidence policy; deep-null filling is not a design goal.

## 2. Pipeline order

The high-level processing sequence is:

1. Read and validate Left and Right measurements.
2. Establish frequency grids, timing anchors, and measurement metadata.
3. Remove time of flight before excess-phase analysis.
4. Align measurements and the selected target for comparison.
5. Estimate acoustic confidence and apply smoothing or frequency-dependent windowing.
6. Form bounded magnitude correction.
7. Apply low-bass, excursion, slope, and boost/cut guards.
8. Apply optional Temporal Decay Control and Hybrid IIR policy.
9. Construct the selected phase response and apply phase safety guards.
10. Synthesize, align, gain-stage, and window the FIR impulse.
11. Validate the final response and export filters, configurations, plots, and summary data.

Automatic mode wraps this pipeline in target selection, candidate search, ranking, refinement, and final validation. It does not replace the DSP guards.

## 3. Measurement inputs

### Text frequency response

Text input supplies frequency, magnitude, and phase. Phase authority depends on valid, consistently referenced phase data. Header and comment rows are ignored.

### WAV impulse response

WAV input preserves time-domain information used for alignment and metadata extraction. Files from one measurement set must keep common timing and relative gain. Per-channel normalization or independent peak movement destroys information needed by stereo and bass-integration workflows.

### Measurement metadata

Packaged measurement sessions can provide repeat quality, timing, harmonic curves, interaural cross-correlation evidence, and RT60 estimates. Features degrade conservatively when optional metadata is absent; missing evidence must not be invented from unrelated cached data.

## 4. Target formation

Manual modes use the selected built-in or custom target. Automatic mode supports three strategies:

- **Auto: search best built-in** compares supported built-in curves using the same measurement context.
- **Adaptive: derive target from room acoustics** makes small, bounded low-frequency changes to a Harman6 baseline from broad stereo evidence.
- **Use selected target curve from Target page** locks the manually selected target.

Target leveling and correction are separate operations. Leveling establishes the reference offset; magnitude correction then shapes the measured response toward the target within the permitted band and limits.

## 5. Magnitude controls

| Control | Technical role |
|---|---|
| Correction lower/upper limit | Bounds the frequency region where correction is active. |
| Max boost / Max cut | Limits positive and negative correction demand. |
| Smoothing | Reduces sensitivity to narrow response detail. |
| Regularization | Reduces inverse demand in weak or uncertain regions. |
| Ignore local correction below | Removes small local correction features while retaining broader shape. |
| Slope limits | Bound rapid correction changes per octave. |
| Low-bass boost lock | Blocks boost below the selected frequency while retaining permitted cuts. |
| Excursion Protection | Reduces risky low-frequency boost demand. |
| Stereo Link | Coordinates channel decisions without requiring identical filters. |

The configured boost limit is a maximum authority, not a requested amount. Automatic mode may apply a stricter effective limit during search and finalization.

## 6. Frequency-dependent analysis

Adaptive Frequency-Domain Windowing (A-FDW) adjusts analysis bandwidth across frequency and confidence. Lower frequencies need enough resolution to describe room modes; higher frequencies need stronger protection from reflection-driven fine structure. The resulting bandwidth is analysis policy, not a separate output filter.

Confidence weighting reduces correction authority where measurements are inconsistent, cancellation-dominated, or otherwise weak. It complements smoothing and regularization rather than replacing their numerical roles.

## 7. Temporal Decay Control

Temporal Decay Control (TDC) addresses supported low-frequency energy decay separately from magnitude matching. It uses decay evidence to form bounded reduction, subject to strength, maximum-reduction, slope, and frequency policies.

TDC must not create boost or substitute a visually attractive decay estimate for reliable measurement evidence. When decay metadata is unavailable or fails quality checks, the workflow uses its documented conservative fallback and reports it.

## 8. Phase strategies

### Minimum Phase

Constructs a causal minimum-phase realization from the selected magnitude behavior. It normally offers the lowest latency and avoids linear-phase pre-energy.

### Linear Phase

Uses a symmetric linear-phase realization. It preserves linear phase at the cost of latency and potential energy before the main impulse, which is bounded by FIR and export guards.

### Mixed Phase

Blends minimum-phase behavior with bounded excess-phase correction, primarily in the configured low-frequency band. Correction strength fades between the full-correction and end frequencies.

### Asymmetric

Uses an asymmetric impulse layout to balance useful phase correction, latency, and pre-ringing containment. It is the general-purpose default in packaged releases.

### Phase guards

The phase path includes bounded correction strength, full/fade frequency limits, group-delay gradient containment, alignment checks, and pre-energy validation. Mixed- and asymmetric-phase processing also use mode-specific containment. Frequent last-resort guard activation indicates an upstream measurement or policy problem.

## 9. Hybrid IIR + FIR

Hybrid IIR targets supported narrow bass modes with CamillaDSP peaking filters while FIR retains broadband magnitude and phase responsibilities. The workflow:

1. identifies candidate modes from magnitude, confidence, and group-delay evidence
2. checks what remains after bounded FIR correction
3. transfers supported FIR cut into an IIR biquad and compensates that transfer in FIR
4. permits extra residual cut only when the remaining mode independently passes its guards
5. exports both stages in the CamillaDSP configuration

The combined cut remains bounded by configured cut authority and source-mode evidence. A Hybrid result is incomplete if only the FIR WAV is deployed.

## 10. Bass Integration

Bass Integration analyzes separately measured main and subwoofer impulse responses while preserving their relative timing. It searches crossover, delay, gain, polarity, and optional phase-model choices against left, right, and mono-centre validation scenarios.

With two subwoofers, the Direct DAC / CamillaDSP path combines their measured pressure responses into one reference and exports one shared mono Sub FIR. Reported delay, polarity, gain trim, and all-pass values apply to that shared branch. The result must be verified after deployment because real routing and crossover implementation affect the acoustic sum.

## 11. FIR synthesis and export windowing

After magnitude and phase construction, inverse transformation produces the FIR impulse. Gain staging reserves the requested margin and keeps stereo behavior consistent. Alignment places the main impulse according to the filter strategy and convolver requirements.

Export windowing is a later realization step:

- **auto** selects the normal bounded layout
- **rew_asym** applies REW-style asymmetric windowing
- legacy configuration values remain accepted for compatibility when supported

Windowing can change onset, symmetry, tail length, and practical latency. It does not redefine the target or rerun the correction policy. See [IR Export Windowing]({{ '/IR_Export_Windowing.html' | relative_url }}) for the focused explanation.

## 12. Output contract

A normal export can contain:

- Left and Right FIR WAV files in the selected output format
- bypass impulses aligned to the correction filter peak
- CamillaDSP configuration and filter snippets
- multi-rate variants when requested
- Sub FIR and crossover information when Bass Integration is active
- `Summary_...txt` with effective settings, guards, warnings, and Automatic-mode decisions

Multi-rate export changes sample rate and tap count by rate family while preserving the intended time span and correction policy. Each rate receives final validation.

## 13. Reproducibility and caches

Comparable runs require the same application version, measurement content, source type, target policy, sample-rate/tap context, correction limits, phase strategy, and relevant feature settings. Automatic search uses deterministic context and recorded parameters.

Cache entries are reusable only when their signatures cover every input that changes the meaning of the result. Version or compatibility changes invalidate stored work when required. The summary and logs expose enough context to distinguish a fresh computation from compatible reuse.

## 14. Related references

- [Why DecayCore Works]({{ '/Why_DecayCore_Works.html' | relative_url }}) — practical design rationale
- [Academic DSP Explanation]({{ '/Academic_DSP_Explanation.html' | relative_url }}) — mathematical model
- [DSP Guards Reference]({{ '/DecayCore_dsp_guards.html' | relative_url }}) — guard taxonomy and triggers
- [Stability and Reproducibility]({{ '/Stability_and_Reproducibility.html' | relative_url }}) — repeatable-run requirements
- [Modes]({{ '/Modes.html' | relative_url }}) — AUTO, Basic, and Advanced policy
