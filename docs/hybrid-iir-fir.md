---
title: Hybrid IIR + FIR Room Correction
nav_title: Hybrid IIR
description: How DecayCore combines narrow IIR biquad cuts with FIR correction for stubborn room modes in the bass region.
permalink: /hybrid-iir-fir/
---

The Hybrid IIR + FIR workflow is an optional extension to DecayCore's standard FIR correction pipeline. It adds a small set of narrow IIR peaking EQ cuts targeting stubborn room modes in the bass region before the FIR filter is synthesized.

## What it is

Standard FIR correction distributes correction energy across the entire filter length. For very narrow, high-Q room modes — common in small or acoustically untreated rooms — a long FIR filter may not be the most efficient tool. An IIR biquad peaking filter can resolve a narrow resonance precisely, with no minimum tap-count requirement.

The hybrid approach uses both:

- **IIR (Peaking EQ biquads):** handle the narrowest, most confident modal peaks in the bass.
- **FIR filter:** corrects the broadband magnitude and phase behavior after the IIR stage has already addressed the strongest peaks.

## How it works

When hybrid IIR is enabled, DecayCore:

1. Detects room modes in the configured bass frequency range using confidence and group-delay excess criteria.
2. Designs conservative Peaking EQ biquads targeting confirmed modal peaks (cuts only — no boosts).
3. Subtracts the IIR biquad magnitude response from the FIR target gain curve. The FIR then corrects only what the IIR has not already handled.
4. Exports the IIR biquad parameters in the CamillaDSP YAML configuration alongside the FIR convolver block.

This preconditioning approach keeps the IIR and FIR responsibilities separated: the IIR handles precision narrow-band cuts, and the FIR handles everything else.

## When to use

Consider hybrid IIR when:

- one or two narrow room modes in the bass remain clearly audible despite FIR correction
- measurements show high-confidence, high-Q peaks with strong group delay excess
- you are using CamillaDSP and can deploy an IIR filter stage in your pipeline

Be more conservative or leave it disabled when:

- measurements are noisy or uncertain
- the room bass does not show clear narrow modal peaks
- you are deploying to a system that cannot run IIR biquads alongside FIR convolution

## Output and CamillaDSP deployment

When hybrid IIR produces biquads, they are added to the exported CamillaDSP YAML configuration as Peaking EQ filter entries alongside the FIR convolver block.

**Important:** the IIR and FIR must both be active in your CamillaDSP pipeline for the hybrid correction to work as intended. If you load only the FIR WAV without the IIR biquads, the bass correction will be incomplete — the FIR was designed expecting the IIR to handle the modal peaks it skipped.

Verify your CamillaDSP pipeline includes both stages before finalizing the deployment.

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Enable or disable hybrid IIR preconditioning |
| `max_filters_per_channel` | `3` | Maximum number of IIR biquad cuts per channel |
| `min_freq_hz` | `20 Hz` | Lowest frequency considered for IIR modal detection |
| `max_freq_hz` | `150 Hz` | Highest frequency considered for IIR modal detection |
| `min_peak_db` | `4.0 dB` | Minimum peak height required to qualify for an IIR cut |
| `min_q` | `3.0` | Minimum allowed Q for designed biquads |
| `max_q` | `12.0` | Maximum allowed Q for designed biquads |
| `max_cut_db` | `6.0 dB` | Maximum allowed cut depth per biquad |
| `min_confidence` | `0.65` | Minimum confidence required at the mode frequency |
| `min_gd_excess_ms` | `10.0 ms` | Minimum group delay excess required for mode detection |
| `min_cut_priority` | `0.0` | Minimum cut priority score required to place a filter |

These parameters are available in the Advanced tab under the hybrid IIR tuning section.

## Related pages

- [FIR Room Correction](../fir-room-correction/)
- [CamillaDSP FIR Room Correction](../camilladsp-fir-room-correction/)
- [Official Manual — Hybrid IIR section](../Official_Manual/)
- [Temporal Decay Control](../temporal-decay-control/)

---

[Home](../) | [GitHub](https://github.com/VilhoValittu/DecayCore) | [Releases](https://github.com/VilhoValittu/DecayCore/releases)

### Disclaimer
AI was used to translate this document from Finnish to English.
