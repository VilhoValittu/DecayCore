---
title: FIR Room Correction
description: Learn how DecayCore uses FIR filters for measurement-based room correction, phase-aware correction and convolution DSP workflows.
permalink: /fir-room-correction/
---

FIR room correction uses finite impulse response filters to correct measured loudspeaker and room behavior. FIR filters can control magnitude response and, depending on the design, phase behavior and timing behavior.

DecayCore measures the system, generates FIR filters, and exports convolution-ready WAV filters for CamillaDSP and other FIR-capable engines.

DecayCore focuses on physically sane, band-limited correction. This means the correction should avoid unrealistic boosts, excessive narrowband edits, and unnecessary high-frequency overcorrection.

## Supported FIR modes

DecayCore supports:

- Linear Phase
- Minimum Phase
- Mixed Phase
- Asymmetric FIR filters

Each mode has different tradeoffs in latency, phase correction, pre-ringing risk, and correction behavior.

## Why not just use parametric EQ?

Parametric EQ (PEQ) is a common and useful tool, but it has limits in room correction.

PEQ applies static gain and filter curves in the frequency domain. It does not control phase behavior or decay behavior. A PEQ boost or cut applied to a room mode may reduce the measured amplitude at a single point, but does not necessarily change how long that mode stores energy.

FIR filters can correct both magnitude and phase simultaneously, and can apply correction with awareness of the measured impulse response. This allows DecayCore to:

- correct magnitude and phase together where beneficial
- avoid narrow corrections that look good in REW but do not translate to better sound
- apply Temporal Decay Control for low-frequency energy behavior, not just amplitude shaping
- use conservative limits to avoid boosting deep nulls or narrow problem areas

PEQ remains useful for simple tonal adjustments or post-correction trim. It is not a substitute for FIR-based room correction when decay behavior, phase response, or cross-band interactions are the actual problem.

## Related pages

- [Measurement workflow](../measurement-workflow/)
- [Minimum phase FIR filter generation](../minimum-phase-fir-generator/)
- [Mixed phase room correction](../mixed-phase-room-correction/)
- [Temporal Decay Control](../temporal-decay-control/)

---

[Home](../) | [GitHub](https://github.com/VilhoValittu/DecayCore) | [Releases](https://github.com/VilhoValittu/DecayCore/releases)
