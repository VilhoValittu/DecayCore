---
title: FIR Room Correction
description: Learn how DecayCore uses FIR filters for measurement-based room correction, phase-aware correction and convolution DSP workflows.
permalink: /fir-room-correction/
---

# FIR Room Correction

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

## Related pages

- [Measurement workflow](../measurement-workflow/)
- [Minimum phase FIR filter generation](../minimum-phase-fir-generator/)
- [Mixed phase room correction](../mixed-phase-room-correction/)
- [Temporal Decay Control](../temporal-decay-control/)

---

[Home](../) | [GitHub](https://github.com/VilhoValittu/DecayCore) | [Releases](https://github.com/VilhoValittu/DecayCore/releases)
