---
title: CamillaDSP FIR Room Correction with DecayCore
description: Use DecayCore to measure your system and generate CamillaDSP-compatible FIR room correction filters.
permalink: /camilladsp-fir-room-correction/
---

DecayCore can generate FIR room correction filters for CamillaDSP and other convolution-capable DSP engines.

DecayCore is a CamillaDSP FIR room correction tool that measures your system and exports convolution-ready WAV impulse response filters.

The recommended workflow is:

1. Measure the left and right speakers with DecayCore.
2. Generate correction filters from the measurements.
3. Export WAV FIR filters.
4. Load the generated filters and config file from filter package into CamillaDSP convolution.

DecayCore supports Linear Phase, Minimum Phase, Mixed Phase and Asymmetric FIR filters. It is designed for physically sane, band-limited correction instead of forcing the measured response to a perfectly flat line.

## Why use FIR correction with CamillaDSP?

CamillaDSP supports convolution filters, which makes it possible to apply precise FIR-based room correction. DecayCore creates these FIR filters with attention to frequency response, phase behavior, group delay, correction limits, and temporal decay.

## External measurement compatibility

DecayCore's own measurement workflow is the recommended path. Existing REW-style measurements may be used where compatible, but REW should not be presented as the primary workflow.

## Related pages

- [Measurement workflow](../measurement-workflow/)
- [FIR room correction](../fir-room-correction/)
- [Temporal Decay Control](../temporal-decay-control/)

---

[Home](../) | [GitHub](https://github.com/VilhoValittu/DecayCore) | [Releases](https://github.com/VilhoValittu/DecayCore/releases)
