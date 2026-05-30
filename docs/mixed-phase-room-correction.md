---
title: Mixed Phase Room Correction
description: DecayCore supports mixed phase FIR room correction with phase-aware correction, acoustic measurement workflow and practical safety limits.
permalink: /mixed-phase-room-correction/
---

DecayCore supports Mixed Phase FIR room correction.

Mixed phase correction can combine magnitude correction with controlled phase correction. The goal is not to blindly force a perfect phase response, but to apply correction where it is useful and safe.

DecayCore's preferred workflow is to measure directly with its built-in measurement feature, generate correction from those measurements, and export convolution-ready WAV FIR filters.

## Phase-aware correction

DecayCore uses practical correction limits to avoid excessive correction, pre-ringing, and unrealistic behavior.

Mixed phase correction is useful when the measured response shows behavior that can benefit from controlled phase correction without turning the filter into an aggressive or fragile correction.

## Related pages

- [Measurement workflow](../measurement-workflow/)
- [FIR room correction](../fir-room-correction/)
- [Minimum Phase FIR Generator](../minimum-phase-fir-generator/)
- [Temporal Decay Control](../temporal-decay-control/)
- [CamillaDSP FIR room correction](../camilladsp-fir-room-correction/)

---

[Home](../) · [GitHub](https://github.com/VilhoValittu/DecayCore) · [Releases](https://github.com/VilhoValittu/DecayCore/releases)
