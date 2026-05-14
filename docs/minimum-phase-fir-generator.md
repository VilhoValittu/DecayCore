---
title: Minimum Phase FIR Filter Generator
description: DecayCore can generate minimum phase FIR correction filters for room correction and convolution DSP workflows.
permalink: /minimum-phase-fir-generator/
---

# Minimum Phase FIR Filter Generator

DecayCore supports Minimum Phase FIR filter generation for room correction workflows.

Minimum phase correction is useful when low latency is important and when the goal is to correct magnitude behavior while keeping the timing behavior closer to a causal acoustic system.

## Minimum phase in DecayCore

DecayCore is designed to generate minimum phase correction in a way that preserves the intended magnitude correction.

This is important because converting an existing linear phase FIR with inappropriate minimum phase methods can change the effective magnitude response.

## Related pages

- [Measurement workflow](../measurement-workflow/)
- [FIR room correction](../fir-room-correction/)
- [Mixed phase room correction](../mixed-phase-room-correction/)
- [CamillaDSP FIR room correction](../camilladsp-fir-room-correction/)

---

[Home](../) | [GitHub](https://github.com/VilhoValittu/DecayCore) | [Releases](https://github.com/VilhoValittu/DecayCore/releases)
