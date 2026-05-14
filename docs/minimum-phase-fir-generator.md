---
title: Minimum Phase FIR Filter Generator
description: DecayCore can generate minimum phase FIR correction filters for room correction, acoustic measurement workflows and convolution DSP engines.
permalink: /minimum-phase-fir-generator/
---

# Minimum Phase FIR Filter Generator

DecayCore supports Minimum Phase FIR filter generation for room correction workflows.

Minimum phase correction is useful when low latency is important and when the goal is to correct magnitude behavior while keeping timing behavior closer to a causal acoustic system.

DecayCore's preferred workflow is to measure directly with its built-in measurement feature, generate the correction from those measurements, and export convolution-ready WAV FIR filters.

## Minimum phase in DecayCore

DecayCore is designed to generate minimum phase correction while preserving the intended magnitude correction.

This matters because converting an existing linear phase FIR with inappropriate minimum phase methods can change the effective magnitude response.

## When to use minimum phase correction

Minimum phase correction is often useful when:

- low latency is important
- the correction should avoid linear-phase pre-ringing
- the goal is mainly magnitude correction
- the playback engine uses convolution WAV filters

## Related pages

- [Measurement workflow](../measurement-workflow/)
- [FIR room correction](../fir-room-correction/)
- [Mixed Phase Room Correction](../mixed-phase-room-correction/)
- [CamillaDSP FIR room correction](../camilladsp-fir-room-correction/)

---

[Home](../) · [GitHub](https://github.com/VilhoValittu/DecayCore) · [Releases](https://github.com/VilhoValittu/DecayCore/releases)
