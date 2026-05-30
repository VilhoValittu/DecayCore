---
title: Temporal Decay Control
description: Temporal Decay Control is a DecayCore feature for controlling low-frequency room behavior independently from simple amplitude flattening.
permalink: /temporal-decay-control/
---

Temporal Decay Control is one of DecayCore's core ideas.

Traditional room correction often focuses mainly on frequency response. DecayCore also considers low-frequency decay behavior, because bass problems are often temporal problems as much as amplitude problems.

## Why frequency response is not enough

A flat-looking frequency response does not automatically mean that bass sounds controlled.

Low-frequency room modes can store energy and decay slowly. If correction only looks at amplitude, it can miss problems that are heard as slow, heavy, or uncontrolled bass.

DecayCore's correction approach is designed to avoid blindly forcing the measured response into a flat line.

## Low-frequency decay and room behavior

In small rooms, bass behavior is strongly affected by room modes, speaker placement, listening position, and boundary interaction.

Temporal Decay Control helps DecayCore treat low-frequency correction as a time-domain and energy-control problem, not only as a static EQ curve problem.

## What Temporal Decay Control does

Temporal Decay Control is designed to reduce excessive low-frequency energy in a controlled way.

The goal is to improve bass clarity without applying unnecessarily aggressive EQ, unrealistic boosts, or narrow correction that only looks good at the microphone.

## Why DecayCore handles bass differently from simple EQ

Simple EQ can reduce or boost frequency bands, but it does not necessarily account for decay behavior, phase behavior, group delay, or the practical limits of correction.

DecayCore combines measurement-based correction, FIR filter generation, phase-aware behavior, and safety limits so that bass correction remains more physically sane.

## Recommended workflow

The recommended workflow is:

1. Measure the system with DecayCore's built-in measurement workflow.
2. Generate correction filters from the measurements.
3. Use Temporal Decay Control as part of the low-frequency correction behavior.
4. Export convolution-ready WAV FIR filters.
5. Load the filters into CamillaDSP or another convolution-capable DSP engine.

## Related pages

- [Measurement workflow](../measurement-workflow/)
- [FIR room correction](../fir-room-correction/)
- [Mixed Phase Room Correction](../mixed-phase-room-correction/)
- [CamillaDSP FIR room correction](../camilladsp-fir-room-correction/)

---

[Home](../) · [GitHub](https://github.com/VilhoValittu/DecayCore) · [Releases](https://github.com/VilhoValittu/DecayCore/releases)
