---
title: DecayCore - FIR Room Correction and Measurement Tool
description: DecayCore is a free FIR room correction and acoustic measurement tool for CamillaDSP, convolution WAV filters, phase-aware correction and temporal decay control.
permalink: /
---

DecayCore is a free FIR room correction, acoustic measurement, and filter generation tool for CamillaDSP, convolution WAV filters, Roon convolution workflows, Equalizer APO, and other FIR-capable DSP engines.

DecayCore includes its own built-in measurement workflow in release builds. The preferred workflow is to measure directly with DecayCore, generate correction filters from those measurements, and export convolution-ready WAV FIR filters.

DecayCore focuses on physically sane, band-limited room correction, phase-aware correction, automatic target optimization, and Temporal Decay Control for low-frequency room behavior.

DecayCore was formerly known as CamillaFIR. The project was renamed to avoid confusion with CamillaDSP while keeping full compatibility with CamillaDSP.

## Main features

- Built-in acoustic measurement workflow in release builds
- FIR room correction from acoustic measurements
- CamillaDSP-compatible WAV filter export
- Linear Phase, Minimum Phase, Mixed Phase and Asymmetric FIR filters
- Automatic target optimization
- Phase-aware correction
- Temporal Decay Control
- Convolution DSP compatibility

## Documentation

- [Getting started](getting-started/)
- [Measurement workflow](measurement-workflow/)
- [CamillaDSP FIR room correction](camilladsp-fir-room-correction/)
- [FIR room correction](fir-room-correction/)
- [Minimum phase FIR filter generation](minimum-phase-fir-generator/)
- [Mixed phase room correction](mixed-phase-room-correction/)
- [Temporal Decay Control](temporal-decay-control/)
- [FAQ](faq/)

## Download

Download DecayCore from the official GitHub releases page:

[DecayCore releases](https://github.com/VilhoValittu/DecayCore/releases)

The built-in measurement feature is available in release builds published in the Releases section.

## Optional compatibility

DecayCore can also work with compatible external measurement workflows, including REW-style measurement data where supported. However, DecayCore's own measurement workflow is the recommended path for new users.
