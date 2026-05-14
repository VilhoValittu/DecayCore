---
title: DecayCore FAQ
nav_title: FAQ
description: Frequently asked questions about DecayCore, acoustic measurement, CamillaDSP, FIR filters and room correction.
permalink: /faq/
---

## What is DecayCore?

DecayCore is a FIR room correction, acoustic measurement, and filter generation tool for CamillaDSP, convolution WAV filters, and other FIR-capable DSP engines.

## Was DecayCore formerly called CamillaFIR?

Yes. DecayCore was formerly known as CamillaFIR. The project was renamed to avoid confusion with CamillaDSP while keeping full CamillaDSP compatibility.

## Does DecayCore work only with CamillaDSP?

No. DecayCore is compatible with CamillaDSP, but the generated FIR filters can also be used with other convolution-capable DSP engines that support compatible WAV impulse response filters.

## Does DecayCore include measurement?

Yes. DecayCore includes its own built-in measurement workflow in release builds published in the Releases section.

## Should I measure with DecayCore or REW?

For new measurements, use DecayCore's own measurement workflow. It is designed for DecayCore's correction pipeline.

Existing REW-style measurement data may be used where compatible, but REW should be presented as an optional external workflow rather than the main DecayCore workflow.

## Where can I download DecayCore?

Download DecayCore from the official GitHub releases page:

[DecayCore releases](https://github.com/VilhoValittu/DecayCore/releases)

The built-in measurement feature is available in release builds published in the Releases section.

## Is DecayCore open source?

DecayCore source availability depends on the repository contents. Some components, such as measurement internals, may be excluded from the public source repository.
