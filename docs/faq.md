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

## Is DecayCore mainly a boost tool?

No. DecayCore is built around physically plausible room correction where controlled cuts and bounded shaping do most of the work. Boost is intentionally limited and should not be used to chase deep nulls, uncertain bass behavior, or a perfectly flat-looking graph.

## Does DecayCore include measurement?

Yes. DecayCore includes its own built-in measurement workflow in release builds published in the Releases section. Measurement has been verified to work on Windows. Linux has been verified to work at least on Ubuntu 22.04. macOS could not be tested due to unavailable test hardware.

## Should I measure with DecayCore or REW?

For new measurements, use DecayCore's own measurement workflow when available on your platform. It is designed for DecayCore's correction pipeline.

When DecayCore's measurement is not available on your platform, use compatible external measurements such as REW-style exports. For subwoofer measurement on Windows, ensure your playback device is configured for 5.1 or 7.1 multichannel in Windows Sound settings.

## Where can I download DecayCore?

Download DecayCore from the official GitHub releases page:

[DecayCore releases](https://github.com/VilhoValittu/DecayCore/releases)

The built-in measurement feature is available in release builds published in the Releases section.

## DecayCore behaves strangely — what should I try first?

Deleting DecayCore's automatic-mode disk caches and `config.json` is a
recommended first step. Stale Optuna journals, the auto-mode result cache, or a
corrupted config can cause unexpected automatic-mode results or startup issues.
These files are regenerated automatically on the next run, so removing them is
safe — your saved measurements and exported filters are not affected.

The repository includes ready-made cleanup scripts under `config_delete/` (one
per operating system) that remove exactly these files for you. See
[Configuration and Data Paths](paths.md#resetting-caches-and-configuration-troubleshooting)
for the full list of files and locations.

## Is DecayCore open source?

DecayCore source availability depends on the repository contents. Some components, such as measurement internals, may be excluded from the public source repository.
