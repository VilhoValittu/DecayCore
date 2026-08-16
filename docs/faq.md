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

Yes. Packaged releases include guided measurement. See [Measurement]({{ '/measurement-workflow/' | relative_url }}) for current platform support and routing requirements.

## Should I measure with DecayCore or REW?

Use DecayCore's guided workflow when it is available and suits your routing. Existing REW text or impulse-response files are also valid when phase, timing, gain, and channel references are consistent.

## Where can I download DecayCore?

Download DecayCore from the official GitHub releases page:

[DecayCore releases](https://github.com/VilhoValittu/DecayCore/releases)

The packaged release also includes guided measurement and Automatic mode.

## DecayCore behaves strangely — what should I try first?

Open **About DecayCore → Maintenance**. Use **Clear automatic-mode caches** for unexpected Automatic results or **Reset settings to defaults** for a damaged configuration. Neither action removes measurements, target presets, or exported filters.

If the app does not start, use the platform cleanup script in `config_delete/`. [Configuration and Data Paths]({{ '/paths.html#resetting-caches-and-configuration-troubleshooting' | relative_url }}) lists the affected files.

## Is DecayCore open source?

DecayCore source availability depends on the repository contents. Some components, such as measurement internals, may be excluded from the public source repository.
