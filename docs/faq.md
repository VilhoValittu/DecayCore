---
title: DecayCore FAQ
nav_title: FAQ
description: Frequently asked questions about DecayCore, acoustic measurement, CamillaDSP, FIR filters and room correction.
permalink: /faq/
---

## What is DecayCore?

DecayCore measures how your speakers and room behave together, then makes FIR correction filters for your playback system. It is for the familiar situation where you like your speakers, love your records, and suspect the room is getting rather too involved in the bass line.

## Was DecayCore formerly called CamillaFIR?

Yes. The name changed from CamillaFIR to DecayCore to avoid confusion with CamillaDSP. They still work together. There are already enough similar names to remember in a hi-fi rack.

## Does DecayCore work only with CamillaDSP?

No. CamillaDSP is one option. You can also load the generated filters into other convolution engines that accept compatible WAV impulse responses. Check the required format, sample rate, and channel routing for your player before loading them.

## Is DecayCore mainly a boost tool?

No. Most of the work comes from controlled cuts and modest shaping. Boost is deliberately limited. If the room cancels a note at your seat, asking the woofer to try harder can use up headroom without solving much. That is a good moment to look at placement, rather than reach for another 6 dB.

## Does DecayCore include measurement?

Yes. Packaged releases include guided measurement. See [Measurement]({{ '/measurement-workflow/' | relative_url }}) for current platform support and routing requirements.

## Should I measure with DecayCore or REW?

Use DecayCore's guided workflow if it suits your setup. If you already have good REW measurements, bring them along; there is no prize for measuring the same room twice before dinner. Text exports or impulse-response files work when phase, timing, gain, and channel references are consistent.

## Where can I download DecayCore?

Download DecayCore from the official GitHub releases page:

[DecayCore releases](https://github.com/VilhoValittu/DecayCore/releases)

The packaged release also includes guided measurement and Automatic mode.

## DecayCore behaves strangely — what should I try first?

Open **About DecayCore → Maintenance**. Use **Clear automatic-mode caches** for unexpected Automatic results or **Reset settings to defaults** for a damaged configuration. Neither action removes measurements, target presets, or exported filters.

If the app does not start, use the platform cleanup script in `config_delete/`. [Configuration and Data Paths]({{ '/paths.html#resetting-caches-and-configuration-troubleshooting' | relative_url }}) lists the affected files.

## Is DecayCore open source?

DecayCore source availability depends on the repository contents. Some components, such as measurement internals, may be excluded from the public source repository.
