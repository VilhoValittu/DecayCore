---
title: FIR Room Correction with DecayCore
description: Understand when FIR room correction is useful and how DecayCore turns room measurements into convolution filters.
permalink: /fir-room-correction/
---

## What does FIR room correction change?

Perhaps a few bass notes dominate everything, or a familiar voice sounds heavier at your listening seat than it should. Measuring the speakers in the room gives you a way to investigate. DecayCore uses those measurements to design finite impulse response (FIR) filters, which can adjust magnitude and phase with precise control over their timing. You load the exported WAV impulses into your convolution engine for playback.

DecayCore focuses on peaks the measurements support correcting, broad tonal errors, low-frequency decay, and bounded phase correction. Deep dips often come from cancellations that move with listening position and cannot be repaired safely with boost. A ruler-flat plot is a tempting hobby in its own right, but the amplifier should not have to fund it.

## Available filter strategies

- **Asymmetric** balances phase correction, latency, and pre-ringing containment.
- **Minimum Phase** prioritizes causal, lower-latency behavior.
- **Mixed Phase** adds bounded excess-phase correction in a selected band.
- **Linear Phase** prioritizes linear-phase behavior at the cost of latency.

## What you need

You need Left and Right measurements and a playback system that accepts convolution filters. Use the packaged application's guided measurement where supported: its RT60 decay data and harmonic curves give the correction measured information about lingering bass and boost risk, alongside the speaker response. Automatic mode can then handle the search. Compatible REW text or impulse-response imports remain an option when needed.

Follow [Getting Started]({{ '/getting-started/' | relative_url }}) to create a first filter, or read [Why DecayCore Works]({{ '/Why_DecayCore_Works.html' | relative_url }}) for the design rationale.
