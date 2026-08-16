---
title: FIR Room Correction with DecayCore
description: Understand when FIR room correction is useful and how DecayCore turns room measurements into convolution filters.
permalink: /fir-room-correction/
---

## What does FIR room correction change?

A finite impulse response (FIR) filter can change magnitude and phase while giving precise control over its time-domain behavior. DecayCore designs FIR filters from measured loudspeaker and room responses, then exports WAV impulses for a convolution engine.

The goal is not to make every point in the response flat. Deep dips are often caused by cancellations that move with listening position and cannot be repaired safely with boost. DecayCore instead emphasizes supported peaks, broad tonal errors, low-frequency decay, and bounded phase correction.

## Available filter strategies

- **Asymmetric** balances phase correction, latency, and pre-ringing containment.
- **Minimum Phase** prioritizes causal, lower-latency behavior.
- **Mixed Phase** adds bounded excess-phase correction in a selected band.
- **Linear Phase** prioritizes linear-phase behavior at the cost of latency.

## What you need

You need Left and Right measurements and a playback system that accepts convolution filters. The packaged application can guide the measurement and Automatic-mode search; compatible REW text or impulse-response files can also be imported.

Follow [Getting Started]({{ '/getting-started/' | relative_url }}) to create a first filter, or read [Why DecayCore Works]({{ '/Why_DecayCore_Works.html' | relative_url }}) for the design rationale.
