---
title: Mixed Phase Room Correction in DecayCore
description: Learn how DecayCore applies bounded excess-phase correction and when Mixed Phase is appropriate.
permalink: /mixed-phase-room-correction/
---

## When should I use Mixed Phase?

Choose **Mixed Phase** when measurements show useful excess-phase behavior that you want to correct inside a limited frequency band. DecayCore blends that correction with a safer phase baseline and fades it out above the configured range.

Mixed Phase needs more judgment than Minimum Phase. Correction strength, full-correction frequency, fade frequency, group-delay limits, and pre-energy guards all affect the result. More phase correction is not automatically better, especially where reflections make measurements position-sensitive.

Start with mode defaults, inspect the final impulse and group delay, and verify the deployed filter with a new measurement. Use **Asymmetric** instead when you want the recommended general-purpose balance without deliberately tuning mixed-phase controls.

Read the [Technical Reference]({{ '/Official_Manual.html' | relative_url }}) for the phase pipeline or [Getting Started]({{ '/getting-started/' | relative_url }}) for the normal workflow.
