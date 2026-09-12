---
title: Mixed Phase Room Correction in DecayCore
description: Learn how DecayCore applies bounded excess-phase correction and when Mixed Phase is appropriate.
permalink: /mixed-phase-room-correction/
---

## When should I use Mixed Phase?

Choose **Mixed Phase** when measurements show useful excess-phase behavior that you want to correct inside a limited frequency band. DecayCore blends that correction with a safer phase baseline and fades it out above the configured range.

Mixed Phase needs more judgment than Minimum Phase. Correction strength, full-correction frequency, fade frequency, group-delay limits, and pre-energy guards all affect the result. More phase correction is not automatically better, especially where reflections make measurements position-sensitive.

Start with the mode defaults and check the final impulse and group delay for obvious problems. Then listen. Use **Asymmetric** when you want the normal starting point without tuning mixed-phase controls yourself.

Read the [Technical Reference]({{ '/Official_Manual.html' | relative_url }}) for the phase pipeline or [Getting Started]({{ '/getting-started/' | relative_url }}) for the normal workflow.
