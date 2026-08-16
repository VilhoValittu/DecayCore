---
title: Temporal Decay Control
description: Temporal Decay Control is a DecayCore feature for controlling low-frequency room behavior independently from simple amplitude flattening.
permalink: /temporal-decay-control/
---

Temporal Decay Control is one of DecayCore's core ideas.

## In brief

Temporal Decay Control (TDC) reduces supported low-frequency energy that lasts too long. It complements magnitude correction; it is not another target curve or a bass-boost tool.

## Why frequency response is not enough

A flat-looking response does not guarantee controlled bass. Low-frequency room modes can store energy and decay slowly, producing bass that sounds heavy or detached even after ordinary equalization.

## What Temporal Decay Control does

TDC uses decay evidence to form a low-frequency reduction separate from target matching. Strength, maximum reduction, slope, and frequency limits contain the result. It does not add boost or treat every response peak as a decay problem.

## When to use it

Use TDC when repeatable measurements show excessive low-frequency decay. Leave it conservative when decay evidence is missing, noisy, or changes strongly with microphone position. Speaker placement, listening position, crossover work, and acoustic treatment remain better solutions for problems that correction cannot address reliably.

## Related pages

- [Measurement workflow]({{ '/measurement-workflow/' | relative_url }})
- [FIR room correction]({{ '/fir-room-correction/' | relative_url }})
- [Mixed Phase Room Correction]({{ '/mixed-phase-room-correction/' | relative_url }})
- [CamillaDSP FIR room correction]({{ '/camilladsp-fir-room-correction/' | relative_url }})
