---
title: Temporal Decay Control
description: Temporal Decay Control is a DecayCore feature for controlling low-frequency room behavior independently from simple amplitude flattening.
permalink: /temporal-decay-control/
---

You know the bass note that is still hanging around when the next one arrives? That lingering energy is why DecayCore pays attention to decay as well as level. The room does not always share the drummer's sense of when to stop.

## In brief

Temporal Decay Control (TDC) reduces supported low-frequency energy that lasts too long. It complements magnitude correction; it is not another target curve or a bass-boost tool.

## Why frequency response is not enough

A frequency-response plot tells you about level across frequencies. It does not tell the whole story of how long the bass hangs around. Low-frequency room modes can store energy and decay slowly, leaving notes heavy or detached even after ordinary equalization. You may have the level about right and still struggle to follow a quick bass line.

## What Temporal Decay Control does

TDC uses decay evidence to form a low-frequency reduction separate from target matching. Strength, maximum reduction, slope, and frequency limits contain the result. It does not add boost or treat every response peak as a decay problem.

## When to use it

Use TDC when repeatable measurements show excessive low-frequency decay. Leave it conservative when decay evidence is missing, noisy, or changes strongly with microphone position. Speaker placement, listening position, crossover work, and acoustic treatment remain better solutions for problems that correction cannot address reliably.

## Related pages

- [Measurement workflow]({{ '/measurement-workflow/' | relative_url }})
- [FIR room correction]({{ '/fir-room-correction/' | relative_url }})
- [Mixed Phase Room Correction]({{ '/mixed-phase-room-correction/' | relative_url }})
- [CamillaDSP FIR room correction]({{ '/camilladsp-fir-room-correction/' | relative_url }})
