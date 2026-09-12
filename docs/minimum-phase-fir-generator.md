---
title: Minimum Phase FIR Filters in DecayCore
description: Learn when to choose DecayCore's minimum-phase FIR mode and what tradeoffs it makes.
permalink: /minimum-phase-fir-generator/
---

## When should I use Minimum Phase?

Choose **Minimum Phase** when causal behavior and lower latency matter more than linear- or excess-phase correction. The filter follows the selected magnitude correction using a minimum-phase realization and avoids the symmetric pre-energy of a linear-phase impulse.

Minimum Phase is useful when latency matters, when your playback chain calls for it, or when you want a conservative reference for comparing modes that do more phase correction. It is a perfectly respectable choice even if you have read enough forum threads to feel you ought to choose something more complicated. Magnitude, boost, cut, slope, bass, and numerical guards all remain active.

Compare the deployed result with **Asymmetric** using the same measurements, target, limits, and playback level. Choose by listening, not by whichever generated curve looks cleaner.

See [Getting Started]({{ '/getting-started/' | relative_url }}) for the first-run workflow and [FIR Room Correction]({{ '/fir-room-correction/' | relative_url }}) for the mode overview.
