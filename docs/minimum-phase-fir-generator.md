---
title: Minimum Phase FIR Filters in DecayCore
description: Learn when to choose DecayCore's minimum-phase FIR mode and what tradeoffs it makes.
permalink: /minimum-phase-fir-generator/
---

## When should I use Minimum Phase?

Choose **Minimum Phase** when causal behavior and lower latency matter more than linear- or excess-phase correction. The filter follows the selected magnitude correction using a minimum-phase realization and avoids the symmetric pre-energy of a linear-phase impulse.

Minimum Phase is useful for latency-sensitive playback, compatibility-focused convolution chains, and a conservative comparison against more phase-active modes. It does not make the correction unrestricted: magnitude, boost, cut, slope, bass, and numerical guards remain active.

Compare the deployed result with **Asymmetric**, the recommended general-purpose starting point, using the same measurements, target, limits, and gain. Re-measure each result instead of comparing generated curves alone.

See [Getting Started]({{ '/getting-started/' | relative_url }}) for the first-run workflow and [FIR Room Correction]({{ '/fir-room-correction/' | relative_url }}) for the mode overview.
