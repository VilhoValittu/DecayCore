---
title: Interactive FIR comparison
description: Compare DecayCore phase strategies against the same room measurement, target, and reference filter.
permalink: /fir-comparison-demo/
wide_content: true
---

Four correction strategies can reach for a similar magnitude response while behaving very differently in time. This example puts DecayCore's Mixed, Minimum, Linear, and Asymmetric phase filters beside the same reference filter, using one room measurement and one adaptive target throughout.

<figure class="report-figure demo-preview">
  <a href="{{ '/demos/fir-comparison/' | relative_url }}">
    <img src="{{ '/pics/fir-comparison-demo.png' | relative_url }}" alt="Interactive FIR comparison showing five filters in a side-by-side metric table" width="1440" height="900">
  </a>
  <figcaption>The reference is descriptive, not a declared winner. Change the reference inside the demo to inspect each difference from another point of view.</figcaption>
</figure>

<div class="action-row">
  <a class="button button--primary" href="{{ '/demos/fir-comparison/' | relative_url }}">Explore the interactive comparison</a>
  <a class="button" href="https://github.com/VilhoValittu/FIR-Compare/releases">Download FIR Compare</a>
</div>

## What you can inspect

The report keeps the comparisons on the same physical frequency grid and exposes the evaluated bands behind each number. You can:

- compare target error, gain, latency, settling, and impulse-energy metrics side by side;
- switch the reference filter and view signed magnitude and group-delay differences;
- inspect relative or raw FIR magnitude and phase with or without bulk delay;
- adjust plot ranges, lock vertical axes, and read values at the nearest sample.

These plots describe one measured example. A smaller target error or a tidier-looking impulse does not by itself establish better audible performance. Check headroom and safety first, then compare filters at matched playback level with familiar recordings.

For the practical checks around a generated filter, continue with [Read the result, then listen]({{ '/results/' | relative_url }}).
