---
title: DecayCore Measured Performance Case Study
nav_title: Performance
description: Inspect a room-specific DecayCore correction example with methodology, limitations, source REW data, response, group-delay, and waterfall plots.
permalink: /performance/
wide_content: true
image: https://vilhovalittu.github.io/DecayCore/performance/source/SPL_15-250.jpg
---

This page presents one inspectable measurement project. It is evidence of how the DecayCore workflow behaved in this setup, not a benchmark across rooms and not a promise of identical results elsewhere.

<div class="performance-report__actions">
  <a class="button button--primary" href="{{ '/performance/DecayCore_Performance.pdf' | relative_url }}">Open full report</a>
  <a class="button" href="{{ '/performance/DecayCore_Performance.pdf' | relative_url }}" download>Download PDF</a>
  <a class="button" href="{{ '/performance/source/1.0.9.1.mdat' | relative_url }}" download>Download REW source project</a>
</div>

## Methodology

- The figures come from the published REW project `1.0.9.1.mdat`.
- The plots identify an uncorrected right-channel measurement and corrected right-channel responses produced with Asymmetric, Linear, Minimum, and Mixed FIR modes.
- The response comparison is shown both from 15–250 Hz and from 15 Hz–24 kHz in the full report.
- The mode-specific report views include distortion, group delay, phase, step response, and waterfall plots.
- The original REW project, exported figures, and seven-page PDF are published with this page so the example can be inspected independently.

## Scope and limitations

This is a single room, loudspeaker setup, microphone position, and source measurement project. Room geometry, placement, measurement quality, target selection, hardware, and correction settings all affect the outcome. The plots therefore demonstrate one documented result rather than a general ranking of FIR modes.

The displayed traces and axes are REW exports. Some separate views use their own displayed ranges, so compare the trace shape and labeled axes rather than treating image dimensions or color ranges as normalized metrics. No aggregate score or multi-room claim is inferred from these figures.

## Low-frequency response

<figure class="report-figure">
  <a href="{{ '/performance/source/SPL_15-250.jpg' | relative_url }}">
    <img src="{{ '/performance/source/SPL_15-250.jpg' | relative_url }}" alt="REW response plot from 15 to 250 Hz comparing the uncorrected right channel with Asymmetric, Linear, Minimum, and Mixed FIR correction" width="1597" height="783" loading="lazy">
  </a>
  <figcaption>The red trace is the uncorrected right-channel measurement. The corrected traces show the four FIR modes recorded in the same published REW project.</figcaption>
</figure>

This view makes the comparison visible without reducing it to “flatter is always better.” DecayCore's correction policy also considers headroom, phase behavior, timing, confidence, and low-frequency safety.

## Asymmetric mode group delay

<figure class="report-figure">
  <a href="{{ '/performance/source/ASYM_GD.jpg' | relative_url }}">
    <img src="{{ '/performance/source/ASYM_GD.jpg' | relative_url }}" alt="REW group-delay plot from 15 to 250 Hz comparing the uncorrected right channel in red with the Asymmetric FIR result in green" width="1597" height="899" loading="lazy">
  </a>
  <figcaption>Group-delay comparison for the uncorrected measurement and the Asymmetric FIR result. The full report contains corresponding views for the other correction modes.</figcaption>
</figure>

Group delay is shown separately because magnitude response alone does not describe time-domain behavior. The result should be interpreted together with phase, step response, and the correction safeguards described in the <a href="{{ '/engineering/' | relative_url }}">engineering documentation</a>.

## Waterfall comparison

<div class="report-figure-pair">
  <figure class="report-figure">
    <a href="{{ '/performance/source/NO_EQ_WATERFALL.jpg' | relative_url }}">
      <img src="{{ '/performance/source/NO_EQ_WATERFALL.jpg' | relative_url }}" alt="REW waterfall plot for the uncorrected right-channel measurement from 15 to 250 Hz" width="1597" height="809" loading="lazy">
    </a>
    <figcaption>Uncorrected right-channel waterfall.</figcaption>
  </figure>
  <figure class="report-figure">
    <a href="{{ '/performance/source/ASYM_WATERFALL.jpg' | relative_url }}">
      <img src="{{ '/performance/source/ASYM_WATERFALL.jpg' | relative_url }}" alt="REW waterfall plot for the Asymmetric FIR corrected right channel from 15 to 250 Hz" width="1597" height="809" loading="lazy">
    </a>
    <figcaption>Asymmetric FIR right-channel waterfall.</figcaption>
  </figure>
</div>

These are separate REW views with labeled plot ranges. They are included to inspect how the measured decay shape changes in this case, not as a normalized before/after metric.

## Full evidence set

The [full PDF report]({{ '/performance/DecayCore_Performance.pdf' | relative_url }}) collects the response, distortion, group-delay, phase, step-response, clarity, and waterfall views. The [REW source project]({{ '/performance/source/1.0.9.1.mdat' | relative_url }}) is provided for readers who want to inspect the underlying saved project directly.

For the design intent behind bounded correction, see [Engineering DecayCore]({{ '/engineering/' | relative_url }}).
