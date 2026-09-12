---
title: Results and verification
nav_title: Results
description: Read DecayCore's generated result, deploy the filters safely, and verify the complete playback chain with a new measurement.
permalink: /results/
---

A DecayCore run produces a technical prediction and export package. The result becomes trustworthy only after the filters have been loaded into the real playback chain and measured again.

## A result has two parts

<div class="doc-grid doc-grid--two">
  <section class="doc-card">
    <h3>Generated analysis</h3>
    <p>Response, phase, group delay, impulse, headroom, warnings, and the effective settings selected by the workflow.</p>
  </section>
  <section class="doc-card">
    <h3>Verification measurement</h3>
    <p>A new measurement with correction active confirms routing, gain, polarity, crossover behavior, sample rate, and the acoustic result.</p>
  </section>
</div>

## Read the generated result

<figure class="report-figure">
  <a href="{{ '/pics/ui_8.png' | relative_url }}">
    <img src="{{ '/pics/ui_8.png' | relative_url }}" alt="DecayCore START and Results page" width="1536" height="960" loading="lazy">
  </a>
  <figcaption>START / Results is the entry point for generation, progress, warnings, and the selected solution.</figcaption>
</figure>

Check the result in this order:

| Check | What you are looking for |
|---|---|
| **System Health** | Resolve every critical finding and understand the warnings. |
| **Magnitude** | Broad improvement without implausible boost into deep cancellations. |
| **Phase and Group Delay** | Smooth, contained timing correction inside the supported range. |
| **Filter and headroom** | Correction demand that stays inside the reported gain and safety limits. |
| **Impulse** | Controlled energy around the main peak, especially for linear- and mixed-phase filters. |
| **Summary** | The intended mode, target, filter type, limits, and Automatic winner details. |

[Open the complete Reading DecayCore Output guide]({{ '/DecayCore_Reading_Output_Guide.html' | relative_url }}).

## Verify the real playback chain

1. Load the correct Left and Right filters and any required subwoofer or Hybrid IIR stages.
2. Confirm the convolver sample rate and channel routing.
3. Start playback at reduced level.
4. Measure again from the same reference position and with the same routing.
5. Compare the corrected measurement with the uncorrected baseline.

The verification measurement should show that the broad response moved in the intended direction without new crossover cancellation, channel mismatch, clipping, or excessive bass loss. Listening remains the final check.

## Keep comparisons honest

One room example cannot prove universal performance. When publishing or comparing results, keep the microphone method, smoothing, level reference, frequency range, target, and playback chain visible. Share the measurement data when possible so others can inspect more than a screenshot.

For deployment details, use the [User Manual]({{ '/User_Manual.html#8-export-and-deploy' | relative_url }}). For the correction rationale, see [How DecayCore works]({{ '/how-it-works/' | relative_url }}).
