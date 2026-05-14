---
title: DecayCore Measurement Workflow
nav_title: Measurement
description: Use DecayCore's built-in acoustic measurement workflow to generate FIR room correction filters.
permalink: /measurement-workflow/
---

DecayCore includes its own acoustic measurement workflow in release builds.

The recommended path is to measure directly with DecayCore, generate correction filters from those measurements, and export convolution-ready WAV FIR filters.

## Why use DecayCore's own measurement workflow?

DecayCore's measurement workflow is designed for its correction pipeline. It helps keep timing, phase, and filter generation behavior consistent from measurement to export.

This is especially important for FIR room correction because the correction process depends not only on frequency response, but also on timing, phase behavior, group delay, and impulse response handling.

## Typical workflow

1. Connect your measurement microphone and audio output.
2. Measure the left and right speakers.
3. Review the measurement data.
4. Generate correction filters.
5. Export WAV FIR filters.
6. Use the filters in CamillaDSP or another convolution engine.

## Optional external measurement workflows

DecayCore may also work with compatible external measurement data, including REW-style measurement exports. This is useful for users who already have existing measurements.

For new measurements, DecayCore's built-in measurement workflow is the preferred path.
