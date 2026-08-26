---
title: DecayCore Changelog
description: Recent user-visible changes to DecayCore measurement, correction, export, and platform support.
hide_page_heading: true
---

# Changelog

This page contains the current development notes and three latest stable releases. [Older releases and Finnish translations are preserved in the archive]({{ '/changelog/archive/' | relative_url }}).

## DecayCore

## [1.2.5] - 26-8-2026

### UI

Maintenance actions are now available under **About DecayCore**. Automatic-mode
caches and settings can be reset separately without removing saved measurements,
target presets or exported filters.

### Linux audio

Linux packages now use the system audio libraries instead of bundling the Ubuntu
audio stack. Build and CI checks prevent incompatible audio libraries from being
included in release artifacts.

Opening the device list no longer opens or probes every ALSA device. PipeWire,
PulseAudio and the system default are prioritized over direct hardware devices.
Audio devices are stored using a stable host API and name identifier, resolved
again before opening a stream, and missing devices now produce a clear error.
Audio backend and device details are also included in the logs for diagnostics.

### Run

Starting a run without readable left and right measurements is now blocked with
a clear message in every mode. Measurement loading failures are shown in the
status line instead of leaving the run at "Reading...".

## [1.2.4] - 14-8-2026

### Measurement

Measurement warning texts have been clarified.

### Adaptive curve

The adaptive curve is now saved to the filter_mode_priors file. This enables more accurate results in automatic mode.

### UI

Texts have been simplified.
The info box now moves along with the UI.

### Export

Multi-rate taps now scale by sample-rate family:
44.1/48 kHz → 65536
88.2/96 kHz → 131072
176.4/192 kHz → 262144
When “Generate all common filter sample rates” is active:
target pre-fetch always uses 44.1 kHz / 65536 taps

## [1.2.3] - 11-8-2026

### Lower memory use after demanding runs

DecayCore now limits its largest DSP caches by both entry count and retained array size. Oversized results are not kept in memory, and runtime caches are cleared when a run ends — including interrupted or failed runs. This prevents high-sample-rate processing and repeated smoothing from leaving hundreds of megabytes reserved while the application is idle, without giving up useful caching during the run itself.

Statistics can now stay in array form inside the DSP pipeline and are converted only when needed for output. This avoids unnecessary intermediate copies during memory-intensive processing.

### More accurate and transparent dual-sub integration

When AUTO Bass Integration is enabled, the two subwoofer measurement inputs are now available directly on the Basic page. Bass Integration v5 combines separately measured subwoofers as their phase-preserving measured pressure sum, matching the shared mono sub branch used in the exported CamillaDSP configuration without inventing a virtual per-sub delay.

Candidate validation now also checks the mono-centre listening scenario alongside the left and right channels and bases safety decisions on the worst result. The results view and export summary show the combined bass response, the measured timing difference between the subwoofers, the combine method, routing and scaling assumptions, and clarify that both subwoofers use one shared mono FIR filter.

## [1.2.2] - 7-8-2026

### Automatic mode

Automatic mode is exclusive to packaged releases; source builds continue to offer the complete Basic and Advanced manual workflows.

### Take your filters anywhere with one click

The **Download filters (.zip)** button returns at the very bottom of the results page, letting you download all filters and configuration files directly from there — especially useful when DecayCore is running on another computer or device.
