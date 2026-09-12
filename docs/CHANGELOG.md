---
title: DecayCore Changelog
description: Recent user-visible changes to DecayCore measurement, correction, export, and platform support.
hide_page_heading: true
---

# Changelog

This page contains the current development notes and three latest stable releases. [Older releases and Finnish translations are preserved in the archive]({{ '/changelog/archive/' | relative_url }}).

## [1.3.1] - 12-9-2026

### Filter optimization

Get more from long FIR filters with **ATC (Automate tap count)**. The new feature
automatically trims filter length at high tap counts while preserving correction
quality. Linear-phase filters benefit the most.

### Export

Filter filenames now include seconds in their timestamps, making consecutive
exports easier to tell apart.

## [1.3.0] - 10-9-2026

### Updates

DecayCore now checks for a newer stable version in the background when the UI
opens. When an update is available, a notification provides a link to the
release page. Updates can also be checked manually with the new **Check for
updates** button.

### Run and performance

The previous successful result and its download packages now remain available
until the new results view is ready.

Preprocessing cache hits are now 29–85% faster.

### UI

The palette button in the top bar now lets you choose between **Retro** and
**Modern** styles. The selection is saved in the settings and the page reloads
automatically. Both styles support dark and light modes.

The Modern style uses the system font, a restrained blue-gray color palette,
soft shadows, rounded surfaces and glass-like navigation. Retro remains the
default, and existing settings files continue to work without changes.

A new **Correction Confidence Map** under Graphs shows where DecayCore applies
correction and where it deliberately leaves the response unchanged.

## [1.2.9] - 8-9-2026

### Plots and preview

All plots now use DecayCore's own **Rust-powered plotting engine**. Plotly is no
longer included in the application packages.

**Target Preview** now keeps the main response plot focused on the target and
measurement curves. H2–H5 harmonics remain available in the dedicated harmonics
plot below.

### Bass Integration

The **−10 dB LFE compensation** setting is now optional, giving you control over
level compensation for your setup. It remains enabled by default, and both the
preview and cached results respect your selection.

Enabling Bass Integration now preserves your **Hybrid IIR** selection instead of
turning it on automatically.

### Automatic mode

A new **Run Cache refinement automatically** option lets you choose whether cache
refinement runs automatically. It is disabled by default.

### UI

**FIR Compare** and its download link are now available under **About DecayCore**
for easy access to the filter comparison tool.

## [1.2.8] - 4-9-2026

### Export

A user-defined **Filter name** field is now available in the export settings on
the Files tab. The name is included in ZIP, CamillaDSP, main, sub, config,
summary and IIR filenames, and is normalized for safe use in filenames.

Target-curve names and timestamps are now also included in sub and CamillaDSP
filenames.

### UI

The **Measurement preview** section is now hidden when no measurement result is
available and appears automatically after a successful measurement.

## [1.2.7] - 31-8-2026

### Bass Integration

Bug fix for LFE +10db level.

## [1.2.6] - 30-8-2026

### Phase correction

A new phase-correction algorithm now applies full correction up to 140 Hz and
tapers it smoothly above that point. Listening tests indicate slightly tighter
bass.

The phase plot now shows the FIR filter phase, and the group-delay plot shows
the FIR filter group delay.

### Automatic mode

The adaptive curve is now the default target in Automatic mode.

### Repeat measurements

Bass-phase consistency between repeat measurements is now checked from
20–300 Hz. Impulse responses from unstable repeats are no longer averaged in
the complex domain. A deterministic medoid measurement is selected as the phase
anchor while the averaged magnitude response is preserved.

The spatially averaged magnitude response is stored in the `__analysis.npz`
sidecar and restored when the WAV file is loaded again without altering its
phase.

Results and summary views now report the following diagnostics from 20–200 Hz:

- L/R phase-difference p90
- coherent-summation loss before FIR correction
- coherent-summation loss after FIR correction
- phase-reference confidence

An unreliable phase reference is no longer used to derive automatic L/R phase
correction.

### Measurement

WASAPI measurement reliability has been improved.

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

The adaptive curve is now saved to the filter-mode priors file, making Automatic
mode target selection more reproducible.

### UI

Texts have been simplified and the information box now moves with the UI.

### Export

Multi-rate taps now scale by sample-rate family: 44.1/48 kHz uses 65536 taps,
88.2/96 kHz uses 131072 taps, and 176.4/192 kHz uses 262144 taps.
