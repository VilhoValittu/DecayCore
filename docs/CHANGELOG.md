---
title: DecayCore Changelog
description: Recent user-visible changes to DecayCore measurement, correction, export, and platform support.
hide_page_heading: true
---

# Changelog

This page contains the current development notes and three latest stable releases. [Older releases and Finnish translations are preserved in the archive]({{ '/changelog/archive/' | relative_url }}).

## [Unreleased]

## [1.3.5] - 19-9-2026

### A small bug, a stronger safety net

This release fixes a small, four-line stereo validation bug with a much longer
list of safeguards around it. The correction engine has not been broadly
changed: the extra work makes the existing safety policy more accurate,
transparent and dependable.

Final FIR validation now evaluates the least favourable left or right channel
for pre-ringing energy and group delay, while reporting both channels
separately. Pre-ringing is measured from the generated FIR itself, so natural
room behaviour no longer causes false rejections. Linear-phase centring delay
is removed before group-delay analysis, eliminating spurious results of around
1.36 seconds, and unreliable spectral-null points are excluded.

The surrounding failure paths are now deliberately conservative. Missing FIR
validation is rejected, new settings default to rejection while explicitly
saved warning policies remain compatible, and an all-rejected fallback is
clearly identified as a hard-gate failure in the report. If the excess-delay
guard cannot be calculated, excess-phase correction is disabled and the
diagnostic is retained instead of silently bypassing the safeguard.

Automatic mode's cache version has been advanced so results created under the
older validation policy cannot be reused. In short: a tiny root cause, a more
trustworthy final check, and clearer evidence when DecayCore chooses the safe
fallback.

## [1.3.4] - 19-9-2026

### Automatic mode

Adaptive-target selection has been redesigned. Automatic mode now evaluates all
built-in targets first and chooses the base curve that best fits the measurement.
The adaptive curve is built from that selection instead of always using Harman6.
Adaptation remains strictly bounded to −1…+1 dB relative to the selected base
target.

### Linux

Fixed automatic browser opening on recent KDE and Arch Linux systems. Packaged
builds now restore the system library path before starting `xdg-open`, preventing
KDE from loading DecayCore's bundled `libstdc++.so.6` and rejecting it because
required GLIBCXX or CXXABI versions are unavailable.

This also ensures that the browser receives the current tokenized DecayCore URL
instead of leaving the user with an old or unauthenticated page that reports
HTTP 403 errors.

## [1.3.3] - 18-9-2026

### UI

The **Modern** style has a new layout. The packaged application now uses
DecayCore's native browser UI engine and no longer includes NiceGUI.

The new **Apply saved auto settings** button in Basic and Advanced modes lets
you apply saved Automatic mode settings for the selected filter type, with the
newest settings listed first. The current target curve and mode safety limits
are preserved.

### Automatic mode

After running Automatic mode with an adaptive target, you can download that
target and use it in Basic and Advanced modes.
