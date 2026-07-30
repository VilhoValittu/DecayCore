---
title: Adaptive Target
nav_title: Adaptive Target
description: How DecayCore derives a room-aware target curve from measurement data instead of searching through built-in curves.
permalink: /adaptive-target/
---

Adaptive Target is one of three target strategies available in DecayCore's AUTO mode. Instead of searching through the library of built-in target curves, it synthesizes a custom target from the room's measured characteristics.

## What it is

In AUTO mode, DecayCore can determine the target curve in three ways:

- **Auto: search best built-in** — evaluates multiple built-in target curves in parallel and picks the best-ranked match. This is the default.
- **Adaptive: derive target from room acoustics** — synthesizes a conservative Harman6-based target from broad, stereo-consistent bass evidence. Does not search built-in curves.
- **Use selected target curve from Target page** — uses the target curve manually selected in the Target tab and disables automatic target search.

The adaptive strategy derives its target directly from the measurements. It does not iterate across multiple candidate curves, which makes it significantly faster than the default search-based approach.

## How it works

When adaptive target is selected, DecayCore:

1. Starts with a Harman6-style reference target as a base shape.
2. Aligns each channel to the reference before measuring broad bass residuals, so the reference curve's own bass shelf is not mistaken for room buildup.
3. Smooths and evaluates the channels separately, then reduces adaptation when they disagree.
4. Bounds target changes to −2.0/+0.75 dB and fades adaptation out by 500 Hz.
5. When reliable stereo RT60 bands are available, uses them only to prevent additional bass lift in a slow-decay room.
6. Preserves the Harman6 shape above 500 Hz by default and proceeds directly to filter generation.

RT60 never creates a tonal adjustment on its own. Optional high-frequency adaptation is disabled in AUTO and requires explicit high-SNR, stereo-consistent evidence at DSP-helper level.

## When to use adaptive target

Adaptive target is useful when:

- you want faster AUTO runs without the multi-curve search overhead
- the room has unusual bass characteristics that may not match any single built-in target well
- you are using DecayCore's built-in measurement tool, which captures RT60 data automatically

## Limitations — RT60 requirement

**RT60 data is useful but not required.**

RT60 data is captured automatically when you use DecayCore's built-in measurement tool. Adaptive target uses reliable stereo RT60 data as a bass-lift guard, not as a broadband target generator.

When using external measurements (REW text exports, WAV impulse files from REW, or other sources), RT60 data is typically not present. The target remains valid; only the decay-based bass-lift guard is omitted.

**If you are using external measurements and RT60 data is not available, the default `Auto: search best built-in` strategy is generally the safer choice.** The built-in curve search evaluates how well different targets match the measured room and picks the best-ranked result regardless of RT60 data.

## Related pages

- [Guide: Recommended AUTO workflow](../guide/)
- [Modes: AUTO, BASIC, ADVANCED]({{ '/Modes.html' | relative_url }})
- [Built-in Measurement](../measurement-workflow/)
- [Official Manual — Adaptive Target section]({{ '/Official_Manual.html' | relative_url }})

---

[Home](../) | [GitHub](https://github.com/VilhoValittu/DecayCore) | [Releases](https://github.com/VilhoValittu/DecayCore/releases)

### Disclaimer
AI was used to translate this document from Finnish to English.
