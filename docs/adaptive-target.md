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
- **Adaptive: derive target from room acoustics** — synthesizes a Harman6-based target adjusted for the specific room's bass buildup, tilt behavior, and RT60 characteristics. Does not search built-in curves.
- **Use selected target curve from Target page** — uses the target curve manually selected in the Target tab and disables automatic target search.

The adaptive strategy derives its target directly from the measurements. It does not iterate across multiple candidate curves, which makes it significantly faster than the default search-based approach.

## How it works

When adaptive target is selected, DecayCore:

1. Starts with a Harman6-style reference target as a base shape.
2. Analyzes the measured room response to estimate the room's natural bass buildup.
3. Estimates the room's overall tilt behavior relative to the base target.
4. Adjusts the bass compensation and tilt compensation fractions based on these estimates.
5. When RT60 data is available, further refines the compensation using measured decay times across bass, mid, and treble frequency bands.
6. Returns the synthesized target and proceeds directly to filter generation — the multi-curve comparison phase is skipped entirely.

The RT60-based adjustment is bounded to prevent the compensation from pushing the target into acoustically unrealistic territory (hard limit: ±2 dB).

## When to use adaptive target

Adaptive target is useful when:

- you want faster AUTO runs without the multi-curve search overhead
- the room has unusual bass characteristics that may not match any single built-in target well
- you are using DecayCore's built-in measurement tool, which captures RT60 data automatically

## Limitations — RT60 requirement

**Adaptive target performs best when RT60 data is available from the measurement.**

RT60 data is captured automatically when you use DecayCore's built-in measurement tool. The tool records decay times per frequency band during the sweep session and stores them alongside the IR WAV files. Adaptive target uses this data to make the bass and tilt compensation room-specific.

When using external measurements (REW text exports, WAV impulse files from REW, or other sources), RT60 data is typically not present. In that case, adaptive target falls back to bass buildup and tilt estimation only — the RT60-based adjustment step is skipped. The result is still a valid target, but it will be less specifically adapted to the room's decay characteristics.

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
