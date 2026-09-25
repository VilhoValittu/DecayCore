---
title: Adaptive Target
nav_title: Adaptive Target
description: How DecayCore selects a built-in base curve and adjusts it using room measurements.
permalink: /adaptive-target/
---

Choosing a target curve can easily become the whole evening. Adaptive Target scores the built-in curves to choose a base, then makes small changes using your room's measurements. It is one of three target strategies in AUTO mode.

## In brief

Adaptive Target makes small low-frequency changes to the selected built-in curve when both channels provide consistent evidence. Start with DecayCore's own measurement so the target has the session's RT60 decay data available. Use the built-in target search when measurement metadata is limited.

## What it is

In AUTO mode, DecayCore can determine the target curve in three ways:

- **Adaptive: derive target from room acoustics** (default) — scores built-in curves to choose a base, then adjusts it using broad, stereo-consistent bass evidence.
- **Auto: search best built-in** — evaluates multiple built-in target curves in parallel and picks the best-ranked match.
- **Use selected target curve from Target page** — uses the target curve manually selected in the Target tab and disables automatic target search.

Adaptive Target uses the built-in curves for base selection but skips the full candidate search across those targets.

## How it works

When adaptive target is selected, DecayCore:

1. Scores the built-in curves against the measurements and selects a base target.
2. Aligns each channel to the reference before measuring broad bass residuals, so the reference curve's own bass shelf is not mistaken for room buildup.
3. Smooths and evaluates the channels separately, then reduces adaptation when they disagree.
4. Bounds target changes to ±1.0 dB relative to that base and fades adaptation out by 500 Hz.
5. Allows additional bass lift only when reliable stereo RT60 bands are available, and suppresses that lift in a slow-decay room.
6. Preserves the selected base curve above 500 Hz by default and passes the target to Automatic mode's filter search.

RT60 never creates a tonal adjustment on its own. Optional high-frequency adaptation is disabled in AUTO and requires explicit high-SNR, stereo-consistent evidence at DSP-helper level.

## When to use adaptive target

Adaptive target is useful when:

- you want to skip the full multi-curve candidate search
- the room has unusual bass characteristics that may not match any single built-in target well
- you are using DecayCore's built-in measurement tool, which captures RT60 data automatically

## Limitations — RT60 requirement

**RT60 data is useful but not required.**

DecayCore's built-in measurement is the recommended source: it analyses the recorded decay and saves RT60 automatically. The session also includes harmonic curves for boost-risk guidance elsewhere in the correction workflow. Adaptive Target itself uses reliable stereo RT60 data to decide whether additional bass lift is permissible and to restrain it in a slow-decay room. RT60 alone does not determine the target's tonal shape.

Ordinary REW frequency-response text exports do not carry recorded decay. A suitable imported impulse-response WAV can supply RT60 through analysis of its decay, though it does not include the separate harmonic curves from a DecayCore session. If reliable RT60 remains unavailable, the target can still reduce an overly strong broad bass shelf, but it will not add bass lift.

**If you are using external measurements and RT60 data is not available, switching to the `Auto: search best built-in` strategy is generally the safer choice.** The built-in curve search evaluates how well different targets match the measured room and picks the best-ranked result regardless of RT60 data.

## Related pages

- [Getting Started]({{ '/getting-started/' | relative_url }})
- [Modes: AUTO, BASIC, ADVANCED]({{ '/Modes.html' | relative_url }})
- [Built-in Measurement]({{ '/measurement-workflow/' | relative_url }})
- [Technical Reference — Target formation]({{ '/Official_Manual.html' | relative_url }})
