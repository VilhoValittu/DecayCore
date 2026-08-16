---
title: DecayCore Glossary
nav_title: Glossary
description: Plain-language definitions for FIR, impulse response, phase, timing, decay, target, and correction terms used by DecayCore.
permalink: /glossary/
---

These definitions describe how terms are used in DecayCore. They are intentionally practical rather than exhaustive.

## A-FDW

**Adaptive Frequency-Domain Windowing** changes analysis resolution by frequency and measurement confidence. It helps DecayCore follow broad, repeatable behavior instead of narrow reflection detail.

## Convolution

The process a playback engine uses to apply an FIR impulse response to audio. CamillaDSP, Roon, and Equalizer APO can perform convolution.

## Excess phase

Phase behavior left after ordinary propagation delay and the selected phase baseline are removed. DecayCore corrects it only within explicit strength and frequency limits.

## FIR

A **finite impulse response** filter. It is stored as a sequence of samples, commonly in a WAV file, and can control magnitude and phase.

## Group delay

Frequency-dependent delay. A group-delay plot helps reveal timing changes, crossover behavior, and low-frequency energy storage.

## Headroom

Unused digital level reserved to prevent clipping when correction or target gain raises the signal.

## HPF

A **high-pass filter** reduces content below its cutoff. It can protect speakers and prevent correction from demanding unsupported deep bass.

## IR

An **impulse response** records how a system reacts over time. It contains timing, phase, and magnitude information used to design and verify correction.

## Magnitude

Signal level at each frequency, normally displayed in decibels. A frequency-response graph is primarily a magnitude view.

## Phase

The timing relationship of a repeating signal at each frequency. Phase correction can improve supported timing behavior, but reflection-driven phase detail must not be followed blindly.

## RT60

An estimate of how long sound energy takes to decay by 60 dB. DecayCore uses reliable RT60 data as evidence and a safety constraint, not as an automatic reason to add bass.

## Target curve

The frequency-response shape that guides correction. A target is a listening preference and system goal, not proof that every measured deviation can be corrected safely.

## Taps

The number of samples in an FIR filter. More taps increase its time span and low-frequency resolution, but also increase processing cost and often latency.

## TDC

**Temporal Decay Control** reduces supported low-frequency energy that lasts too long. It complements magnitude correction and does not create bass boost.

## Time of flight (TOF)

The propagation delay from loudspeaker to microphone. DecayCore removes this linear delay before analyzing excess phase so distance is not mistaken for a phase error.

## Verification measurement

A new measurement made with the exported filters active. It checks the real playback chain, routing, gain, crossover, loudspeakers, and room rather than only the offline prediction.
