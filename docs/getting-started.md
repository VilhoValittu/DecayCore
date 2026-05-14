---
title: Getting Started with DecayCore
nav_title: Getting Started
description: Start using DecayCore for acoustic measurement, FIR room correction and CamillaDSP-compatible convolution filters.
permalink: /getting-started/
---

DecayCore is designed to measure your system, generate FIR room correction filters, and export convolution-ready WAV filters for CamillaDSP and other FIR-capable DSP engines.

## Recommended workflow

1. Download the latest release build.
2. Measure your speakers with DecayCore's built-in measurement workflow.
3. Review the measured response.
4. Choose a filter mode and target behavior.
5. Generate FIR correction filters.
6. Export WAV filters.
7. Load the filters into CamillaDSP or another convolution engine.

## Filter modes

DecayCore supports:

- Linear Phase
- Minimum Phase
- Mixed Phase
- Asymmetric FIR filters

Each mode has different tradeoffs in latency, phase behavior, pre-ringing risk, and correction behavior.

## External measurement compatibility

External measurement data, including REW-style data, may be used in compatible workflows. For new users, DecayCore's own measurement workflow is preferred because it is designed for the program's correction pipeline.
