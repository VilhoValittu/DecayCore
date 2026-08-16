---
title: Getting Started with DecayCore
nav_title: Getting Started
description: Download DecayCore, measure or import your speakers, create your first FIR filters, and verify the result.
permalink: /getting-started/
---

This is the shortest path from downloading DecayCore to using a verified pair of correction filters.

## Before you start

You need a calibrated measurement microphone, a way to play the measurement sweep, and a playback system that accepts FIR convolution filters. For most users, the packaged release is the right choice because it includes guided measurement and Automatic mode.

## 1. Download and open DecayCore

1. Download the package for your operating system from [GitHub Releases](https://github.com/VilhoValittu/DecayCore/releases/latest).
2. Extract the archive and start DecayCore.
3. If the browser does not open, go to `http://127.0.0.1:8080`.

See the [Installation guide]({{ '/installation/' | relative_url }}) for platform-specific steps or source installation.

## 2. Measure your speakers

The recommended method is the guided workflow on the **Measure** page:

1. Connect your calibrated microphone.
2. Load its calibration file.
3. Measure Left and Right separately.
4. Save the session and load the resulting impulse-response WAV files on the **Files** page.

You can instead import compatible REW text exports containing frequency, magnitude, and phase, or mono impulse-response WAV files. See [Measurement]({{ '/measurement-workflow/' | relative_url }}) for platform support, subwoofer routing, and export requirements.

## 3. Start with Automatic mode

For a first run in a packaged release:

- Mode: **DecayCore automatic mode (recommended)**
- Filter type: **Asymmetric**
- AUTO goal: **balanced**
- Target strategy: **Auto: search best built-in**
- Max boost: leave the default conservative limit

Automatic mode searches and ranks several guarded presets. Avoid trying to fill deep dips with extra boost; moving the speakers, changing the crossover, or treating the room is usually more effective.

Source checkouts do not include the packaged Automatic mode engine. Use **Basic** for a conservative manual starting point when running from source.

## 4. Generate and inspect

1. Open **START / Results** and press **START**.
2. Wait for the run to finish.
3. Review warnings, the selected solution, response plots, and the summary.
4. If a critical health check appears, correct its cause before exporting.

## 5. Export and load the filters

Download the result ZIP or open the output folder shown on the results page. It contains convolution-ready WAV filters and supporting configuration files.

- **CamillaDSP:** use the generated YAML or load the left and right WAV files into convolution filters.
- **Roon:** load the ZIP or compatible WAV set in Convolution.
- **Equalizer APO:** use the Convolution filter and leave enough preamp headroom.

## 6. Measure again

Activate the filters at a reduced listening level and repeat the measurement. Confirm that:

- the broad response moved in the intended direction
- bass did not become weak or excessively boosted
- the playback chain does not clip
- left and right channel assignment is correct

Do not judge success from the generated graph alone. The verification measurement includes the real playback chain and room.

## Next steps

- [User Manual]({{ '/User_Manual.html' | relative_url }}) — settings, outputs, deployment, and troubleshooting
- [Reading DecayCore Output]({{ '/DecayCore_Reading_Output_Guide.html' | relative_url }}) — result graphs and summary fields
- [Engineering DecayCore]({{ '/engineering/' | relative_url }}) — correction principles and technical references
