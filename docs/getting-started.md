---
title: Getting Started with DecayCore
nav_title: Start here
description: Download DecayCore, create your first FIR filters, and try them with familiar music.
permalink: /getting-started/
---

This is the quickest way to make and try your first pair of DecayCore filters.

## Your first DecayCore filter

1. Download and extract DecayCore.
2. Open DecayCore.
3. Measure Left and Right, or import compatible measurements.
4. Select **Automatic**, **Asymmetric**, and the **balanced** goal.
5. Press **START**.
6. Check the warnings and selected result.
7. Export and load the filters into your convolver.
8. Listen to familiar music. Keep the comparison level-matched. Do not keep a filter just because its graph looks better.

> **Need more detail?** Use the [Measurement guide]({{ '/measurement-workflow/' | relative_url }}), [Installation guide]({{ '/installation/' | relative_url }}), or the complete [User Manual]({{ '/User_Manual.html' | relative_url }}).

## Detailed steps

### Before you start

You need a calibrated measurement microphone, a way to play the measurement sweep, and a playback system that accepts FIR convolution filters. For most users, the packaged release is the right choice because it includes guided measurement and Automatic mode.

### 1. Download and open DecayCore

1. Download the package for your operating system from [GitHub Releases](https://github.com/VilhoValittu/DecayCore/releases/latest).
2. Extract the archive and start DecayCore.
3. If the browser does not open, go to `http://127.0.0.1:8080`.

See the [Installation guide]({{ '/installation/' | relative_url }}) for platform-specific steps or source installation.

### 2. Measure your speakers

The recommended method is the guided workflow on the **Measure** page:

1. Connect your calibrated microphone.
2. Load its calibration file.
3. Measure Left and Right separately.
4. Save the session and load the resulting impulse-response WAV files on the **Files** page.

You can instead import compatible REW text exports containing frequency, magnitude, and phase, or mono impulse-response WAV files. See [Measurement]({{ '/measurement-workflow/' | relative_url }}) for platform support, subwoofer routing, and export requirements.

### 3. Start with Automatic mode

For a first run in a packaged release:

- Mode: **DecayCore automatic mode (recommended)**
- Filter type: **Asymmetric**
- AUTO goal: **balanced**
- Target strategy: **Adaptive: derive target from room acoustics**
- Max boost: leave the default conservative limit

Automatic mode searches and ranks several guarded presets. Avoid trying to fill deep dips with extra boost; moving the speakers, changing the crossover, or treating the room is usually more effective.

Source checkouts do not include the packaged Automatic mode engine. Use **Basic** for a conservative manual starting point when running from source.

### 4. Generate and inspect

1. Open **START / Results** and press **START**.
2. Wait for the run to finish.
3. Review warnings, the selected solution, response plots, and the summary.
4. If a critical health check appears, correct its cause before exporting.

### 5. Export and load the filters

Download the result ZIP or open the output folder shown on the results page. It contains convolution-ready WAV filters and supporting configuration files.

- **CamillaDSP:** use the generated YAML or load the left and right WAV files into convolution filters.
- **Roon:** load the ZIP or compatible WAV set in Convolution.
- **Equalizer APO:** use the Convolution filter and leave enough preamp headroom.

### 6. Listen and optionally verify

Start at a low volume and use recordings you know well. Listen for tonal balance, bass integration, clarity, imaging, and anything that sounds unnatural.

Compare with bypass at the same volume. A louder result will often seem better.

The plots can help explain what you hear, but they are not the goal. If the filter sounds worse, go back and change it.

DecayCore does not include exported filters in its own measurement path. Measuring with correction active is possible only if you route the sweep through your convolver. If you are not sure how to do that, skip the verification measurement and trust the listening test.

## Next steps

- [User Manual]({{ '/User_Manual.html' | relative_url }}) — settings, outputs, deployment, and troubleshooting
- [Reading DecayCore Output]({{ '/DecayCore_Reading_Output_Guide.html' | relative_url }}) — result graphs and summary fields
- [How DecayCore works]({{ '/how-it-works/' | relative_url }}) — correction principles and technical references
