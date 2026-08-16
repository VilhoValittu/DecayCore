---
title: DecayCore Measurement Workflow
nav_title: Measurement
description: Measure loudspeakers with DecayCore or import compatible REW text and impulse-response files.
permalink: /measurement-workflow/
---

Good correction starts with a clean, repeatable measurement. DecayCore can capture measurements in packaged releases or read compatible files created elsewhere.

## Platform support

| Platform | Main speaker measurement | Subwoofer measurement | Notes |
|---|---|---|---|
| Windows | Verified | Supported with multichannel output | Configure the playback device for 5.1 or 7.1 so the LFE channel is available. |
| Ubuntu Linux | Verified on Ubuntu 22.04 | Routing depends on the audio setup | Measurement audio requires the PortAudio system library. |
| macOS | Not verified on project test hardware | Not supported as a guaranteed workflow | Compatible external measurements can be imported. |

The measurement engine is included in packaged releases, not in the public source checkout.

## Guided measurement

1. Connect the measurement microphone and select the input and output devices.
2. Load the microphone calibration file.
3. Select Left and Right, the number of listening positions, and repeats per channel.
4. Set measurement volume at the end of the signal chain, such as the amplifier or an analog volume control. Avoid reducing level digitally before the amplifier because that also reduces measurement signal-to-noise ratio.
5. Run the guided session and follow the prompts when moving the microphone.
6. Review rejected takes. If many are rejected, check levels, connections, noise, and microphone placement before increasing rejection strictness.
7. Save the session. Keep all files from the session at the same gain and timing reference.

For subwoofer measurement on Windows, configure the output device for 5.1 or 7.1 in Windows Sound settings before starting. DecayCore sends the subwoofer sweep to the LFE channel.

## Importing existing measurements

### REW text export

Export each channel separately and include frequency, magnitude, and phase. DecayCore accepts normal REW header and comment lines. Use consistent timing references for left and right.

### Impulse-response WAV

Use mono files with consistent sample rate, gain, and timing. For REW exports, use `float32`, normalization, and the same `t=0` placement for every channel. Do not normalize files from a shared DecayCore measurement session individually.

## Measurement checklist

- Use the same microphone position and procedure for corresponding channels.
- Avoid clipping and background noise.
- Keep speakers, crossovers, routing, and volume unchanged during a measurement set.
- Save the uncorrected measurements for comparison.
- After deploying a filter, measure again with correction active.

Continue with [Getting Started]({{ '/getting-started/' | relative_url }}) or read the complete [User Manual]({{ '/User_Manual.html' | relative_url }}).
