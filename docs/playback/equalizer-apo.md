---
title: Equalizer APO, PipeWire and HLC with DecayCore
description: Load DecayCore FIR filters into Equalizer APO on Windows, a PipeWire filter-chain on Linux, or the HLC convolver with ready-made configs.
permalink: /playback/equalizer-apo/
---

These convolvers play the FIR filters from a short config file. DecayCore writes the config next to the filters, so you only fill in where the files live. Each box in Output settings is separate; tick only the ones you use.

## Equalizer APO

Tick **Equalizer APO file** in Output settings before the run. The export then holds an Equalizer APO config (`.txt`) with a Convolution line for each channel.

1. Copy the config lines into a config file, for example `config.txt`, in the Equalizer APO config folder.
2. Put the WAV files next to it, or replace the file names in the lines with full paths.
3. Check that Left and Right go to the right channels.
4. Add enough preamp reduction to prevent clipping.

## PipeWire

Tick **PipeWire file** in Output settings before the run. The export then holds a PipeWire filter-chain config (`.conf`) that adds a stereo output called **DecayCore room correction**.

1. Save it as `~/.config/pipewire/pipewire.conf.d/decaycore.conf`.
2. Replace `/path/to` in it with the folder that holds the WAV files.
3. Restart PipeWire.
4. Choose **DecayCore room correction** as the output.

## HLC convolver

Tick **HLC convolver file** in Output settings before the run. The export then holds `Config_...cfg` for the correction and `bypass/Bypass_Config_...cfg` for a level- and timing-matched comparison. Load the config in HLC and keep the bypass config at hand for switching.

## Check before listening

- The Equalizer APO and PipeWire configs name the sample rate they were designed for at the top. Match it to the playback rate.
- Hybrid IIR can't be ticked while any of these boxes is, because these convolvers play the FIR alone.
- Start quietly and compare with bypass at the same volume.

Back to [all players]({{ '/playback/' | relative_url }}).
