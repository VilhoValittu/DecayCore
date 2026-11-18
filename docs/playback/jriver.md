---
title: JRiver Media Center convolution with DecayCore
description: Load DecayCore FIR filters into JRiver Media Center with a convolution folder that follows the playback rate.
permalink: /playback/jriver/
---

JRiver Media Center reads a convolution `.cfg` file and switches to the matching file when the playback rate changes. DecayCore exports one for every rate.

## Export

Tick **JRiver convolution folder** in Output settings before the run. The export then holds a `JRiver` folder with your filters and one `.cfg` per exported rate, named for example `DecayCore2.0_441.cfg`.

When you tick the JRiver folder and no Equalizer APO, PipeWire or HLC file, the export leaves out the main FIR WAV files and the CamillaDSP files that load them. The JRiver folder then holds the only copy of the filters.

## Load

1. Copy the `JRiver` folder from the export to a permanent place on the playback computer.
2. Open **Tools → Options → Audio → DSP Studio & Output Format → Convolution**.
3. Choose a `.cfg` from the folder. When the playback rate changes, JRiver switches to the file named for it.

JRiver keeps a loaded filter until it restarts, so restart it after you replace the files.

## Check before listening

- Filters with subwoofer outputs still need [CamillaDSP]({{ '/playback/camilladsp/' | relative_url }}).
- Hybrid IIR can't be ticked while the JRiver folder is, because JRiver plays the FIR alone.
- Start quietly and compare with convolution off at the same volume.

Back to [all players]({{ '/playback/' | relative_url }}).
