---
title: Roon convolution with DecayCore
description: Load DecayCore FIR filters into Roon's DSP Engine with the ready-made convolution ZIP for every exported sample rate.
permalink: /playback/roon/
---

Roon takes a convolution ZIP that holds a filter for each sample rate and picks the one that matches playback. DecayCore builds that ZIP for you.

## Export

Tick **Roon convolution ZIP** in Output settings before the run. The export then holds a `Roon_Convolution_...zip` with your filters for every exported rate. Export every rate your library plays.

When you tick the Roon ZIP and no Equalizer APO, PipeWire or HLC file, the export leaves out the main FIR WAV files and the CamillaDSP files that load them. The Roon ZIP then holds the only copy of the filters.

## Load

1. Open the export and find `Roon_Convolution_...zip` inside it.
2. In Roon, open **DSP Engine → Convolution** for your zone.
3. Load the Roon ZIP as it is.

Don't load the export ZIP itself: it also holds the bypass filters, and Roon may pick those.

## Check before listening

- Filters with subwoofer outputs leave out the Roon ZIP. Use [CamillaDSP]({{ '/playback/camilladsp/' | relative_url }}) for those.
- Hybrid IIR can't be ticked while the Roon ZIP is, because Roon plays the FIR alone.
- Start quietly and compare with convolution off at the same volume.

Back to [all players]({{ '/playback/' | relative_url }}).
