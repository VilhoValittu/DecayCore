---
title: HQPlayer convolution with DecayCore
description: Load DecayCore left and right FIR filters into HQPlayer's convolution engine.
permalink: /playback/hqplayer/
---

HQPlayer takes one filter per channel and scales it to the playback rate itself, so one export covers your whole library.

## Export

Tick **HQPlayer convolution folder** in Output settings before the run. The export then holds an `HQPlayer` folder with the left and right filters of the highest exported rate, as `L_<rate>.wav` and `R_<rate>.wav` mono files, and a `README.txt` with these steps. Export at the highest rate you play. The folder is the same with the Mono and Stereo layouts.

## Load

1. Copy the `HQPlayer` folder to a permanent place on the playback computer.
2. In HQPlayer, open **Convolution → Engine setup…**.
3. Use **Browse…** to load the `L_` file for the left channel and the `R_` file for the right.
4. Tick **Enabled** and click **OK**.

With the matrix processor on, load the same files in the **Process** field of the left and right pipelines instead, and leave the convolution engine off.

## Check before listening

- Check the channel assignment. Getting Left and Right the right way round is still one of hi-fi's more cost-effective adjustments.
- Hybrid IIR can't be ticked while the HQPlayer folder is, because HQPlayer plays the FIR alone.
- Filters with subwoofer outputs leave out the HQPlayer folder. Use [CamillaDSP]({{ '/playback/camilladsp/' | relative_url }}) for those.
- Start quietly and compare with convolution off at the same volume.

Back to [all players]({{ '/playback/' | relative_url }}).
