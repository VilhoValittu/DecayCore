---
title: HQPlayer convolution with DecayCore
description: Load DecayCore left and right FIR filters into HQPlayer's convolution engine.
permalink: /playback/hqplayer/
---

HQPlayer takes one filter per channel and scales it to the playback rate itself, so one export covers your whole library.

## Export

Export with the **Mono** layout at the highest sample rate you play. You get one WAV file for the left channel and one for the right. No extra Output settings box is needed.

## Load

1. Copy the left and right WAV files to a permanent place on the playback computer.
2. Open HQPlayer's convolution settings and load the left file for the left channel and the right file for the right.
3. Turn convolution on.

## Check before listening

- Check the channel assignment. Getting Left and Right the right way round is still one of hi-fi's more cost-effective adjustments.
- Hybrid IIR cuts are not part of the WAV files. Leave Hybrid IIR off for HQPlayer.
- Start quietly and compare with convolution off at the same volume.

Back to [all players]({{ '/playback/' | relative_url }}).
