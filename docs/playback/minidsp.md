---
title: miniDSP FIR and PEQ filters with DecayCore
description: Export DecayCore correction for miniDSP plugins such as the 2x4 HD and Flex, with phase in the FIR block and EQ in the PEQ bank.
permalink: /playback/minidsp/
---

miniDSP plugins don't load WAV files. They run a short FIR block next to a bank of PEQ filters, so DecayCore splits the correction between the two and writes files each block can import.

## How the correction is split

- **FIR block:** the main-channel FIR corrects phase and group delay only, so all of its taps go to timing.
- **PEQ bank:** the magnitude correction becomes peaking filters, cuts and boosts, fitted to the same safe correction curve the FIR would have used. Every boost limit still applies, and narrow room-mode cuts are part of the same filters.
- **Headroom:** the FIR carries the attenuation the largest boost needs, so the FIR and PEQ together don't clip.
- **High-pass:** goes to the plugin's crossover block instead of the FIR. The README in the export gives its type and frequency, and the export adds it as biquad files.
- **Subwoofer outputs** keep their EQ in the FIR.

## Taps per output

The four outputs of a 2x4 HD or Flex share 4,096 FIR taps. DecayCore shortens a longer tap setting for the run so every filter loads:

| FIR outputs | Taps per output |
|---|---|
| Left and right | 2,042 |
| With one subwoofer | 1,362 |
| With two subwoofers | 1,024 |

Your saved tap setting stays as it is. Output settings, Setup and START show the limit.

## Settings

1. Tick **miniDSP files (FIR .bin, PEQ .txt)** in Output settings before the run.
2. Set the sample rate to the fixed rate your plugin runs at, for example 96 kHz on the 2x4 HD.
3. Choose Mixed Phase, Asymmetric or Linear Phase. Minimum Phase leaves the FIR no phase correction to do.
4. Set **miniDSP PEQ filters / output** in Advanced → Hybrid IIR tuning if your plugin has more or fewer PEQ slots. The default is 10, to match the 2x4 HD.

While miniDSP is ticked, Hybrid IIR and the Equalizer APO, PipeWire, HLC, Roon and JRiver files are unavailable: they would play the FIR without its EQ. The CamillaDSP file carries the same PEQ.

## Load

1. Load each `.bin` from the `MiniDSP` folder into the FIR block of the matching output.
2. Load each `_PEQ.txt` into the PEQ bank of the same output with the plugin's advanced biquad import. Without the PEQ files the correction has no EQ at all.
3. Set the high-pass in the crossover block as the README describes.
4. With subwoofers, set the crossover, delay, gain and polarity in the plugin's own blocks, using the values in the CamillaDSP file.

## Check before listening

- Check that each output got both its FIR and its PEQ file.
- Start quietly and compare with the FIR and PEQ blocks bypassed at the same volume.

Back to [all players]({{ '/playback/' | relative_url }}).
