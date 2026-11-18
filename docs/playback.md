---
title: Load your filters into a player
nav_title: Playback
description: Step-by-step guides for loading DecayCore filters into CamillaDSP, Roon, JRiver, HQPlayer, Equalizer APO, PipeWire, HLC and miniDSP.
permalink: /playback/
---

DecayCore makes the filters; your player runs them. Tick the files for your player in **Output settings** before you press **START**, then follow its page below. Each page covers what the export holds, how to load it, and what to check before the first record.

<div class="doc-grid">
  <section class="doc-card">
    <h3><a href="{{ '/playback/camilladsp/' | relative_url }}">CamillaDSP</a></h3>
    <p>Ready-made configuration and WAV filters, including Hybrid IIR and subwoofer outputs.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/playback/roon/' | relative_url }}">Roon</a></h3>
    <p>One convolution ZIP with a filter for every exported sample rate.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/playback/jriver/' | relative_url }}">JRiver Media Center</a></h3>
    <p>A convolution folder that switches filters with the playback rate.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/playback/hqplayer/' | relative_url }}">HQPlayer</a></h3>
    <p>One filter per channel, scaled to the playback rate by HQPlayer.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/playback/equalizer-apo/' | relative_url }}">Equalizer APO, PipeWire and HLC</a></h3>
    <p>Ready-made convolver configs for Windows, Linux and the HLC convolver.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/playback/minidsp/' | relative_url }}">miniDSP</a> <span class="doc-card__tag">New</span></h3>
    <p>FIR .bin and PEQ .txt files sized for the 2x4 HD and Flex FIR block.</p>
  </section>
</div>

## Before any player

- **Match the sample rate.** A filter is designed for the rate you export. Export every rate your player switches between, or the one rate your DSP runs at.
- **Start quietly.** Load the filters with the volume down, then bring it up on music you know well.
- **Compare with bypass at the same volume.** The export holds bypass filters that match the level and timing of the correction, so the comparison is fair.
- **Hybrid IIR needs CamillaDSP.** Only the CamillaDSP configuration carries the Hybrid IIR cuts. Equalizer APO, PipeWire, HLC, Roon and JRiver play the FIR alone, so Hybrid IIR can't be ticked while their files are. The miniDSP files fit their own PEQ filters instead.
- **Subwoofers need CamillaDSP or miniDSP.** Filters with subwoofer outputs leave out the Roon ZIP. CamillaDSP gets the subwoofer FIR and crossover settings; on a miniDSP you set the subwoofer blocks by hand.

The full list of exported files is in the [User Manual]({{ '/manual/#8-export-and-deploy' | relative_url }}). [Results]({{ '/results/' | relative_url }}) explains how to judge the filters once they play.
