---
title: CamillaDSP FIR Room Correction with DecayCore
description: Generate and deploy DecayCore FIR filters and optional Hybrid IIR stages in CamillaDSP.
permalink: /camilladsp-fir-room-correction/
---

## How does DecayCore work with CamillaDSP?

DecayCore designs the correction; CamillaDSP applies it during playback. A normal export includes Left and Right FIR WAV files plus CamillaDSP configuration material. Load each filter into the matching channel, preserve the exported sample-rate and gain assumptions, and leave enough headroom to avoid clipping.

If Hybrid IIR is active, use both the exported peaking filters and the FIR convolution stage. Loading only the WAV omits correction transferred to IIR. Bass Integration exports a shared Sub FIR and the crossover, delay, gain, and polarity settings needed by the sub branch.

Always start at reduced volume and measure again after deployment. The verification measurement catches routing, rate, crossover, polarity, and gain differences that an offline prediction cannot see.

Use [Getting Started]({{ '/getting-started/' | relative_url }}) for filter creation and the [User Manual]({{ '/User_Manual.html' | relative_url }}) for deployment details.
