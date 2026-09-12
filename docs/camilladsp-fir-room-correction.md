---
title: CamillaDSP FIR Room Correction with DecayCore
description: Generate and deploy DecayCore FIR filters and optional Hybrid IIR stages in CamillaDSP.
permalink: /camilladsp-fir-room-correction/
---

## How does DecayCore work with CamillaDSP?

DecayCore makes the filters, and CamillaDSP runs them while you listen. A normal export includes Left and Right FIR WAV files plus CamillaDSP configuration material. Load each filter into the matching channel, keep the exported sample-rate and gain settings, and leave enough headroom to avoid clipping. Getting Left and Right the right way round is still one of hi-fi's more cost-effective adjustments.

If Hybrid IIR is active, use both the exported peaking filters and the FIR convolution stage. Loading only the WAV omits correction transferred to IIR. Bass Integration exports a shared Sub FIR and the crossover, delay, gain, and polarity settings needed by the sub branch.

Start at a low volume and listen to familiar music. Compare with bypass at the same volume. The graphs can help find problems, but your ears make the final decision.

Use [Getting Started]({{ '/getting-started/' | relative_url }}) for filter creation, the [User Manual]({{ '/User_Manual.html' | relative_url }}) for deployment, and [Results]({{ '/results/' | relative_url }}) for optional measurement through CamillaDSP.
