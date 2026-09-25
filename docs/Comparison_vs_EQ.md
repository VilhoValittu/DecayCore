---
title: DecayCore vs Conventional EQ-Based Room Correction
description: Compare DecayCore's bounded FIR workflow with magnitude-only parametric EQ.
hide_page_heading: true
---

# DecayCore vs Conventional EQ-Based Room Correction

## In brief

This comparison uses a magnitude-only parametric EQ workflow as its reference: measure the frequency response, fit PEQ filters, and set the output level. Other EQ tools may offer some of the controls listed here.

## Magnitude-only PEQ workflow

PEQ filters change level at selected frequencies. A magnitude plot alone does not show how long bass energy decays or whether a narrow dip moves with microphone position. Those questions need additional measurement and listening checks before adding another filter.

## What DecayCore adds

| Aspect | Magnitude-only PEQ workflow | DecayCore |
|---|---|---|
| Propagation delay (TOF) | No separate phase-analysis step | Removed before excess-phase analysis |
| Phase strategy | Phase follows the selected PEQ filters | Linear / Minimum / Mixed / Asymmetric + optional 2058-safe mode |
| Excess-phase safety | No separate excess-phase correction | Mixed-phase fade + excess-delay and pre-ringing guards |
| Room modes / ringing | PEQ cuts change amplitude | **Temporal Decay Control (TDC)** with strength + max reduction + slope limit |
| Reflection handling | Depends on which peaks and dips are selected | Confidence-weighted correction + smoothing + A-FDW |
| Correction bounds | Set through the chosen PEQ filters | Explicit limits for boost/cut/slope/phase band and low-bass safety |
| Headroom handling | Set by the operator | Auto-headroom gain model with configurable margin |
| Stereo consistency | Left and Right settings chosen separately | Stereo-link options with a shared leveling anchor, shared or hybrid windowing, and channel-specific final auto-gain |
| Reproducible A/B | Requires a consistent comparison method | Optional **comparison mode** with fixed analysis grid |

## Listen to the result

Compare at matched playback level. Check whether bass notes decay more evenly and whether voices and transients still sound natural. A flatter magnitude plot alone cannot establish which filter sounds better.

For operating guidance, see [Getting Started]({{ '/getting-started/' | relative_url }}). For the engineering rationale, see [Why DecayCore Works]({{ '/Why_DecayCore_Works.html' | relative_url }}).
