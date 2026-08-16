---
title: DecayCore vs Conventional EQ-Based Room Correction
description: Compare DecayCore's bounded, phase-aware, time-domain FIR workflow with conventional magnitude-focused IIR and minimum-phase equalization.
hide_page_heading: true
---

# DecayCore vs Conventional EQ-Based Room Correction

## In brief

Conventional equalization usually concentrates on magnitude. DecayCore also controls correction confidence, phase strategy, timing, and low-frequency decay while keeping every stage bounded.

## Conventional approach (typical IIR / minimum-phase EQ)
Many room-correction workflows:
- treat response errors mainly as magnitude EQ problems
- focus on PEQ/IIR fitting with limited phase-domain control
- do not directly control decay behavior
- can overfit reflection-driven dips and combing

### Common consequences
- bass can get flatter but still ring
- narrow treble corrections can sound harsh
- results may change noticeably with small mic-position differences

---

## DecayCore approach (current)

| Aspect | Conventional EQ | DecayCore |
|---|---|---|
| Input sources | usually frequency-response only | REW TXT + WAV/IR inputs |
| Propagation delay (TOF) | often implicit | explicitly removed before phase analysis |
| Phase strategy | mostly minimum-phase behavior | Linear / Minimum / Mixed / Asymmetric + optional 2058-safe mode |
| Excess-phase safety | limited | Mixed-phase fade + excess-delay and pre-ringing guards |
| Room modes / ringing | mainly amplitude shaping | **Temporal Decay Control (TDC)** with strength + max reduction + slope limit |
| Reflection handling | may invert combing dips | confidence-weighted correction + smoothing + A-FDW |
| Correction bounds | tool-dependent | explicit limits for boost/cut/slope/phase band and low-bass safety |
| Headroom handling | manual gain staging | auto-headroom gain model with configurable margin |
| Stereo consistency | often per-channel behavior | stereo-link options with a shared leveling anchor, shared or hybrid windowing, and channel-specific final auto-gain |
| Reproducible A/B | harder across fs/taps | optional **comparison mode** with fixed analysis grid |
| Multi-rate output | uncommon | native multi-rate FIR export |
| Runtime diagnostics | often limited | Summary version stamp, timing breakdown, and System Health checks |

---

## Audible result (typical)
- tighter bass decay (less overhang)
- fewer "false-detail" treble corrections
- cleaner transients with safer phase behavior
- more repeatable tuning between runs

DecayCore intentionally avoids aggressive inversion and prioritizes corrections that remain stable and physically plausible.

For operating guidance, see [Getting Started]({{ '/getting-started/' | relative_url }}). For the engineering rationale, see [Why DecayCore Works]({{ '/Why_DecayCore_Works.html' | relative_url }}).
