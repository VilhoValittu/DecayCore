---
title: Engineering DecayCore
nav_title: Engineering
description: Explore DecayCore's DSP rationale, acoustic guardrails, reproducibility, operating modes, adaptive target, and temporal decay control.
permalink: /engineering/
---

DecayCore is designed as a conservative measurement and FIR-correction system. Its engineering documentation separates acoustic intent, numerical safeguards, operating policy, and measured evidence so that correction choices remain inspectable.

## Start with the design rationale

<div class="doc-grid doc-grid--two">
  <section class="doc-card">
    <h3><a href="{{ '/Why_DecayCore_Works.html' | relative_url }}">Why DecayCore Works</a></h3>
    <p>A practical overview of measurement alignment, confidence-weighted correction, phase safety, decay control, and headroom.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/Academic_DSP_Explanation.html' | relative_url }}">Academic DSP Rationale</a></h3>
    <p>The mathematical model and the implemented sequence from time-of-flight removal through FIR synthesis.</p>
  </section>
</div>

## Safety and reproducibility

<div class="doc-grid doc-grid--two">
  <section class="doc-card">
    <h3><a href="{{ '/DecayCore_dsp_guards.html' | relative_url }}">DSP Guards Reference</a></h3>
    <p>Principled acoustic guards, technical numerical guards, dependency clusters, and operational triggers.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/Stability_and_Reproducibility.html' | relative_url }}">Stability and Reproducibility</a></h3>
    <p>Deterministic processing, bounded correction, input robustness, comparable evaluation, and run auditing.</p>
  </section>
</div>

## Modes and room-aware behavior

<div class="doc-grid">
  <section class="doc-card">
    <h3><a href="{{ '/Modes.html' | relative_url }}">AUTO, BASIC, and ADVANCED</a></h3>
    <p>How operating modes change search behavior, exposed controls, and conservative defaults.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/temporal-decay-control/' | relative_url }}">Temporal Decay Control</a></h3>
    <p>Why low-frequency problems are evaluated in time as well as amplitude.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/adaptive-target/' | relative_url }}">Adaptive Target</a></h3>
    <p>Room-aware target synthesis and its measurement-metadata requirements.</p>
  </section>
</div>

## Evidence and advanced workflows

<div class="doc-grid doc-grid--two">
  <section class="doc-card">
    <h3><a href="{{ '/performance/' | relative_url }}">Measured performance case study</a></h3>
    <p>Methodology, limitations, source REW data, response, group-delay, waterfall, and full report.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/hybrid-iir-fir/' | relative_url }}">Hybrid IIR + FIR correction</a></h3>
    <p>An optional workflow for narrow low-frequency cuts before bounded FIR synthesis.</p>
  </section>
</div>

The user-facing workflow remains in the [User Manual]({{ '/User_Manual.html' | relative_url }}). These engineering documents explain why the controls and defaults behave as they do.
