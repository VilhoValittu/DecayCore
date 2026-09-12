---
title: How DecayCore works
nav_title: How it works
description: Understand DecayCore's correction principles, focused features, mathematical model, DSP guards, and operating policies.
permalink: /how-it-works/
---

Start with the correction concepts, then go deeper only when you need the mathematical model or implementation-level reference.

## Concepts

<div class="doc-grid doc-grid--two">
  <section class="doc-card">
    <h3><a href="{{ '/Why_DecayCore_Works.html' | relative_url }}">Why DecayCore Works</a></h3>
    <p>A plain-language explanation of measurement confidence, cuts-first correction, phase safety, decay control, and headroom.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/Comparison_vs_EQ.html' | relative_url }}">DecayCore vs conventional EQ</a></h3>
    <p>How a bounded magnitude-and-time workflow differs from magnitude-focused equalization.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/temporal-decay-control/' | relative_url }}">Temporal Decay Control</a></h3>
    <p>Why low-frequency decay is treated separately from ordinary magnitude matching.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/adaptive-target/' | relative_url }}">Adaptive Target</a></h3>
    <p>How bounded target changes are derived from broad, stereo-consistent room evidence.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/User_Manual.html#4-choose-the-target-and-filter-type' | relative_url }}">Filter types</a></h3>
    <p>When to use Asymmetric, Minimum Phase, Mixed Phase, or Linear Phase correction.</p>
  </section>
</div>

## Workflows and focused features

<div class="doc-grid">
  <section class="doc-card">
    <h3><a href="{{ '/Modes.html' | relative_url }}">AUTO, BASIC, and ADVANCED</a></h3>
    <p>Search behavior, manual control, defaults, and policy clamps.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/hybrid-iir-fir/' | relative_url }}">Hybrid IIR + FIR</a></h3>
    <p>Division of narrow modal cuts and broadband correction between two filter stages.</p>
  </section>
</div>

## Technical reference

<div class="doc-grid">
  <section class="doc-card">
    <h3><a href="{{ '/Official_Manual.html' | relative_url }}">DSP pipeline</a></h3>
    <p>The processing stages, control roles, phase strategies, synthesis, export, and cache contract.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/Academic_DSP_Explanation.html' | relative_url }}">Mathematical model</a></h3>
    <p>Equations for alignment, confidence, magnitude shaping, phase construction, TDC, and FIR synthesis.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/DecayCore_dsp_guards.html' | relative_url }}">DSP guards</a></h3>
    <p>The complete taxonomy of acoustic-policy and numerical-safety guards.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/Stability_and_Reproducibility.html' | relative_url }}">Stability and reproducibility</a></h3>
    <p>Requirements for comparable runs and the diagnostics used to explain differences.</p>
  </section>
</div>

## Terminology

- [Glossary]({{ '/glossary/' | relative_url }}) — short definitions for the terms used across these pages

For application steps rather than engineering detail, use the [User Manual]({{ '/User_Manual.html' | relative_url }}).
