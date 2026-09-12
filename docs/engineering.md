---
title: How DecayCore works
nav_title: How it works
description: Find out how DecayCore handles bass, phase, decay, and headroom, with the maths available when you want to go further.
permalink: /how-it-works/
---

If you have ever moved a speaker a few centimetres and then listened to the same track six times, you already know why the details matter. These pages explain what DecayCore measures, what it tries to correct, and where it leaves well alone. Start with the concepts; the equations are here for the evening that was supposed to involve just one album.

## Concepts

<div class="doc-grid doc-grid--two">
  <section class="doc-card">
    <h3><a href="{{ '/Why_DecayCore_Works.html' | relative_url }}">Why DecayCore Works</a></h3>
    <p>Why some peaks are worth cutting, some dips are best left alone, and your amplifier deserves a little breathing room.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/Comparison_vs_EQ.html' | relative_url }}">DecayCore vs conventional EQ</a></h3>
    <p>What looking at timing and decay adds to the familiar job of adjusting frequency response.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/temporal-decay-control/' | relative_url }}">Temporal Decay Control</a></h3>
    <p>What to do when a bass note hangs around after the player has moved on.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/adaptive-target/' | relative_url }}">Adaptive Target</a></h3>
    <p>How the measurements from both speakers guide small changes to the bass target.</p>
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
    <p>Let Automatic find a starting point, use Basic for manual adjustment, or get into the details with Advanced.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/hybrid-iir-fir/' | relative_url }}">Hybrid IIR + FIR</a></h3>
    <p>How IIR handles narrow bass resonances while FIR takes care of broader correction.</p>
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
    <p>The limits that keep correction strength, timing, and the underlying calculations under control.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/Stability_and_Reproducibility.html' | relative_url }}">Stability and reproducibility</a></h3>
    <p>Requirements for comparable runs and the diagnostics used to explain differences.</p>
  </section>
</div>

## Terminology

- [Glossary]({{ '/glossary/' | relative_url }}) — short definitions for the terms used across these pages

For application steps rather than engineering detail, use the [User Manual]({{ '/User_Manual.html' | relative_url }}).
