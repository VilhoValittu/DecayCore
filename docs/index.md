---
title: DecayCore - FIR Room Correction and Measurement Tool
description: DecayCore is a free FIR room correction and acoustic measurement tool for CamillaDSP, convolution WAV filters, phase-aware correction and temporal decay control.
permalink: /
hide_title: true
---

<section class="hero">
  <div>
    <p class="hero__eyebrow">FIR room correction and measurement</p>
    <h1>DecayCore — FIR Room Correction and Measurement Tool</h1>
    <p class="hero__copy">A free acoustic measurement tool, FIR room correction tool, and FIR filter generator for CamillaDSP, convolution WAV filters, Roon convolution workflows, Equalizer APO, and other FIR-capable DSP engines. DecayCore was formerly known as CamillaFIR.</p>
    <p class="hero__copy">DecayCore focuses on physically sane, band-limited room correction, phase-aware correction, automatic target optimization, and Temporal Decay Control for low-frequency room behavior.</p>
    <div class="action-row">
      <a class="button button--primary" href="https://github.com/VilhoValittu/DecayCore/releases">Download releases</a>
      <a class="button" href="{{ '/getting-started/' | relative_url }}">Getting started</a>
      <a class="button" href="{{ '/installation/' | relative_url }}">Installation</a>
      <a class="button" href="{{ '/User_Manual.html' | relative_url }}">User manual</a>
    </div>
  </div>
  <div class="hero__visual">
    <img class="hero__logo" src="{{ '/pics/DecayCore_logo_light.png' | relative_url }}" alt="DecayCore logo" width="250" height="250">
  </div>
</section>

## Built for correction that stays believable

<div class="feature-grid">
  <section class="feature-card">
    <h3>Measurement-first workflow</h3>
    <p>Release builds include DecayCore's own acoustic measurement workflow for consistent timing, phase, and export behavior.</p>
  </section>
  <section class="feature-card">
    <h3>Convolution-ready export</h3>
    <p>Generate FIR WAV filters for CamillaDSP and other engines that support convolution impulse responses.</p>
  </section>
  <section class="feature-card">
    <h3>Guarded DSP</h3>
    <p>Correction is bounded by boost, phase, timing, and low-frequency safety policies instead of chasing a perfect-looking graph.</p>
  </section>
</div>

## Core capabilities

<div class="feature-grid">
  <section class="feature-card">
    <h3>Automatic target optimization</h3>
    <p>AUTO mode searches for conservative, explainable correction choices from the loaded measurements.</p>
  </section>
  <section class="feature-card">
    <h3>Multiple FIR modes</h3>
    <p>Linear Phase, Minimum Phase, Mixed Phase, and Asymmetric FIR workflows are supported.</p>
  </section>
  <section class="feature-card">
    <h3>Temporal Decay Control</h3>
    <p>Low-frequency room behavior can be shaped with time-domain-aware controls rather than simple amplitude flattening.</p>
  </section>
</div>

## Documentation

<div class="doc-grid">
  <section class="doc-card">
    <h3><a href="{{ '/getting-started/' | relative_url }}">Getting started</a></h3>
    <p>The shortest path from release download to generated FIR filters.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/installation/' | relative_url }}">Installation</a></h3>
    <p>Install DecayCore from a release package or run it from Python source.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/measurement-workflow/' | relative_url }}">Measurement workflow</a></h3>
    <p>How DecayCore's built-in measurement flow fits the correction pipeline.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/camilladsp-fir-room-correction/' | relative_url }}">CamillaDSP FIR correction</a></h3>
    <p>Use exported FIR filters in CamillaDSP convolution setups.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/fir-room-correction/' | relative_url }}">FIR room correction</a></h3>
    <p>The practical role of FIR filters in room correction workflows.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/minimum-phase-fir-generator/' | relative_url }}">Minimum Phase FIR Generator</a></h3>
    <p>Minimum phase FIR correction for low-latency convolution workflows.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/mixed-phase-room-correction/' | relative_url }}">Mixed Phase Room Correction</a></h3>
    <p>Phase-aware FIR correction with practical safety limits.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/temporal-decay-control/' | relative_url }}">Temporal Decay Control</a></h3>
    <p>Why bass problems are often time-domain problems too.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/faq/' | relative_url }}">FAQ</a></h3>
    <p>Short answers about measurement, exports, compatibility, and releases.</p>
  </section>
</div>

DecayCore was formerly known as CamillaFIR. The project was renamed to avoid confusion with CamillaDSP while keeping full compatibility with CamillaDSP.
