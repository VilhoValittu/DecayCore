---
title: DecayCore — Measure the room. Reveal the music.
description: Measure the room. Reveal the music. DecayCore turns acoustic measurements into bounded, phase-aware FIR correction for a more truthful listening experience.
permalink: /
hide_title: true
wide_content: true
image: https://vilhovalittu.github.io/DecayCore/pics/DecayCore_logo_light.png
---

<section class="hero">
  <div class="hero__content">
    <p class="hero__eyebrow">DecayCore · Measurement-led FIR correction</p>
    <h1>Measure the room.<br><span>Reveal the music.</span></h1>
    <p class="hero__copy hero__copy--lead">DecayCore listens to how your loudspeakers and room behave, then builds bounded, phase-aware FIR correction that lets the recording come through.</p>
    <p class="hero__copy">Magnitude, phase, timing, and low-frequency decay stay inside explicit acoustic guardrails. Export convolution-ready WAV filters for CamillaDSP, Roon, Equalizer APO, MiniDSP, and other FIR-capable DSP engines.</p>
    <div class="action-row">
      <a class="button button--primary" href="https://github.com/VilhoValittu/DecayCore/releases/latest">Download DecayCore</a>
      <a class="button" href="{{ '/performance/' | relative_url }}">Explore measured performance</a>
      <a class="action-link" href="{{ '/getting-started/' | relative_url }}">Getting started <span aria-hidden="true">→</span></a>
    </div>
  </div>
  <div class="hero__visual">
    <div class="hero__logo-frame">
      <img class="hero__logo" src="{{ '/pics/DecayCore_logo.svg' | relative_url }}" alt="DecayCore Resonance Disk logo" width="256" height="256">
    </div>
    <p class="hero__signal" aria-hidden="true"><span></span><i></i><span></span></p>
  </div>
</section>

<section class="home-evidence" aria-labelledby="measured-evidence-title">
  <div class="home-evidence__copy">
    <p class="section-kicker">Inspect the evidence</p>
    <h2 id="measured-evidence-title">A measured case study, not a universal promise</h2>
    <p>The published REW project compares an uncorrected right-channel measurement with Linear, Minimum, Mixed, and Asymmetric FIR results generated from the same source project.</p>
    <ul class="evidence-list">
      <li>One inspectable room and measurement dataset</li>
      <li>Uncorrected and four corrected responses</li>
      <li>Response, group-delay, phase, step, distortion, and waterfall views</li>
    </ul>
    <p><a class="text-link" href="{{ '/performance/' | relative_url }}">Read the methodology, limitations, and full report <span aria-hidden="true">→</span></a></p>
  </div>
  <figure class="evidence-figure">
    <a href="{{ '/performance/' | relative_url }}">
      <img src="{{ '/performance/source/SPL_15-250.jpg' | relative_url }}" alt="REW response plot from 15 to 250 Hz comparing the uncorrected right channel with Asymmetric, Linear, Minimum, and Mixed FIR correction" width="1597" height="783" loading="lazy">
    </a>
    <figcaption>Published 15–250 Hz comparison from the included REW project. This single-room result illustrates the workflow; it does not predict results in other rooms.</figcaption>
  </figure>
</section>

## Engineering principles

<div class="feature-grid">
  <section class="feature-card">
    <h3><a href="{{ '/DecayCore_dsp_guards.html' | relative_url }}">Acoustic guardrails</a></h3>
    <p>Boost, cut, pre-energy, phase, timing, and low-confidence regions are controlled by explicit DSP policies.</p>
  </section>
  <section class="feature-card">
    <h3><a href="{{ '/Stability_and_Reproducibility.html' | relative_url }}">Stability and reproducibility</a></h3>
    <p>Deterministic processing, bounded correction, comparable evaluation, and an audit trail keep runs inspectable.</p>
  </section>
  <section class="feature-card">
    <h3><a href="{{ '/Why_DecayCore_Works.html' | relative_url }}">Time-domain rationale</a></h3>
    <p>Measurement alignment, confidence-weighted shaping, phase safety, and decay control are treated as one workflow.</p>
  </section>
</div>

<p class="section-link"><a href="{{ '/engineering/' | relative_url }}">Explore the engineering documentation <span aria-hidden="true">→</span></a></p>

## Interface

<p>The interface follows the same path as the signal flow: measure or import, configure bounded correction, run the process, and inspect the result.</p>

<div class="screenshot-gallery screenshot-gallery--featured">
  <figure class="screenshot-item">
    <a href="{{ '/pics/ui_2.png' | relative_url }}">
      <img src="{{ '/pics/ui_2.png' | relative_url }}" alt="Measure tab for configuring capture devices and running guided room measurements" width="1536" height="960" loading="lazy">
    </a>
    <figcaption class="screenshot-item__caption">Measure — configure capture devices and run guided room measurements</figcaption>
  </figure>
  <figure class="screenshot-item">
    <a href="{{ '/pics/ui_8.png' | relative_url }}">
      <img src="{{ '/pics/ui_8.png' | relative_url }}" alt="Start and Results tab ready to launch correction with the left and right measurements loaded" width="1536" height="960" loading="lazy">
    </a>
    <figcaption class="screenshot-item__caption">Start / Results — launch correction and follow its progress</figcaption>
  </figure>
  <figure class="screenshot-item">
    <a href="{{ '/pics/ui_4.png' | relative_url }}">
      <img src="{{ '/pics/ui_4.png' | relative_url }}" alt="Target tab showing the measured speaker responses and target curve preview" width="1536" height="960" loading="lazy">
    </a>
    <figcaption class="screenshot-item__caption">Target — shape correction against the measured response</figcaption>
  </figure>
</div>

<p class="section-link"><a href="{{ '/User_Manual.html' | relative_url }}">Follow the complete interface workflow in the user manual <span aria-hidden="true">→</span></a></p>

## Documentation

<div class="doc-grid">
  <section class="doc-card">
    <h3><a href="{{ '/getting-started/' | relative_url }}">Getting started</a></h3>
    <p>The shortest path from release download to generated FIR filters.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/installation/' | relative_url }}">Installation</a></h3>
    <p>Packaged releases and source installation for supported platforms.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/engineering/' | relative_url }}">Engineering</a></h3>
    <p>DSP rationale, safety policies, modes, decay control, and reproducibility.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/performance/' | relative_url }}">Performance case study</a></h3>
    <p>Methodology, limitations, source data, measured plots, and the full report.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/User_Manual.html' | relative_url }}">User manual</a></h3>
    <p>Practical reference for measurement, configuration, export, and deployment.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/faq/' | relative_url }}">FAQ</a></h3>
    <p>Short answers about measurement, compatibility, exports, and troubleshooting.</p>
  </section>
</div>

DecayCore was formerly known as CamillaFIR. The project was renamed to avoid confusion with CamillaDSP while keeping full compatibility with CamillaDSP.
