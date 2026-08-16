---
title: DecayCore — Measure the room. Reveal the music.
description: Measure your listening room and create FIR filters for clearer, more controlled playback with DecayCore.
permalink: /
hide_title: true
wide_content: true
image: https://vilhovalittu.github.io/DecayCore/pics/DecayCore_logo_light.png
---

<section class="hero">
  <div class="hero__content">
    <p class="hero__eyebrow">Room measurement and FIR correction</p>
    <h1>Measure the room.<br><span>Reveal the music.</span></h1>
    <p class="hero__copy hero__copy--lead">DecayCore measures how your speakers behave in your room and creates correction filters for your playback system.</p>
    <p class="hero__copy">Use the guided measurement workflow or import existing measurements, then export WAV filters for CamillaDSP, Roon, Equalizer APO, MiniDSP, and other FIR-capable systems.</p>
    <div class="action-row">
      <a class="button button--primary" href="https://github.com/VilhoValittu/DecayCore/releases/latest">Download DecayCore</a>
      <a class="button" href="{{ '/getting-started/' | relative_url }}">Create your first filter</a>
      <a class="action-link" href="{{ '/performance/' | relative_url }}">See a measured example <span aria-hidden="true">→</span></a>
    </div>
  </div>
  <div class="hero__visual">
    <div class="hero__logo-frame">
      <img class="hero__logo" src="{{ '/pics/DecayCore_logo.svg' | relative_url }}" alt="" width="256" height="256">
    </div>
    <p class="hero__signal" aria-hidden="true"><span></span><i></i><span></span></p>
  </div>
</section>

## From measurement to filter

1. **Measure or import.** Measure left and right speakers in DecayCore, or load compatible REW text or impulse-response files.
2. **Generate.** Start with Automatic mode and the Asymmetric filter type. DecayCore searches for a conservative result and reports the settings it used.
3. **Export and verify.** Load the WAV filters into your convolver, then measure again with correction active.

<p class="section-link"><a href="{{ '/getting-started/' | relative_url }}">Follow the complete first-filter workflow <span aria-hidden="true">→</span></a></p>

## What DecayCore is designed to do

<div class="feature-grid">
  <section class="feature-card">
    <h3>Correct measured problems</h3>
    <p>DecayCore reduces supported peaks and broad response errors without trying to fill every deep cancellation.</p>
  </section>
  <section class="feature-card">
    <h3>Protect headroom</h3>
    <p>Correction strength, bass boost, timing changes, and filter gain stay inside explicit safety limits.</p>
  </section>
  <section class="feature-card">
    <h3>Show its work</h3>
    <p>Result graphs and the exported summary explain the selected filter, warnings, limits, and effective settings.</p>
  </section>
</div>

## Measured evidence

<section class="home-evidence" aria-labelledby="measured-evidence-title">
  <div class="home-evidence__copy">
    <h3 id="measured-evidence-title">One inspectable room, not a universal promise</h3>
    <p>The published case study compares one uncorrected measurement with four FIR results generated from the same REW project. It includes response, phase, group-delay, step, distortion, and waterfall views.</p>
    <p><a class="text-link" href="{{ '/performance/' | relative_url }}">Read the method, limitations, and full report <span aria-hidden="true">→</span></a></p>
  </div>
  <figure class="evidence-figure">
    <a href="{{ '/performance/' | relative_url }}">
      <img src="{{ '/performance/source/SPL_15-250.jpg' | relative_url }}" alt="Response from 15 to 250 Hz before correction and with four DecayCore FIR modes" width="1597" height="783" loading="lazy">
    </a>
    <figcaption>Published 15–250 Hz comparison from the included REW project.</figcaption>
  </figure>
</section>

## Interface

<p>Work through the numbered pages from measurement and files to settings, target, correction, and results.</p>

<div class="screenshot-gallery screenshot-gallery--featured">
  <figure class="screenshot-item">
    <a href="{{ '/pics/ui_2.png' | relative_url }}">
      <img src="{{ '/pics/ui_2.png' | relative_url }}" alt="Measure page with audio devices and guided room-measurement controls" width="1536" height="960" loading="lazy">
    </a>
    <figcaption class="screenshot-item__caption">Measure — configure devices and run a guided session</figcaption>
  </figure>
  <figure class="screenshot-item">
    <a href="{{ '/pics/ui_4.png' | relative_url }}">
      <img src="{{ '/pics/ui_4.png' | relative_url }}" alt="Target page with measured speaker responses and target preview" width="1536" height="960" loading="lazy">
    </a>
    <figcaption class="screenshot-item__caption">Target — inspect the measurements and target curve</figcaption>
  </figure>
  <figure class="screenshot-item">
    <a href="{{ '/pics/ui_8.png' | relative_url }}">
      <img src="{{ '/pics/ui_8.png' | relative_url }}" alt="START and Results page ready to generate filters from loaded measurements" width="1536" height="960" loading="lazy">
    </a>
    <figcaption class="screenshot-item__caption">START / Results — generate filters and follow progress</figcaption>
  </figure>
</div>

## Learn more

<div class="doc-grid">
  <section class="doc-card">
    <h3><a href="{{ '/measurement-workflow/' | relative_url }}">Measurement</a></h3>
    <p>Supported platforms, microphone setup, guided capture, and external imports.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/User_Manual.html' | relative_url }}">User manual</a></h3>
    <p>The practical reference for settings, results, export, and troubleshooting.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/engineering/' | relative_url }}">Engineering</a></h3>
    <p>How DecayCore handles magnitude, phase, timing, decay, safety, and reproducibility.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/glossary/' | relative_url }}">Glossary</a></h3>
    <p>Short definitions for the acoustic and DSP terms used in the documentation.</p>
  </section>
</div>

DecayCore was formerly called CamillaFIR. The name changed to avoid confusion with CamillaDSP; CamillaDSP compatibility remains.
