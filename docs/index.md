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
    </div>
  </div>
  <div class="hero__visual">
    <div class="hero__logo-frame">
      <img class="hero__logo" src="{{ '/pics/DecayCore_logo.svg' | relative_url }}" alt="" width="256" height="256">
    </div>
    <p class="hero__signal" aria-hidden="true"><span></span><i></i><span></span></p>
  </div>
</section>

## How it works

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
  <section class="feature-card">
    <h3>Control decay and timing</h3>
    <p>Measured time-domain evidence can constrain phase correction and support bounded low-frequency decay control.</p>
  </section>
</div>

## One workflow, from input to result

<p>Work through the numbered pages from measurement and files to settings, target, correction, and results. Automatic mode handles the search; every effective choice remains visible in the result.</p>

<div class="screenshot-gallery screenshot-gallery--single">
  <figure class="screenshot-item">
    <a href="{{ '/pics/ui_8.png' | relative_url }}">
      <img src="{{ '/pics/ui_8.png' | relative_url }}" alt="DecayCore START and Results page ready to generate FIR correction filters" width="1536" height="960" loading="lazy">
    </a>
    <figcaption class="screenshot-item__caption">START / Results — generate filters, follow progress, and inspect the selected solution</figcaption>
  </figure>
</div>

## Result and verification

<div class="home-evidence home-evidence--copy">
  <div class="home-evidence__copy">
    <p class="section-kicker">Trust the complete signal chain</p>
    <h2>A generated graph is a prediction, not the final proof</h2>
    <p>Inspect the result and exported summary, deploy the filter at reduced level, and measure again with correction active. The verification measurement catches routing, gain, crossover, polarity, and sample-rate mistakes that an offline prediction cannot see.</p>
    <p class="section-link"><a href="{{ '/results/' | relative_url }}">Learn how to read and verify a result <span aria-hidden="true">→</span></a></p>
  </div>
</div>

## Works with

<p class="compatibility-list">CamillaDSP <span>·</span> Roon <span>·</span> Equalizer APO <span>·</span> MiniDSP <span>·</span> other FIR convolvers</p>

## Learn more

<div class="doc-grid">
  <section class="doc-card">
    <h3><a href="{{ '/User_Manual.html' | relative_url }}">User manual</a></h3>
    <p>The practical reference for settings, results, export, and troubleshooting.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/how-it-works/' | relative_url }}">How it works</a></h3>
    <p>How DecayCore handles magnitude, phase, timing, decay, safety, and reproducibility.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/faq/' | relative_url }}">FAQ</a></h3>
    <p>Direct answers about measurements, filters, targets, latency, safety, and deployment.</p>
  </section>
</div>

DecayCore was formerly called CamillaFIR. The name changed to avoid confusion with CamillaDSP; CamillaDSP compatibility remains.
