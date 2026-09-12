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
    <p class="hero__copy hero__copy--lead">You know that bass note that seems to turn up on every record? Your room may have a favourite. DecayCore measures your speakers in your listening room and creates FIR correction filters to help bring things back into balance.</p>
    <p class="hero__copy">Start with DecayCore's guided measurement: it captures room decay (RT60) and harmonic distortion alongside the speaker response, giving the correction more measured information to work with. Export WAV filters for CamillaDSP, Roon, Equalizer APO, MiniDSP, and other FIR-capable systems, then put on a record you know by heart.</p>
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

1. **Measure in DecayCore.** Put the microphone where you listen and use the guided workflow. The session includes RT60 and harmonic data to help assess lingering bass and the risk of adding boost. Compatible REW text and impulse-response imports remain available when you need them.
2. **Generate.** Start with Automatic mode and the Asymmetric filter type. DecayCore tries conservative settings and shows you which ones it chose. There is plenty of time to become very particular about them later.
3. **Load and listen.** Play something familiar. Yes, *Hotel California* counts. Compare with bypass at the same volume and give the music a chance before opening another graph.

<p class="section-link"><a href="{{ '/getting-started/' | relative_url }}">Follow the complete first-filter workflow <span aria-hidden="true">→</span></a></p>

## What DecayCore is designed to do

<div class="feature-grid">
  <section class="feature-card">
    <h3>Correct measured problems</h3>
    <p>Take down the bass peak that keeps stealing the show and smooth broad tonal imbalances where the measurements support it. Deep cancellations need a different approach; pouring boost into them mostly gives the amplifier more work.</p>
  </section>
  <section class="feature-card">
    <h3>Protect headroom</h3>
    <p>A kick drum needs somewhere to go. Explicit limits on correction strength, bass boost, timing changes, and filter gain keep the correction within bounds.</p>
  </section>
  <section class="feature-card">
    <h3>Explain what it changed</h3>
    <p>The graphs and summary show what changed, where the limits stepped in, and what needs your attention. Useful when you want to know why that filter won.</p>
  </section>
  <section class="feature-card">
    <h3>Control decay and timing</h3>
    <p>Some bass notes linger well past their invitation. TDC and bounded phase correction address bass decay and timing where the measurements give them something reliable to work with.</p>
  </section>
</div>

## One workflow, from input to result

<p>Work through the numbered pages from measurement to export. Automatic mode handles the search and shows its chosen settings, so you have a starting point for listening without spending the evening trying every combination.</p>

<div class="screenshot-gallery screenshot-gallery--single">
  <figure class="screenshot-item">
    <a href="{{ '/pics/ui_8.png' | relative_url }}">
      <img src="{{ '/pics/ui_8.png' | relative_url }}" alt="DecayCore START and Results page ready to generate FIR correction filters" width="1536" height="960" loading="lazy">
    </a>
    <figcaption class="screenshot-item__caption">START / Results — generate filters, follow progress, and inspect the selected solution</figcaption>
  </figure>
</div>

## The graph is not the result

<div class="home-evidence home-evidence--copy">
  <div class="home-evidence__copy">
    <p class="section-kicker">Listen first</p>
    <h2>The result is what you hear</h2>
    <p>A tidy response plot is satisfying. So is hearing the bass line clearly enough to follow what the player is doing. The second one is why I bother.</p>
    <p>Load the filters, start quietly, and play a few records you know well. Compare with bypass at the same volume. Listen to the weight of a kick drum, the body of a voice, and whether the bass lets the next note through. If it sounds worse, it is worse. Your ears do not owe the graph a positive review.</p>
    <p>A corrected-system measurement can be useful, but DecayCore does not route its sweep through the exported filters automatically. That requires a separate loop through your convolver.</p>
    <p class="section-link"><a href="{{ '/results/' | relative_url }}">How to read and test a result <span aria-hidden="true">→</span></a></p>
  </div>
</div>

## Works with

<p class="compatibility-list">CamillaDSP <span>·</span> Roon <span>·</span> Equalizer APO <span>·</span> MiniDSP <span>·</span> other FIR convolvers</p>

## Learn more

<div class="doc-grid">
  <section class="doc-card">
    <h3><a href="{{ '/User_Manual.html' | relative_url }}">User manual</a></h3>
    <p>What the controls do, how to load your filters, and where to look when something sounds odd.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/how-it-works/' | relative_url }}">How it works</a></h3>
    <p>The thinking behind the correction, with phase, decay, and the maths available when curiosity wins.</p>
  </section>
  <section class="doc-card">
    <h3><a href="{{ '/faq/' | relative_url }}">FAQ</a></h3>
    <p>Questions about measurements, filters, targets, and getting everything playing nicely together.</p>
  </section>
</div>

DecayCore was formerly called CamillaFIR. The name changed to avoid confusion with CamillaDSP; CamillaDSP compatibility remains.
