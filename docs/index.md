---
title: DecayCore — Measure the room. Reveal the music.
description: Measure your listening room and create FIR filters for clearer, more controlled playback with DecayCore.
permalink: /
hide_title: true
wide_content: true
image: https://vilhovalittu.github.io/DecayCore/pics/DecayCore_logo_light.png
---

<section class="hero">
  <svg class="hero__decay" viewBox="0 0 1440 200" preserveAspectRatio="none" aria-hidden="true" focusable="false">
    <path class="hero__decay-ring" d="M0,100.0 L8,100.0 L16,100.0 L24,100.0 L32,100.0 L40,100.0 L48,100.0 L56,100.0 L64,100.0 L72,86.7 L80,46.1 L88,44.0 L96,81.0 L104,130.3 L112,157.3 L120,143.7 L128,100.0 L136,57.4 L144,45.5 L152,71.9 L160,117.2 L168,149.4 L176,146.4 L184,111.2 L192,69.0 L200,49.5 L208,65.6 L216,105.3 L224,140.3 L232,146.4 L240,120.0 L248,80.3 L256,55.3 L264,62.1 L272,95.1 L280,130.7 L288,144.0 L296,126.4 L304,90.8 L312,62.5 L320,61.0 L328,86.8 L336,121.1 L344,139.9 L352,130.4 L360,100.0 L368,70.3 L376,62.1 L384,80.4 L392,112.0 L400,134.4 L408,132.3 L416,107.8 L424,78.4 L432,64.8 L440,76.1 L448,103.7 L456,128.1 L464,132.3 L472,113.9 L480,86.3 L488,68.9 L496,73.6 L504,96.6 L512,121.4 L520,130.6 L528,118.3 L536,93.6 L544,73.9 L552,72.9 L560,90.8 L568,114.7 L576,127.8 L584,121.2 L592,100.0 L600,79.4 L608,73.6 L616,86.4 L624,108.3 L632,123.9 L640,122.5 L648,105.4 L656,85.0 L664,75.5 L672,83.4 L680,102.6 L688,119.5 L696,122.5 L704,109.7 L712,90.5 L720,78.4 L728,81.6 L736,97.6 L744,114.9 L752,121.3 L760,112.8 L768,95.5 L776,81.8 L784,81.1 L792,93.6 L800,110.2 L808,119.3 L816,114.7 L824,100.0 L832,85.6 L840,81.6 L848,90.5 L856,105.8 L864,116.7 L872,115.6 L880,103.8 L888,89.5 L896,83.0 L904,88.4 L912,101.8 L920,113.6 L928,115.6 L936,106.7 L944,93.4 L952,84.9 L960,87.2 L968,98.4 L976,110.4 L984,114.8 L992,108.9 L1000,96.9 L1008,87.3 L1016,86.9 L1024,95.5 L1032,107.1 L1040,113.4 L1048,110.3 L1056,100.0 L1064,90.0 L1072,87.2 L1080,93.4 L1088,104.0 L1096,111.6 L1104,110.9 L1112,102.6 L1120,92.7 L1128,88.1 L1136,91.9 L1144,101.3 L1152,109.5 L1160,110.9 L1168,104.7 L1176,95.4 L1184,89.5 L1192,91.1 L1200,98.9 L1208,107.2 L1216,110.3 L1224,106.2 L1232,97.8 L1240,91.2 L1248,90.9 L1256,96.9 L1264,104.9 L1272,109.4 L1280,107.1 L1288,100.0 L1296,93.0 L1304,91.1 L1312,95.4 L1320,102.8 L1328,108.1 L1336,107.6 L1344,101.8 L1352,94.9 L1360,91.8 L1368,94.4 L1376,100.9 L1384,106.6 L1392,107.6 L1400,103.3 L1408,96.8 L1416,92.7 L1424,93.8 L1432,99.2 L1440,105.0"/>
    <path class="hero__decay-tight" d="M0,100.0 L4,100.0 L8,100.0 L12,100.0 L16,100.0 L20,100.0 L24,100.0 L28,100.0 L32,100.0 L36,100.0 L40,100.0 L44,100.0 L48,100.0 L52,100.0 L56,100.0 L60,100.0 L64,100.0 L68,100.0 L72,86.8 L76,64.0 L80,48.8 L84,43.6 L88,48.9 L92,63.2 L96,83.4 L100,105.5 L104,125.5 L108,139.8 L112,146.2 L116,144.0 L120,133.9 L124,118.2 L128,100.0 L132,82.8 L136,69.6 L140,62.5 L144,62.6 L148,69.5 L152,81.5 L156,96.2 L160,110.9 L164,122.8 L168,130.0 L172,131.4 L176,127.0 L180,118.0 L184,106.2 L188,93.9 L192,83.4 L196,76.4 L200,74.0 L204,76.4 L208,83.0 L212,92.3 L216,102.5 L220,111.8 L224,118.4 L228,121.3 L232,120.3 L236,115.6 L240,108.4 L244,100.0 L248,92.1 L252,86.0 L256,82.7 L260,82.8 L264,85.9 L268,91.5 L272,98.3 L276,105.0 L280,110.5 L284,113.8 L288,114.5 L292,112.5 L296,108.3 L300,102.9 L304,97.2 L308,92.3 L312,89.1 L316,88.0 L320,89.1 L324,92.2 L328,96.5 L332,101.2 L336,105.4 L340,108.5 L344,109.8 L348,109.4 L352,107.2 L356,103.9 L360,100.0 L364,96.3 L368,93.5 L372,92.0 L376,92.0 L380,93.5 L384,96.1 L388,99.2 L392,102.3 L396,104.9 L400,106.4 L404,106.7 L408,105.8 L412,103.8 L416,101.3 L420,98.7 L424,96.5 L428,95.0 L432,94.5 L436,95.0 L440,96.4 L444,98.4 L448,100.5 L452,102.5 L456,103.9 L460,104.5 L464,104.3 L468,103.3 L472,101.8 L476,100.0 L480,98.3 L484,97.0 L488,96.3 L492,96.3 L496,97.0 L500,98.2 L504,99.6 L508,101.1 L512,102.2 L516,102.9 L520,103.1 L524,102.7 L528,101.8 L532,100.6 L536,99.4 L540,98.4 L544,97.7 L548,97.4 L552,97.7 L556,98.3 L560,99.2 L564,100.2 L568,101.2 L572,101.8 L576,102.1 L580,102.0 L584,101.5 L588,100.8 L592,100.0 L596,99.2 L600,98.6 L604,98.3 L608,98.3 L612,98.6 L616,99.2 L620,99.8 L624,100.5 L628,101.0 L632,101.4 L636,101.4 L640,101.2 L644,100.8 L648,100.3 L652,99.7 L656,99.2 L660,98.9 L664,98.8 L668,98.9 L672,99.2 L676,99.7 L680,100.1 L684,100.5 L688,100.8 L692,101.0 L696,100.9 L700,100.7 L704,100.4 L708,100.0 L712,99.6 L716,99.4 L720,99.2 L724,99.2 L728,99.4 L732,99.6 L736,99.9 L740,100.2 L744,100.5 L748,100.6 L752,100.7 L756,100.6 L760,100.4 L764,100.1 L768,99.9 L772,99.7 L776,99.5 L780,99.5 L784,99.5 L788,99.6 L792,99.8 L796,100.1 L800,100.2 L804,100.4 L808,100.4 L812,100.4 L816,100.3 L820,100.2 L824,100.0 L828,99.8 L832,99.7 L836,99.6 L840,99.6 L844,99.7 L848,99.8 L852,100.0 L856,100.1 L860,100.2 L864,100.3 L868,100.3 L872,100.3 L876,100.2 L880,100.1 L884,99.9 L888,99.8 L892,99.8 L896,99.7 L900,99.8 L1440,100"/>
  </svg>
  <div class="wrap hero__grid">
    <div class="hero__content">
      <p class="eyebrow">Room measurement and FIR correction</p>
      <h1>Measure the room.<br><span>Reveal the music.</span></h1>
      <p class="hero__lead">You know that bass note that seems to turn up on every record? Your room may have a favourite. DecayCore measures your speakers in your listening room and creates FIR correction filters to help bring things back into balance.</p>
      <p class="hero__copy">Export WAV filters for CamillaDSP, Roon, Equalizer APO, MiniDSP, and other FIR-capable systems.</p>
      <div class="action-row">
        <a class="button button--primary" href="https://github.com/VilhoValittu/DecayCore/releases/latest">Download DecayCore</a>
        <a class="button" href="{{ '/getting-started/' | relative_url }}">Create your first filter</a>
      </div>
      <p class="hero__availability">The packaged download includes guided measurement and Automatic mode. The public source checkout provides Basic and Advanced manual filtering.</p>
    </div>
    <figure class="hero__visual">
      <a class="window" href="{{ '/results/' | relative_url }}" aria-label="Read how to interpret a DecayCore result">
        <span class="window__bar" aria-hidden="true"><i></i><i></i><i></i><span class="window__url">127.0.0.1:8080</span></span>
        <img src="{{ '/pics/result-response-example.png' | relative_url }}" alt="DecayCore result showing target-tracking figures and a left-channel response graph" width="1610" height="845">
      </a>
      <figcaption>The exported-filter curve is a prediction, not a follow-up room measurement.</figcaption>
    </figure>
  </div>
</section>

<section class="strip" aria-label="Compatible systems">
  <div class="wrap strip__inner">
    <p class="strip__label">Exports filters for</p>
    <ul class="chips">
      <li>CamillaDSP</li>
      <li>Roon</li>
      <li>Equalizer APO</li>
      <li>MiniDSP</li>
      <li>Other FIR convolvers</li>
    </ul>
  </div>
</section>

<section class="band">
  <div class="wrap">
    <p class="kicker">How it works</p>
    <h2>Three steps from measurement to listening</h2>
    <ol class="steps">
      <li class="step">
        <h3>Measure in DecayCore</h3>
        <p>Put the microphone where you listen and use the guided workflow. The session includes RT60 and harmonic data to help assess lingering bass and the risk of adding boost. Compatible REW text and impulse-response imports remain available when you need them.</p>
      </li>
      <li class="step">
        <h3>Generate</h3>
        <p>Start with Automatic mode and the Asymmetric filter type. DecayCore tries conservative settings and shows you which ones it chose. There is plenty of time to become very particular about them later.</p>
      </li>
      <li class="step">
        <h3>Load and listen</h3>
        <p>Play something familiar. Yes, <em>Hotel California</em> counts. Compare with bypass at the same volume and give the music a chance before opening another graph.</p>
      </li>
    </ol>
    <p class="section-link"><a href="{{ '/getting-started/' | relative_url }}">Follow the complete first-filter workflow <span aria-hidden="true">→</span></a></p>
  </div>
</section>

<section class="band band--tint">
  <div class="wrap">
    <p class="kicker">Features</p>
    <h2>What DecayCore is designed to do</h2>
    <div class="feature-grid">
      <section class="feature-card">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="3.5"/><path d="M12 1.5v4M12 18.5v4M1.5 12h4M18.5 12h4"/></svg>
        <h3>Correct measured problems</h3>
        <p>Take down the bass peak that keeps stealing the show and smooth broad tonal imbalances where the measurements support it. Deep cancellations need a different approach; pouring boost into them mostly gives the amplifier more work.</p>
      </section>
      <section class="feature-card">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.5l7.5 3v6c0 4.7-3.2 8.3-7.5 10-4.3-1.7-7.5-5.3-7.5-10v-6z"/><path d="M8.5 12l2.5 2.5 4.5-5"/></svg>
        <h3>Protect headroom</h3>
        <p>A kick drum needs somewhere to go. Explicit limits on correction strength, bass boost, timing changes, and filter gain keep the correction within bounds.</p>
      </section>
      <section class="feature-card">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 3.5v17h17"/><path d="M7.5 15l3.5-5 3 3 5-6.5"/></svg>
        <h3>Explain what it changed</h3>
        <p>The graphs and summary show what changed, where the limits stepped in, and what needs your attention. Useful when you want to know why that filter won.</p>
      </section>
      <section class="feature-card">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12h3l1.5-7 2.5 14 2.2-11 1.8 8 1.3-5 1 2H22"/></svg>
        <h3>Control decay and timing</h3>
        <p>Some bass notes linger well past their invitation. TDC and bounded phase correction address bass decay and timing where the measurements give them something reliable to work with.</p>
      </section>
    </div>
  </div>
</section>

<section class="band">
  <div class="wrap">
    <p class="kicker">A look inside</p>
    <h2>One workflow, from input to result</h2>
    <p class="band__intro">Work through the pages from measurement to export. Automatic mode handles the search and shows its chosen settings, so you have a starting point for listening without spending the evening trying every combination.</p>
    <div class="shots">
      <figure class="shot">
        <a href="{{ '/pics/ui_1.png' | relative_url }}"><img src="{{ '/pics/ui_1.png' | relative_url }}" alt="Basic tab with the mode selector, automatic goal, and target strategy" width="1536" height="960" loading="lazy"></a>
        <figcaption>Basic: choose the mode, goal, and target strategy.</figcaption>
      </figure>
      <figure class="shot">
        <a href="{{ '/pics/ui_4.png' | relative_url }}"><img src="{{ '/pics/ui_4.png' | relative_url }}" alt="Target tab with a target curve preview and correction limits" width="1536" height="960" loading="lazy"></a>
        <figcaption>Target: shape the target curve, leveling, and gain limits.</figcaption>
      </figure>
      <figure class="shot">
        <a href="{{ '/pics/ui_6.png' | relative_url }}"><img src="{{ '/pics/ui_6.png' | relative_url }}" alt="IR Window and Decay Control tab" width="1536" height="960" loading="lazy"></a>
        <figcaption>IR Window &amp; Decay Control: export windowing and temporal processing.</figcaption>
      </figure>
    </div>
  </div>
</section>

<section class="band band--dark">
  <div class="wrap listen">
    <div class="listen__lead">
      <p class="kicker">The graph is not the result</p>
      <h2>The result is what you hear</h2>
      <p>A tidy response plot is satisfying. So is hearing the bass line clearly enough to follow what the player is doing. The second one is why I bother.</p>
      <p class="section-link"><a href="{{ '/results/' | relative_url }}">How to read and test a result <span aria-hidden="true">→</span></a></p>
    </div>
    <div class="listen__body">
      <p>Load the filters, start quietly, and play a few records you know well. Compare with bypass at the same volume. Listen to the weight of a kick drum, the body of a voice, and whether the bass lets the next note through. If it sounds worse, it is worse. Your ears do not owe the graph a positive review.</p>
      <p>A corrected-system measurement can be useful, but DecayCore does not route its sweep through the exported filters automatically. That requires a separate loop through your convolver.</p>
    </div>
  </div>
</section>

<section class="band">
  <div class="wrap">
    <p class="kicker">Learn more</p>
    <h2>Where to go next</h2>
    <div class="doc-grid doc-grid--four">
      <section class="doc-card">
        <h3><a href="{{ '/installation/' | relative_url }}">Installation</a></h3>
        <p>Install a packaged release on Windows, Linux, or macOS, or run the manual workflow from source.</p>
      </section>
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
    <p class="fineprint">DecayCore was formerly called CamillaFIR. The name changed to avoid confusion with CamillaDSP; CamillaDSP compatibility remains.</p>
  </div>
</section>

<section class="cta">
  <div class="wrap cta__inner">
    <div class="cta__brand">
      <img class="cta__mark" src="{{ '/pics/DecayCore_logo.svg' | relative_url }}" alt="" width="88" height="88">
      <div>
        <h2>Create your first filter</h2>
        <p>Download DecayCore, measure Left and Right, and let Automatic mode find a starting point.</p>
      </div>
    </div>
    <div class="action-row">
      <a class="button button--primary" href="https://github.com/VilhoValittu/DecayCore/releases/latest">Download DecayCore</a>
      <a class="button" href="{{ '/getting-started/' | relative_url }}">Start here</a>
    </div>
  </div>
</section>
