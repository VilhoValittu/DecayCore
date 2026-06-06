---
title: Performance report
nav_title: Performance
description: Embedded PDF performance report for DecayCore automatic mode and adaptive target comparison.
permalink: /performance/
---

This page embeds the DecayCore performance report PDF directly in the browser. If your browser does not display embedded PDFs, use the download link below.

<div class="performance-report">
  <div class="performance-report__actions">
    <a class="button button--primary" href="{{ '/performance/DecayCore_Performance.pdf' | relative_url }}">Open PDF</a>
    <a class="button" href="{{ '/performance/DecayCore_Performance.pdf' | relative_url }}" download>Download PDF</a>
  </div>

  <object
    class="performance-report__frame"
    data="{{ '/performance/DecayCore_Performance.pdf' | relative_url }}"
    type="application/pdf"
  >
    <p>Your browser cannot display the PDF inline. Use the <a href="{{ '/performance/DecayCore_Performance.pdf' | relative_url }}">Open PDF</a> link instead.</p>
  </object>
</div>
