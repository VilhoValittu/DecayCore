---
layout: default
title: "Quick Start: How to Use the DecayCore Filter Maker"
description: "Step-by-step guide to generating convolution filters using DecayCore, built-in sweep measurements, and the recommended Automatic Asymmetric mode."
permalink: /quick-start/
---

This guide shows you how to use **DecayCore**, an automated **FIR filter maker**, to create high-quality room correction filters for your audio system.

---

## Step 1: Measure your speakers
Before using the **FIR filter generator**, you need accurate measurement data.

**Recommended: use the built-in DecayCore measurement tool**

1.  **Microphone Setup:** Connect your calibrated measurement microphone (e.g., UMIK-1).
2.  **Open the Measurement tab** in DecayCore and load your microphone calibration file.
3.  **Run sweeps** for the left and right channels separately.
4.  The built-in tool saves IR WAV files directly — load them from the Files tab when done.

**Platform support:** Measurement has been verified to work on Windows. Linux has been verified to work at least on Ubuntu 22.04. macOS could not be tested due to unavailable test hardware. Subwoofer measurement on Windows requires the playback device to be configured for 5.1 or 7.1 multichannel in Windows Sound settings.

**Alternative: import from REW (Room EQ Wizard)**

If you already have REW measurements:
*   TXT workflow: `File` > `Export` > `Measurement as Text` — select "Include Phase".
*   WAV/IR workflow: export as `Mono`, `float32`, `Normalise`, `Place t=0 (256)`.

---

## Step 2: Recommended Mode – Automatic Asymmetric
For the best balance between sonic accuracy and system stability, we recommend using the **Automatic Asymmetric mode**.

* **Why Asymmetric?** Unlike standard linear-phase filters, the **Asymmetric FIR filter maker** strategy provides high-precision correction with lower pre-ringing risk and optimized latency.
* **Automatic Workflow:**
    1.  Set the filter mode to **Asymmetric**.
    2.  Enable **DecayCore Automatic mode** to align your target curve and measurements automatically.
    3.  The engine handles **TOF (Time of Flight) removal** automatically to ensure phase correction acts only on excess phase.
    4.  It utilizes **Adaptive FDW (Frequency Dependent Windowing)** to weight correction based on acoustic confidence.

---

## Step 3: Generate and Export
Now, let the **FIR filter generator** process your data.

1.  **Launch DecayCore:** Open the application.
2.  **Load Measurement:** Select the L/R TXT or WAV files from the Files tab.
3.  **Set Target Curve:** Choose a flat target or load a custom house curve.
4.  **Safety Guards:** We recommend a safe limit of **+3 dB** for Max Boost to avoid clipping.
    * **Low-Bass:** Enable safe bass correction policies.
5.  **Process:** Click generate. DecayCore produces a ZIP package containing your **convolution-ready** output files.

---

## Step 4: Apply to Your DSP
Load the resulting **WAV FIR filter** into your preferred engine:

* **CamillaDSP:** Add the exported WAV files to your convolution block.
* **Roon:** Upload the WAV (or ZIP) into Roon's Convolution settings.
* **Equalizer APO:** Load the WAV filter and ensure sufficient preamp headroom.

## Step 5: Verify
**Always re-measure** your speakers with the FIR filter active to confirm the correction behaves as expected.

---

## Why use DecayCore?
Unlike basic EQ, this **FIR filter maker** handles **excess phase** and **temporal decay (TDC)**, providing tighter bass and more repeatable tuning.

[← Back to Home]({{ site.baseurl }}/) | [Read the FAQ]({{ site.baseurl }}/faq/)
