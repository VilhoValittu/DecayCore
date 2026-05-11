---
layout: default
title: "Quick Start: How to Use the DecayCore Filter Maker"
description: "Step-by-step guide to generating convolution filters using DecayCore, REW measurements, and the recommended Automatic Asymmetric mode."
permalink: /quick-start/
---

# Quick Start Guide: Generating FIR Filters

This guide shows you how to use **DecayCore**, an automated **FIR filter maker**, to create high-quality room correction filters for your audio system.

---

## Step 1: Measure with REW (Room EQ Wizard)
Before using the **FIR filter generator**, you need accurate measurement data.

1.  **Microphone Setup:** Connect your calibrated measurement microphone (e.g., UMIK-1).
2.  **Calibration:** Ensure your microphone calibration file is loaded in REW.
3.  **Sweep:** Run a frequency response sweep (typically 0 Hz – 24 kHz).
4.  **Export:**
    * Normal L/R workflow: go to `File` > `Export` > `Measurement as Text`.
    * **Crucial:** Select "Include Phase" in the text export settings.
    * Legacy L/R WAV workflow: use `Mono`, `float32`, `Normalise`, and `Place t=0 (256)`.
    * AUTO Bass Integration WAV workflow: export separate IR WAVs as `Mono` + `float32`, keep the same measurement gain and the same timing reference for every main/sub file, and do **not** normalize each file separately or move each file to its own `t=0`.
    * `AVR / Receiver (LFE+Main)`: export `L main only`, `R main only`, `L sub only`, and `R sub only`.
    * `Direct DAC / CamillaDSP sub output`: export `L main only`, `R main only`, `Sub 1 only`, and optionally `Sub 2 only`.

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

## Step 4: Generate and Export
Now, let the **FIR filter generator** process your data.

1.  **Launch DecayCore:** Open the application.
2.  **Load Measurement:** Select the normal L/R TXT or WAV files, or the separate main/sub WAV set if you are using AUTO Bass Integration.
3.  **Set Target Curve:** Choose a flat target or load a custom house curve.
4.  **Safety Guards:** We recommend a safe limit of **+3 dB** for Max Boost to avoid clipping.
    * **Low-Bass:** Enable safe bass correction policies.
5.  **Process:** Click generate. DecayCore produces a ZIP package containing your **convolution-ready** output files.

---

## Step 5: Apply to Your DSP
Load the resulting **WAV FIR filter** into your preferred engine:

* **CamillaDSP:** Add the exported WAV files to your convolution block. In `Direct DAC / CamillaDSP sub output` Bass Integration runs, also load the generated `Sub_...wav` for the subwoofer path.
* **Roon:** Upload the WAV (or ZIP) into Roon's Convolution settings.
* **Equalizer APO:** Load the WAV filter and ensure sufficient preamp headroom.

## Step 6: Verify
**Always re-measure** your speakers with the FIR filter active to confirm the correction behaves as expected.

If you used AUTO Bass Integration, verify that playback uses the same topology and crossover arrangement that was used during measurement.

---

## Why use DecayCore?
Unlike basic EQ, this **FIR filter maker** handles **excess phase** and **temporal decay (TDC)**, providing tighter bass and more repeatable tuning.

[← Back to Home]({{ site.baseurl }}/) | [Read the FAQ]({{ site.baseurl }}/faq)
