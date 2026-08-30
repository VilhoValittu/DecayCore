---
title: DecayCore User Manual
nav_title: Manual
description: Practical instructions for measuring, creating, exporting, deploying, and troubleshooting DecayCore FIR filters.
---

This manual follows the work you do in DecayCore: prepare measurements, choose a workflow, generate filters, inspect the result, and deploy it safely.

## 1. Before you start

You need:

- a calibrated measurement microphone
- separate Left and Right measurements
- a playback system that accepts FIR convolution filters
- enough control of playback volume to begin verification quietly

DecayCore accepts impulse-response WAV files and text exports containing frequency, magnitude, and phase. Packaged releases also include guided measurement and **DecayCore automatic mode (recommended)**.

> **Source-build note:** the public source checkout provides the complete **Basic** and **Advanced** manual workflows. Guided measurement and the native Automatic mode decision engine are included only in packaged releases.

### A safe first run

1. Measure Left and Right or load compatible files on **Files**.
2. Select **DecayCore automatic mode (recommended)** and **Asymmetric**.
3. Keep the **balanced** goal and **Adaptive: derive target from room acoustics** target strategy.
4. Press **START** on **START / Results**.
5. Review warnings and the winning solution.
6. Export the ZIP, load the filters into your convolver, and measure again.

When running from source, use **Basic** instead of Automatic mode.

## 2. Measure or import

### Guided measurement

Open **Measure** and configure the input device, output device, microphone calibration, selected channels, positions, and repeats. The guided session measures each selected channel, pauses when the microphone must move, rejects unusable takes, and saves one impulse-response WAV per channel.

Set measurement volume at the end of the signal chain, such as the amplifier or an analog volume control. Digital attenuation earlier in the chain also lowers measurement signal-to-noise ratio.

Keep all WAV files from one session at their original gain and timing. Do not normalize channels separately or move each impulse to a different time origin.

On Windows, subwoofer measurement requires the playback device to be configured for 5.1 or 7.1. The sweep uses the LFE channel. Main-speaker measurement has also been verified on Ubuntu 22.04; macOS measurement has not been verified on project hardware.

### REW text files

Export Left and Right separately with:

- frequency in hertz
- magnitude in decibels
- phase in degrees

Normal REW headers and comments are accepted. Phase is required for phase-aware correction.

### Impulse-response WAV files

Use mono WAV files with consistent sample rate, gain, and timing. When exporting from REW, use the same options for every channel; `float32`, normalization, and consistent `t=0` placement are suitable defaults.

### Input checklist

- Avoid clipping, background noise, and changed routing between channels.
- Use the same microphone position and procedure for corresponding measurements.
- Keep sample rates consistent when practical.
- Check that Left and Right files are not swapped.

## 3. Choose the operating mode

All modes use the same DSP engine. The mode changes who chooses the settings and how tightly the interface constrains them.

### Automatic mode

Automatic mode evaluates candidate presets, ranks them against the current measurements, refines the winner, and applies it before export. It is the recommended starting point in packaged releases.

Choose an AUTO goal:

- **balanced** — general-purpose default
- **room-safe** — more conservative for difficult measurements
- **low-ripple** — prioritizes smoother low-frequency response around modes
- **prefer bass** — gives bass target matching more weight
- **subwoofers** — focuses analysis and leveling on subwoofer work

The scores compare candidates from the same measurement set. Do not use them to compare unrelated speakers or rooms.

Automatic mode manages many Advanced settings. Fields that remain enabled are deliberate expert overrides. The final summary reports the effective values used, including any winner refinement.

### Basic

Basic provides manual control with conservative defaults and hard safety clamps. Use it for source builds, learning the controls, or making a repeatable manual reference filter.

### Advanced

Advanced exposes more correction, phase, timing, and protection controls. It is intended for measured experiments where you understand the effect of each change. Safety-critical numerical guards remain active.

Selecting a mode does not rewrite every visible value. Use **Apply mode defaults** when you want to reset the controls to that mode's baseline.

## 4. Choose the target and filter type

### Target strategies in Automatic mode

- **Adaptive: derive target from room acoustics** (default) starts from Harman6 and makes small, bounded low-frequency changes from stereo measurement evidence. Reliable RT60 data can permit additional bass lift, but never creates it by itself.
- **Auto: search best built-in** evaluates several built-in curves and selects the best-ranked result. A good alternative when measurement metadata is limited or the sources are external.
- **Use selected target curve from Target page** uses the curve you chose or imported and skips automatic target selection.

In Basic and Advanced modes, select the target directly on **Target**. A mild house curve is usually a safer starting point than a perfectly flat in-room target.

### Filter types

| Filter type | Use it when | Main tradeoff |
|---|---|---|
| **Asymmetric** | You want the recommended general-purpose result. | Balances latency, phase correction, and pre-ringing containment. |
| **Minimum Phase** | Low latency and causal behavior matter most. | Does not provide linear-phase correction. |
| **Mixed Phase** | You want bounded excess-phase correction in a selected band. | Needs careful phase limits and verification. |
| **Linear Phase** | Linear-phase behavior matters more than latency. | Long filters can add latency and energy before the main impulse. |

## 5. Set practical limits

For a first run, leave mode defaults in place. These controls matter most when diagnosing or comparing results.

### Sample rate and taps

Match the sample rate to the playback pipeline. More taps give better low-frequency and time resolution but increase processing cost and latency. Multi-rate export scales the tap count for each sample-rate family.

### Correction range

Use the lower and upper correction limits to keep correction inside the measured, trustworthy band. Broad full-range correction requires better measurement quality than bass-focused correction.

### Maximum boost and cut

Maximum boost is a ceiling, not a target. Deep cancellation dips often change with microphone position and should not be filled aggressively. Cuts are usually more reliable for reducing peaks.

Maximum cut controls the strongest permitted attenuation. Leave enough headroom for the target and any boost that remains.

### Local correction threshold

**Ignore local correction below (dB)** removes narrow correction details smaller than the threshold while keeping the broad response shape. The range is `0–3 dB`; `0` keeps all local detail and larger values produce a smoother filter.

### HPF and low-bass protection

A high-pass filter (HPF) protects the system below its useful range. In Automatic mode, enabling HPF lets the search choose its frequency and slope. Low-bass boost lock and Excursion Protection can block risky boost while still allowing cuts.

### Level matching and Stereo Link

Level matching establishes the reference between measurement and target. Stereo Link keeps left and right decisions coherent without forcing identical correction where the channels differ.

## 6. Advanced correction features

### Temporal Decay Control

Temporal Decay Control (TDC) reduces supported low-frequency energy that decays too slowly. It is separate from ordinary magnitude matching. Use it only where measurement evidence indicates a decay problem, and keep strength and maximum reduction bounded.

### Adaptive Frequency-Domain Windowing

Adaptive Frequency-Domain Windowing (A-FDW) changes analysis resolution with frequency and acoustic confidence. It helps the filter follow broad, repeatable behavior instead of narrow reflection detail.

### Phase correction

Excess-phase strength controls how much supported phase error is corrected. **Full phase correction up to** and **Phase correction faded out by** bound the correction band. The group-delay gradient limit contains abrupt timing changes.

### Hybrid IIR + FIR

Hybrid IIR can move supported narrow bass cuts into CamillaDSP peaking filters and let FIR handle the remaining broadband magnitude and phase correction. The exported CamillaDSP configuration must include both IIR and FIR stages; loading only the WAV omits the transferred cuts.

### Bass Integration

Bass Integration uses separate main and subwoofer impulse measurements to create a phase-aligned shared sub branch. With two subwoofers, DecayCore combines their measured responses and exports one shared mono Sub FIR. Apply the reported crossover, delay, polarity, and gain settings, then verify them in the room.

### IR export windowing

Impulse-response export windowing controls how the final FIR is placed and shaped in time. It does not change the target or the correction policy used to design the filter. **auto** is the normal choice; **rew_asym** provides a REW-style asymmetric window.

### XO Phase Model

The crossover phase model describes expected crossover rotation so phase correction does not mistake it for an unrelated loudspeaker error. Use it only when the crossover topology and slopes are known.

## 7. Generate and read the result

Open **START / Results** and press **START**. During an Automatic run, DecayCore may evaluate targets, search preset candidates, refine finalists, and validate the winner. Repeated runs with the same measurements and relevant settings may reuse compatible cache data.

Review these items before export:

- **System Health:** resolve every `CRIT`; inspect each `WARN`.
- **Magnitude:** confirm the broad target match improved without implausible boost.
- **Phase and Group Delay:** look for contained, smooth timing behavior.
- **Filter:** check the actual correction demand and headroom.
- **Impulse:** inspect energy around the main peak, especially with linear or mixed-phase filters.
- **Summary:** confirm mode, target, effective limits, warnings, and Automatic winner details.

Automatic ranking is a decision aid, not an absolute sound-quality score.

## 8. Export and deploy

The result ZIP normally includes:

- Left and Right FIR WAV files in the selected output format
- bypass FIR files for level- and timing-matched comparison
- `Summary_...txt`
- CamillaDSP configuration files
- Sub FIR and crossover settings when Bass Integration is used

The default output location is `Documents/DecayCore/filters/<version>/`. If it is not writable, the results page reports the fallback location.

### CamillaDSP

Use the generated YAML when possible, or add each WAV to the correct convolution channel. Hybrid IIR requires the exported peaking filters as well as the FIR stage. Confirm sample rate, channel routing, and gain.

### Roon

Load the compatible ZIP or WAV set in **DSP Engine → Convolution**. Ensure the package contains the rate used by playback.

### Equalizer APO

Load each WAV through Convolution, verify channel assignment, and add enough preamp reduction to prevent clipping.

## 9. Verify safely

1. Start playback at reduced level.
2. Confirm Left, Right, and any Sub output are routed correctly.
3. Listen for missing bass, excess brightness, unstable imaging, or obvious timing problems.
4. Measure again with correction active.
5. Compare the verification measurement and exported summary with the uncorrected baseline.

Change one major setting at a time so the result remains explainable.

## 10. Troubleshooting

### The run will not start

Check that Left and Right files exist, are readable, and match the selected workflow. Resolve `CRIT` health checks. If the app still behaves unexpectedly, use **About DecayCore → Maintenance** to clear Automatic-mode caches or reset settings.

### The result sounds thin or bass is weak

Check target selection, HPF, Low-bass boost lock, Excursion Protection, TDC strength, channel routing, and subwoofer polarity. Make sure the convolver has not loaded a bypass or wrong-rate filter.

### The result sounds bright or aggressive

Lower the upper correction limit, choose a gentler target, increase smoothing, or compare with Basic mode defaults. Do not compensate by adding broad gain without checking headroom.

### Bass still rings

Confirm that the measurement is reliable and TDC is active where needed. A persistent room mode may need speaker or listening-position changes, acoustic treatment, or a carefully verified Hybrid IIR cut.

### Stereo image is unstable

Confirm consistent Left and Right measurements, correct file assignment, Stereo Link settings, and alignment. Re-measure if either channel was captured from a different microphone position or routing state.

### Many measurement takes are rejected

Check clipping, noise, cables, device selection, sweep level, and microphone movement. Fix the measurement before weakening quality checks.

## 11. Essential terms

- **FIR:** a finite impulse response filter used by a convolution engine.
- **IR:** an impulse response containing timing, phase, and magnitude information.
- **Target curve:** the response shape the correction aims toward.
- **Headroom:** unused digital level reserved to prevent clipping.
- **Excess phase:** phase behavior left after ordinary propagation delay and the chosen baseline are removed.
- **Group delay:** frequency-dependent delay, used to inspect timing behavior.
- **RT60:** an estimate of how long sound energy takes to decay by 60 dB.
- **Taps:** the number of samples in the FIR; more taps increase time span and processing cost.

The best result is usually a conservative, repeatable filter that survives a verification measurement—not the flattest generated graph.
