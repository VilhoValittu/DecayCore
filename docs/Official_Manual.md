# DecayCore - Official Manual (v1.0.7)

## 1. Overview
DecayCore generates **FIR room-correction filters** from built-in sweep measurements, compatible external measurement imports, and WAV/IR captures.
It prioritizes **time-domain correctness** before frequency-domain equalization.

DecayCore explicitly separates:
- **Propagation delay (Time-of-Flight / TOF)** → removed before phase analysis
- **Excess phase distortion** → handled by FIR phase reconstruction (Linear / Minimum / Mixed / Asymmetric)

## 2A. Detailed DSP Signal Flow

The diagram below describes the internal signal-processing architecture
at a more technical level than the simplified pipeline overview.

```
                ┌──────────────────────────────┐
                │ DecayCore / external input   │
                │   (Magnitude + Phase)        │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Robust Parsing & Normalization │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Time-of-Flight Detection    │
                │  & Phase Reference Alignment │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Confidence & Reflection     │
                │  Analysis (GD, slope, etc.)  │
                └──────────────┬───────────────┘
                               │
                ├──────────────┴───────────────┐
                ▼                              ▼
     ┌────────────────────┐        ┌────────────────────────┐
     │ Magnitude Path     │        │ Phase Path             │
     └─────────┬──────────┘        └──────────┬─────────────┘
               │                                │
               ▼                                ▼
   ┌──────────────────────────┐     ┌────────────────────────────┐
   │ Target Construction      │     │ Phase Mode Selection       │
   │ (house curve, tilt, XO)  │     │ (Linear/Min/Mixed/Asym)    │
   └─────────┬────────────────┘     └──────────┬─────────────────┘
             │                                  │
             ▼                                  ▼
   ┌──────────────────────────┐     ┌────────────────────────────┐
   │ Hybrid IIR               │     │ Excess-Phase Reconstruction │
   │ Preconditioning          │     │ & Min-Phase Separation      │
   │ (optional; modal IIR     │     └──────────┬─────────────────┘
   │  cuts subtracted from    │                │
   │  FIR gain curve)         │                ▼
   └─────────┬────────────────┘     ┌────────────────────────────┐
             │                      │ Conditional GD Stabilization│
             ▼                      │ (Bass-focused, soft-limited)│
   ┌──────────────────────────┐     └──────────┬─────────────────┘
   │ Level Matching           │                │
   │ (Smart / Manual)         │                ▼
   └─────────┬────────────────┘     ┌────────────────────────────┐
             │                      │ Phase Safety Clamp (±45°)  │
             ▼                      └──────────┬─────────────────┘
   ┌──────────────────────────┐                │
   │ Magnitude Correction     │                │
   │ - Boost/Cut limits       │                │
   │ - Slope limits           │                │
   │ - Confidence Pull        │                │
   │ - A-FDW                  │                │
   └─────────┬────────────────┘                │
             │                                 │
             └──────────────┬──────────────────┘
                            ▼
                ┌──────────────────────────────┐
                │ Optional Temporal Decay      │
                │ Control (TDC)                │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │ FIR Synthesis (IFFT)        │
                │ + Normalization              │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │ Multi-rate Export            │
                │ + Optional Hybrid IIR        │
                │   biquad export (CamillaDSP) │
                └──────────────────────────────┘
```

### Architectural principles

- Magnitude-domain safety (boost/cut/slope/confidence) prevents
  physically unsafe or measurement-driven overcorrection.

- Phase-domain reconstruction is explicitly separated from magnitude logic.

- Group-delay stabilization operates only as a **conditional spike guard**
  and does not act as a wideband phase shaper.

- Temporal Decay Control (TDC) modifies time-domain energy storage
  independently from steady-state magnitude equalization.

- Optional Hybrid IIR preconditioning subtracts narrow IIR biquad cuts
  from the FIR gain curve before magnitude correction, allowing the FIR
  to focus on the remaining broadband response.


---

## 2b. Processing pipeline (high level)
1. Import REW magnitude + phase
2. Robust parsing and unit normalization
3. Optional smoothing (Standard / Psychoacoustic / Adaptive FDW)
4. TOF detection & removal
5. Confidence analysis & reflection detection
6. Target curve construction (built-in / adaptive / user-selected)
7. Optional Hybrid IIR preconditioning (modal biquad cuts subtracted from FIR gain curve)
8. Level matching (Smart Scan or Manual window)
9. Magnitude correction with safety guards
10. Phase reconstruction (Linear / Minimum / Mixed / Asymmetric)
11. Optional TDC (decay control)
12. FIR synthesis, optional normalization
13. Multi-rate export + optional Hybrid IIR biquad export (CamillaDSP YAML)

---

## 3. Installation

## Download

- Windows: [latest DecayCore release](https://github.com/VilhoValittu/DecayCore/releases/latest)
- macOS (Intel + Apple Silicon): [latest DecayCore release](https://github.com/VilhoValittu/DecayCore/releases/latest)
    -macOS builds are community-supported. Limited direct testing.
- Linux: [latest DecayCore release](https://github.com/VilhoValittu/DecayCore/releases/latest)
- All releases: [DecayCore releases](https://github.com/VilhoValittu/DecayCore/releases)

---

## 4. Input data and measurement workflows

The recommended path is the built-in measurement tool. REW exports and WAV/IR files are also supported as import sources.

### 4.1 Built-in measurement tool

Measure directly inside DecayCore from the Measurement tab. The tool saves IR WAV files that can be loaded immediately from the Files tab.

Tips:
- Use a calibrated measurement microphone.
- Measure Left and Right channels separately with the same microphone position and procedure.
- Keep all session files at the same gain — do not normalize between channels.

### 4.2 REW text export (`.txt`)

DecayCore expects text exports with columns:

- Frequency (Hz)
- Magnitude (dB)
- Phase (deg)

Headers are optional. Comment lines starting with `*`, `#`, or `;` are ignored.

Tips:
- Export both Left and Right separately from REW.
- Use a consistent time reference in REW (same measurement procedure per channel).
- Phase data is required for full phase-aware correction behavior.

### 4.3 WAV impulse export (`.wav`)

DecayCore can import impulse-response WAV files and convert them internally for processing.

Use this when:

- you saved IR files from the built-in measurement tool
- you prefer an IR-based workflow from REW
- you want DecayCore to derive the response using its own windowing path

Practical guidance:

- use mono impulse files for each channel
- keep the export settings consistent between left and right
- match the sample rate to the playback rate when practical

---

## 5. Core controls and what they do

### 5.1 Base sample rate and taps
- **Base Sample Rate (fs):** the sample rate used for FIR design.
- **Taps:** FIR length. Higher taps → better low-frequency resolution but more latency.

**Multi-rate generation:** exports multiple sample rates (44.1/48/88.2/96/176.4/192 kHz).

**Auto-taps mapping (multi-rate):** keeps FIR time-length roughly constant across sample rates using a 44.1 kHz reference.

### 5.2 Filter type
- **Linear Phase:** best timing precision, can create audible pre-ringing at high frequencies.
- **Minimum Phase:** no pre-ringing; magnitude correction only, phase derived via minimum-phase reconstruction.
- **Mixed Phase:** linear phase below a split frequency, minimum phase above.
- **Asymmetric Linear:** linear phase, but with an asymmetric time window to suppress audible pre-ringing while preserving the leading edge.

#### Which filter type should I use?

DecayCore automatic mode was tested with identical measurements and target curve using four filter types.

Selection is based on **Best rank score**, which evaluates:

- target match
- DSP artifacts (ripple, GD gradient, phase limits)
- headroom / boost safety
- acoustic events
- stereo consistency (L/R delta)
- harmonic-aware boost penalties in nonlinear bands
- IACC-aware stereo-width protection

##### Results

| Filter type | Best rank score |
|---|---|
| **Asymmetric** | **90.46** |
| Linear | 90.23 |
| Mixed | 88.27 |
| Minimum | 78.91 |

##### Recommendation

**Most users should choose: Asymmetric**

It provides:

- near-linear phase accuracy
- excellent target matching
- minimal DSP artifacts
- practical latency

##### Alternative choices

**Linear phase**

Use if maximum phase linearity is required and latency is not an issue.

**Mixed phase**

Good low-latency option with robust behaviour, but slightly higher DSP penalty.

**Minimum phase**

Generally not recommended for automatic mode results due to higher DSP penalties.

#### Why Asymmetric Filters Exist

Traditional FIR room-correction filters typically fall into two categories:

| Type | Strength | Limitation |
|---|---|---|
| **Linear phase** | Perfect phase symmetry and very accurate correction | Very high latency |
| **Minimum / Mixed phase** | Low latency and practical for real-time use | Limited phase correction |

In real listening systems this creates an unavoidable trade-off:

- **Linear phase filters** can achieve extremely accurate correction, but often introduce **hundreds of milliseconds of latency**.
- **Minimum or mixed-phase filters** are practical for playback but cannot fully correct phase behaviour.

##### The idea behind asymmetric filters

DecayCore introduces **asymmetric FIR filters** to bridge this gap.

Instead of forcing the impulse response to be perfectly symmetric (linear phase) or fully causal (minimum phase), the filter is designed so that **most of the energy occurs after the main impulse while allowing controlled asymmetry**.

This enables:

- near-linear correction accuracy
- practical latency
- reduced pre-ringing artifacts
- stable stereo alignment

##### Impulse response comparison

```text
Linear phase (symmetric)
<------ pre ------|------ post ------>
                  ^
                main impulse


Mixed / minimum phase
                  ^
                main impulse
                  |------------>


Asymmetric (DecayCore)
               ^
             main impulse
               |---------------------->
```

**Linear phase** filters distribute energy symmetrically around the impulse, which increases latency.

**Mixed/minimum phase** filters place all energy after the impulse, reducing latency but limiting correction accuracy.

**Asymmetric filters** intentionally place **most energy after the impulse while keeping controlled asymmetry**, allowing strong correction with significantly lower latency than fully linear filters.

---

#### Real automatic-mode comparison

Using identical measurements and the same target curve, DecayCore automatic mode produced the following **Best rank score** results:

| Filter type | Best rank score |
|---|---|
| **Asymmetric** | **90.46** |
| Linear | 90.23 |
| Mixed | 88.27 |
| Minimum | 78.91 |

The **rank score** evaluates multiple factors simultaneously:

- target matching accuracy
- DSP artifacts (ripple, GD gradient, phase limits)
- headroom safety and boost limits
- acoustic event penalties
- stereo consistency (L/R delta)
- harmonic-aware boost penalties in nonlinear bands
- IACC-aware stereo-width protection

In this comparison, **Asymmetric filters achieved the highest score**, combining strong correction accuracy with minimal DSP penalties.

---

#### When to use asymmetric filters

For most systems, **asymmetric filters are the recommended default** because they provide the best balance between:

- correction accuracy
- DSP stability
- latency

Other filter types still have their place:

| Filter type | Recommended when |
|---|---|
| **Asymmetric** | Best overall balance (recommended default) |
| **Linear phase** | Maximum phase accuracy and latency is irrelevant |
| **Mixed phase** | Low-latency real-time playback |
| **Minimum phase** | Compatibility or special DSP workflows |

##### In short

Asymmetric filters exist because **room correction should not require choosing between accuracy and usability**.

They allow DecayCore to deliver **high-quality correction while remaining practical for real listening systems**.

#### Asymmetric Linear

Asymmetric Linear is a **low-latency linear-phase mode** that reduces audible pre-ringing
by shifting the impulse peak earlier in time.

The **Left window (ms)** parameter defines the **latency target**:
Only **Auto** and **Asymmetric** windowing modes are available.
Legacy **Symmetric** and **Off** modes have been removed to simplify the UI
and focus on the most effective REW-based strategies.

**Practical guidance:**
- **85 ms (default):** best balance between low latency and stable bass correction
- **80–150 ms:** safe operating range
- **< 50 ms:** extreme low-latency mode, expert use only

##### Automatic safety behavior (important)

To prevent unstable bass behavior at very low latency, DecayCore applies
automatic safeguards in REW Asymmetric mode:

- When **Left < 15 ms**
  → bass-first (A-FDW confidence shaping) is automatically limited to low frequencies
- When **Left < 10 ms**
  → low-frequency **boosts are disabled** (cuts are still allowed)

These safeguards do **not** reduce correction quality at mid and high frequencies,
but prevent excessive ripple and instability in the bass region.


### 5.3 Smoothing for plots/filters
- **Standard smoothing (1/6, 1/12 etc):** classic fractional-octave smoothing.
- **DecayCore Reference:** heavier smoothing where the ear is less sensitive (useful for robust targets).

- Filter smoothing is not always easy to read in DecayCore graphs due to limited plot space.

### 5.4 Safety limits (highly recommended)
- **Max boost (dB):** hard safety ceiling for positive gain.
- **Max cut (dB):** maximum allowed attenuation depth.
- **Max slope (dB/oct):** limits how fast correction can change over frequency.
- **Independent slope limits for boost/cut:** optional, prevents small boosts from being flattened while keeping cuts controlled.
- **Excursion protection:** blocks bass boost below a chosen frequency.
- **HPF (subsonic):** protects woofers from ultra-low content.

**HPF behavior (important):**
- HPF is applied as a **true magnitude high-pass filter** in the FIR path.
- The HPF response is added directly to the correction curve
  (equivalent to applying a Butterworth HPF to the final FIR magnitude).
- This ensures **magnitude and phase consistency**.
- Prevents double-HPF behavior, incorrect low-frequency response,
  and artificial group-delay artifacts.

### 5.5 Level matching
DecayCore aligns measurement and target levels before synthesizing the filter.

Modes:
- **Smart Scan (Automatic Optimization):** searches for a stable frequency window where measurement follows target shape best, then computes offset using Median or Average.
- **Manual Window:** you choose the lower/upper frequency limits and the target level.

Recommended:
- Use **Median** for room measurements (immune to narrow peaks/dips).
- Use **Average** mainly for nearfield or very smooth data.

### 5.6 Hybrid IIR (FIR + IIR bass preconditioning)

Hybrid IIR is an optional bass preconditioning stage that combines narrow IIR Peaking EQ biquad cuts with the standard FIR pipeline.

#### What it does

When enabled, DecayCore:

1. Detects narrow room modes in the configured bass frequency range using confidence and group-delay excess thresholds.
2. Designs conservative Peaking EQ biquad cuts for confirmed modal peaks (cuts only — no boosts ever).
3. Subtracts the IIR biquad magnitude response from the FIR target gain curve before magnitude correction. The FIR corrects only the remaining residual response.
4. Exports the IIR biquad parameters into the CamillaDSP YAML configuration alongside the FIR convolver.

This keeps the two filter types clearly separated in responsibility: IIR for precision narrow-band cuts, FIR for everything else.

#### When to use

Hybrid IIR is appropriate when:

- narrow, high-Q room modes in the bass persist despite FIR correction
- measurement confidence and group delay excess confirm that those peaks are reliable
- the deployment target is CamillaDSP and an IIR biquad filter stage can be inserted in the pipeline

Leave it disabled when:

- measurements are noisy or low-confidence in the bass region
- the room does not show clear narrow modal peaks
- the deployment target cannot run IIR biquads alongside the FIR convolver

#### CamillaDSP deployment

When hybrid IIR produces biquads, they are included in the exported CamillaDSP YAML as Peaking EQ filter entries. **Both the IIR biquad stage and the FIR convolver must be active in the pipeline.** Loading only the FIR WAV without the IIR biquads will leave the bass correction incomplete because the FIR target was designed with the IIR contribution already subtracted.

#### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Enables hybrid IIR preconditioning |
| `max_filters_per_channel` | `3` | Maximum IIR biquad cuts per channel |
| `min_freq_hz` | `20 Hz` | Lower bound of modal detection range |
| `max_freq_hz` | `150 Hz` | Upper bound of modal detection range |
| `min_peak_db` | `4.0 dB` | Minimum peak height required to place a cut |
| `min_q` | `3.0` | Minimum allowed biquad Q |
| `max_q` | `12.0` | Maximum allowed biquad Q |
| `max_cut_db` | `6.0 dB` | Maximum cut depth per biquad |
| `min_confidence` | `0.65` | Minimum confidence required at the mode frequency |
| `min_gd_excess_ms` | `15.0 ms` | Minimum group delay excess required |
| `min_cut_priority` | `0.0` | Minimum cut priority score to place a filter |
| `max_voice_clarity_risk` | `0.45` | Limits cuts that could reduce voice clarity |

Controls are in the Advanced tab under a collapsible hybrid IIR tuning section.

See also: [Hybrid IIR + FIR Room Correction](hybrid-iir-fir.html)

---

### 5.7 Adaptive Target

In AUTO mode, DecayCore supports three target selection strategies:

- **Auto: search best built-in** (default) — evaluates multiple built-in target curves in parallel and picks the best-ranked result.
- **Adaptive: derive target from room acoustics** — synthesizes a custom Harman6-based target from the measured room's bass buildup, tilt, and RT60 characteristics. Skips the multi-curve search entirely.
- **Use selected target curve from Target page** — uses the target curve manually selected in the Target tab.

#### How adaptive target synthesis works

1. Starts from a Harman6-style reference shape.
2. Estimates the room's natural bass buildup and overall tilt from the measured response.
3. Adjusts bass and tilt compensation fractions based on those estimates.
4. When RT60 data is available, further refines compensation using measured decay times across bass (20–125 Hz), mid (400–2000 Hz), and treble (2000–8000 Hz) bands. The RT60 adjustment is bounded to ±2 dB.

#### RT60 requirement

RT60 data is automatically available when measurements are produced by DecayCore's built-in measurement tool. With external REW exports or WAV impulse files, RT60 data is typically absent — the RT60 refinement step is skipped and the target is derived from bass buildup and tilt only.

If RT60 data is not available, `Auto: search best built-in` is generally the safer strategy.

See also: [Adaptive Target](adaptive-target.html)

---

## 6. Temporal Decay Control (TDC)
TDC is **not EQ**. It targets resonant energy storage (ringing) rather than steady-state amplitude.

Controls:
- **TDC Strength (0–100%)**: how strongly decay is shortened.
- **TDC Max Reduction (dB)**: hard cap for the total reduction applied per frequency bin.
- **TDC Slope Limit (dB/oct)**: optional smoothing of the TDC reduction curve (predictable, avoids narrow notches).

When to enable:
- Room modes dominate the bass (slow decay, boomy notes).

When to reduce or disable:
- Very dry rooms or nearfield measurements where decay is already short.

---

## 7. Outputs
Typical output package contains:
- FIR filters (`.wav` 32-bit float)
- Bypass FIR WAV files (identity impulse at the same peak position as the correction filter, for A/B comparison without reloading a different config)
- Bypass config file (`.yml`) for easy A/B switching between corrected and bypassed signal
- Summary report (`Summary.txt`)
- Config snippet (`.cfg`)
- CamillaDSP YAML (`.yml`) — includes Hybrid IIR Peaking EQ biquad blocks when Hybrid IIR is enabled and produces biquads
- Optional dashboard plots (PNG, depending on export/performance mode)

The Summary report typically includes:
- correction range, smoothing, FDW/A-FDW info
- max boost/cut/slope limits applied
- RT60 estimate and confidence summary
- match score and (optionally) comparison-mode grid info

### Output directory
All generated filter packages (`.zip`) are written by default to
`Documents/DecayCore/filters/<version>/` on all platforms.
If that location is not writable, DecayCore automatically falls back to a safe writable directory.
The active export path is shown in the Results view.

### IR export windowing vs DSP correction

IR windowing applied during FIR export is intentionally separated from the
actual DSP correction logic.
This distinction is important for understanding why FIR files may differ
in time-domain appearance without changing the audible correction.

See:
- [IR Export Windowing vs DSP Correction](IR_Export_Windowing.html)

---

## 8. MiniDSP / limited-taps workflow (practical)
Many MiniDSP devices have limited FIR taps per channel.
A reliable approach is:

1. Use IIR/PEQ on subs (and delay) to get subs reasonably flat and aligned.
2. Measure mains alone.
3. Generate a DecayCore filter for mains.
4. Keep correction minimum frequency above the sub crossover (example: 80 Hz).
5. If an IIR crossover exists on the device, DecayCore can “unwrap” the crossover phase in the measurement with FIR.
6. Finally align subs/mains timing with delay around the crossover point.

---

## 9. Troubleshooting

### Too aggressive treble
- Use heavier smoothing.
- Lower max slope.
- Limit correction max frequency.

### Bass boost feels unsafe
- Set excursion protection frequency.
- Enable HPF.
- Reduce max boost.

### HPF does not seem to affect bass
- Verify that HPF is enabled and frequency/order are non-zero.
- Check the filter magnitude plot: a proper roll-off should be visible below HPF frequency.
- HPF is applied in the FIR magnitude path, not by disabling correction below cutoff.

---

## 10. DSP Design Rationale

DecayCore is built around a small set of explicit design principles.
This section summarizes the reasoning behind the architecture.

### 10.1 Separation of physical phenomena

Room measurements contain multiple independent effects:

1. Propagation delay (Time-of-Flight)
2. Loudspeaker minimum-phase behavior
3. Excess phase distortion
4. Room-induced modal energy storage

Treating these as a single “EQ problem” leads to overcorrection
and unstable filters.

DecayCore separates these domains explicitly:

- TOF is removed before phase analysis.
- Minimum-phase and excess-phase components are handled separately.
- Room decay is treated in the time domain (TDC), not as static amplitude EQ.

This separation reduces unintended cross-coupling between magnitude,
phase, and decay shaping.

---

### 10.2 Confidence-weighted correction

Measured data is not equally reliable across frequency.
Reflection density, windowing, and signal-to-noise ratio
all influence trustworthiness.

Instead of applying uniform correction strength,
DecayCore uses confidence-aware logic:

- Adaptive FDW (A-FDW)
- Confidence Pull
- Bass-first masking logic

Low-confidence regions are smoothed or gently pulled toward a safe target,
preventing aggressive corrections driven by measurement artefacts.

---

### 10.3 Phase reconstruction philosophy

Phase correction is applied to excess-phase only.
Loudspeaker minimum-phase and theoretical crossover phase
are preserved unless explicitly modified.

Additional safeguards:

- Phase correction clamp (±45°)
- Conditional group-delay gradient stabilization

The GD stabilization stage is intentionally limited:
it acts only as a spike guard and does not reshape
wideband phase trends.

The objective is transient integrity,
not visual flatness of group delay.

---

### 10.4 Time-domain priority

Many correction systems optimize magnitude first
and treat time-domain behaviour as secondary.

DecayCore reverses this priority:

- TOF is corrected before phase modelling.
- Phase reconstruction precedes decay shaping.
- Temporal Decay Control modifies energy storage directly.

This ordering minimizes pre-ringing,
reduces modal ringing,
and preserves leading-edge clarity.

---

### 10.5 Determinism and reproducibility

Given identical inputs and configuration,
DecayCore produces deterministic outputs.

Safety limits and internal clamps are:

- explicitly documented,
- reported in Summary.txt,
- and visible in DSP info.

The system avoids hidden heuristics that alter behaviour silently.

The result is a correction workflow that is
transparent, repeatable, and technically defensible.

---

### 10.6 Group-delay gradient limiter (mathematical definition)

DecayCore includes an optional group-delay (GD) gradient limiter used as a **conditional spike guard**
to prevent artificial phase “kinks” (typically from unwrap/interpolation artefacts)
without reshaping wideband phase trends.

In version 3.0.1 and later, the limiter is:

- **Bass-focused (20–250 Hz)**
- **Soft-limited (tanh)**
- **Conditionally enabled**

It acts strictly as a spike guard, not as a wideband phase shaper.

**Group delay from phase**

Let the unwrapped phase be \( \phi(f) \) in radians, frequency \( f \) in Hz.
Group delay in seconds:

\[
\tau_g(f) = -\frac{1}{2\pi}\frac{d\phi(f)}{df}
\]

In milliseconds:

\[
\mathrm{GD}_{ms}(f) = 1000 \cdot \tau_g(f)
                 = -\frac{1000}{2\pi}\frac{d\phi(f)}{df}
\]

**Gradient per octave**

The limiter operates on the GD slope with respect to the log-frequency axis (octaves):

\[
g(f) = \frac{d\,\mathrm{GD}_{ms}(f)}{d(\log_2 f)}
\quad [\mathrm{ms}/\mathrm{oct}]
\]

**Soft limiting**

Instead of hard clipping, a soft limiter is used to preserve natural trends while compressing extremes:

\[
g_{lim}(f) = L \cdot \tanh\!\Big(\frac{g(f)}{L}\Big)
\]

where \(L\) is the configured limit in \(\mathrm{ms}/\mathrm{oct}\) (e.g. 30 ms/oct when enabled).

**Reconstruction**

The limited GD curve is reconstructed by integrating \(g_{lim}(f)\) over \(\log_2 f\),
anchored at the band center for stability. The limited phase is then obtained by integrating:

\[
\frac{d\phi(f)}{df} = -2\pi \frac{\mathrm{GD}_{ms}(f)}{1000}
\]

In practice:

- The limiter operates only within the **bass-focused band (20–250 Hz)**.
- It is **conditionally enabled** (e.g. bypassed when A-FDW and Bass-first
  stabilization are active, except in high-risk windowing modes).
- The soft-limiting function ensures continuity and avoids sharp clipping artefacts.

This guarantees that group-delay stabilization does not reduce transient
liveliness or alter broadband phase behaviour

---

### 10.7 FIR length vs time / frequency resolution (practical tradeoff)

FIR design always trades time-domain behaviour against frequency-domain resolution.
For a filter with \(N\) taps at sample rate \(f_s\):

- **Time length** (impulse duration):
  \[
  T \approx \frac{N}{f_s}
  \]

- **Frequency-bin spacing / resolution** (typical FFT grid intuition):
  \[
  \Delta f \approx \frac{f_s}{N}
  \]

- **Linear-phase latency** (group delay of a symmetric FIR):
  \[
  \tau \approx \frac{N-1}{2f_s}
  \]

Implications:

- More taps (higher \(N\)) improve low-frequency precision and reduce ripple sensitivity,
  but increase latency and can make time-domain constraints (e.g. low-latency asymmetric exports)
  harder to satisfy.

- Higher sample rate (higher \(f_s\)) reduces time length and latency for the same \(N\),
  but also increases \(\Delta f\). This is why multi-rate export commonly scales taps to keep
  the **time length** approximately constant across sample rates.

Practical guidance:

- Use more taps when you need finer low-frequency control (room modes / long decay).
- Use shorter time length when low latency is required (live monitoring / AV sync),
  accepting reduced LF resolution and relying more on conservative phase behaviour and safety guards.

---

## 11. Built-in Measurement

DecayCore includes an integrated measurement tool that plays a sine sweep, records the response, and produces IR WAV files ready to import directly into the filter-generation workflow. It can replace a separate REW session when the hardware setup allows it.

### 11.1 What it produces

- Per-channel IR WAV files for Left, Right, Sub1, and Sub2
- Spatial averaging across multiple listening positions (magnitudes averaged; primary-position phase preserved)
- Repeat-based outlier rejection per channel

### 11.2 Configuration

Open the measurement dialog to set:

- **Positions** (1–12): number of listening positions in the room
- **Repeats per channel** (1–12): sweeps captured per channel per position; default is 5
- **Primary position**: which position supplies the reference phase and timing for spatial averaging
- **Channel selection**: checkboxes for Left, Right, Sub1, Sub2
- **Outlier rejection**: enable/disable and strictness level (safe / normal / strict)
- Per-channel settings: audio output and input device, channel index, sample rate, sweep frequency range and length, output gain, optional microphone calibration file

### 11.3 Measurement steps

1. Configure the session and start
2. For each position the tool captures all selected channels in sequence
3. After each position it pauses and prompts you to move the microphone to the next position
4. If Sub1 is selected it pauses before the first sub sweep so you can turn the subwoofer on
5. If Sub2 is also selected it pauses again after Sub1 completes so you can switch subwoofers
6. After all positions are captured it aggregates the results and saves the IR files
7. A summary shows how many takes were kept and rejected per channel

### 11.4 Subwoofer measurement — Windows requirement

> **Subwoofer measurement is only guaranteed to work on Windows.**
>
> The sweep is sent to the LFE channel (channel index 3 in a 5.1 or 7.1 layout). On Windows the output device must be configured for **5.1 or 7.1 multichannel playback** in Windows Sound settings. Without multichannel mode active the LFE channel does not exist at the driver level and the subwoofer will not receive the sweep signal.
>
> To enable multichannel output: open Sound settings → select the playback device → Properties → Advanced or Spatial sound → set the format to 5.1 or 7.1.
>
> On other platforms subwoofer routing may work in some configurations but is not tested or supported.

### 11.5 Outlier rejection

Takes are assessed per channel after all repeats for a position are captured.

- **Hard failures** (clipping, excessive noise, severe peak-timing deviation) are always rejected regardless of strictness
- **Strictness levels** set the median-absolute-deviation threshold for softer outliers:
  - Safe: 3.5 σ — keeps most takes unless they are clearly wrong
  - Normal: 2.75 σ — balanced default
  - Strict: 2.0 σ — discards anything that deviates meaningfully from the group median
- The post-session summary reports how many takes were kept per channel

If many takes are rejected, check the signal level, cable connections, and microphone placement before tightening the strictness setting.

### 11.6 Using the results

- Load the saved IR WAV files via the **WAV impulse export (.wav)** path (section 4.3)
- Do not normalize individual IR files and do not move them to separate `t=0` references — keep all files from the same session at the same gain so that relative timing and level are preserved

---

## 12. Bass Integration (Beta)

> **This feature is currently in Beta.** Behaviour, parameter names, and output formats may change in future releases. Feedback is welcome — please report issues or observations via the GitHub issue tracker or the support channel.

Bass Integration is an optional workflow for systems that include one or two dedicated subwoofers. It produces a phase-aligned mono Sub FIR filter alongside the normal L/R correction filters and recommends crossover, delay, gain, and polarity settings.

### 12.1 Integration path

Bass Integration uses the **Direct DAC / CamillaDSP sub output** path: the subwoofer is driven directly by a separate amplifier channel or a CamillaDSP pipeline output. Main speakers and subwoofer are measured independently and then aligned in phase, delay, gain, and polarity.

### 12.2 Single combined filter for multiple subwoofers

**When two subwoofers (Sub1 + Sub2) are measured, DecayCore generates one shared mono sub filter, not two separate filters.**

The current Direct DAC path first peak-aligns Sub2 relative to Sub1 and then forms one peak-aligned vector-averaged combined sub reference. FIR generation and main/sub alignment are run against that one shared combined sub branch.

The resulting mono sub filter is intended to drive both subwoofers simultaneously from a single output channel (or a summed/bridged output). Reported delay, gain, polarity, and optional allpass values apply to this shared combined sub branch, not to two separate per-sub filters. If DecayCore reports a CamillaDSP main delay, that is a compensating L/R main-channel delay relative to the shared sub branch, not a per-sub setting.

This does not replace per-unit DSP correction if the subwoofers are in very different positions with very different response shapes.

### 12.3 What the feature produces

- Recommended crossover frequency (Main HPF / Sub LPF)
- Recommended delay, gain, and polarity for the sub output
- Dedicated mono Sub FIR WAV file included in the export ZIP
- Diagnostics: overlap ripple, cancellation risk, sub dominance, XO group delay mismatch, phase error

### 12.4 Interpretation of diagnostics

| Metric | Good | Marginal |
|--------|------|----------|
| Overlap ripple | < 8 dB | < 12 dB |
| Sub dominance | < 8 dB | < 12 dB |
| XO group-delay RMS mismatch | < 12 ms | < 20 ms |

If metrics are in the marginal range, the integration may still be usable but the crossover region should be verified with a post-filter measurement before finalizing the setup.

### 12.5 Limitations (Beta)

- Sub FIR generation requires separate sub measurement WAV files from the built-in measurement tool or compatible external captures.
- Subwoofer measurement via the built-in tool requires Windows (see section 11.4).
- The combined mono filter approach is appropriate for symmetrically placed dual-sub setups. Asymmetric placements may benefit from per-unit correction outside of DecayCore first.
- Cancellation risk and overlap ripple are model-based estimates, not measured verification. Always re-measure after applying the exported settings.

### Disclaimer
AI was used to translate this document from Finnish to English.
