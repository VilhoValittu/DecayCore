---
title: DecayCore Changelog
description: Recent user-visible changes to DecayCore measurement, correction, export, and platform support.
hide_page_heading: true
---

# Changelog

This page contains the five latest stable releases. [Older releases and Finnish translations are preserved in the archive]({{ '/changelog/archive/' | relative_url }}).

## [Unreleased]

## [1.4.2] - 24-10-2026

### Deep bass nulls now show clearly in the target curve

The target curve dips where the room has a deep null, so the filter no longer tries to fill it. In a test, a 16 dB dip half an octave wide, typical of a boundary cancellation, left the target unchanged. It now lowers the target by about 8 dB. Broad dips and ordinary ripple are left to the correction. A file exported in SPL now gives the same filter as one in dBFS. Earlier Automatic results are rebuilt.

### CamillaDSP files no longer cut the level by a fixed 6 dB

Without a subwoofer, the file plays 6 dB louder at the same volume, so lower your volume before the first listen. With a subwoofer, it cuts 6.02 dB for the summed channels, plus any gain from the sub trim. Raise Auto Headroom Margin for more clipping headroom.

### Linkwitz-Riley or Butterworth for each crossover

Pick Butterworth or Linkwitz-Riley for each crossover on the XO tab, to match your speakers. Butterworth stays the default.

### Results check your crossovers against the measurement

Results shows how well each crossover matches the measured phase, for example "80Hz: 91% good", and warns when the measurement doesn't contain one. Crossovers too subtle to see in a room read "not verifiable". The check never changes the filter.

### Finer bass resolution in the analysis

Third-octave smoothing no longer flattens narrow room modes. A 9 dB mode with a Q of 30 now reads 8 dB instead of 3 dB. Filters for smooth measurements are unchanged. Earlier Automatic results are rebuilt.

### Implausible room-decay readings no longer shape Bass-first AI or Hybrid IIR

With Bass-first AI on, a measured RT60 that DecayCore already rejects, such as 0.04 s or 8 s, no longer sets the modal limit or narrows the Hybrid IIR band. Earlier Automatic results are rebuilt.

### High-pass filters attenuate as designed in Asymmetric and Mixed Phase

A 20 Hz, 24 dB/oct high-pass gave 12 dB at 10 Hz instead of 23 dB. Results now checks the exported high-pass against the one you set. Earlier Automatic results are rebuilt.

### Automatic mode keeps high-pass and low-pass filters at full depth

A late refinement step no longer limits their stopband to Max Cut. A 40 Hz, 24 dB/oct high-pass with 131,072 taps now attenuates 10 Hz by 42 dB instead of 17 dB. Earlier Automatic results that used this step are rebuilt.

### Low crossovers keep an Asymmetric filter's response

With a crossover at 100 Hz or lower, an Asymmetric filter could bend the bass by 4 dB or more. It now keeps its designed response, as Linear Phase already does. Earlier Automatic results that used a crossover are rebuilt.

## [1.4.1] - 23-10-2026

### Group-delay gradient readout covers the audio band

The Results tab and the text report now measure the excess-phase group-delay gradient from 20 Hz up, so its peak is no longer reported at a few hertz. The filters and Automatic mode ranking are unchanged.

### Faster bass decay check

When a left/right filter lengthens bass decay, DecayCore tries lighter phase correction. Those attempts now reuse the correction already calculated, so a 65,536-tap Asymmetric run at 44.1 kHz finishes about a third sooner. The filters are identical to before.

### Faster, clearer measurement timeout errors

When the audio stream never starts, a measurement now gives up after two seconds instead of waiting out the whole sweep, so a failing device no longer holds the start of the sweep back by about 30 seconds. The error says whether the input device delivered any audio. On macOS, if it delivered none, it points to System Settings > Privacy & Security > Microphone. The log lists the devices that were opened and how many frames were played and recorded.

### macOS launcher asks for microphone access as DecayCore

`Start_Decay.command` now opens the app bundle instead of running it inside Terminal. The microphone prompt and the entry in Privacy & Security name DecayCore, not Terminal.

### Crossovers no longer distort the filter

With a crossover set on the XO tab, Asymmetric and Linear Phase filters could cut the midrange by about 20 dB if dB/oct was over 12. The filter now keeps the frequency response it was designed with, and Automatic mode ranks these runs like any others. Earlier Automatic results that used a crossover are rebuilt. Thanks **Nomid from ASR-forum** to finding this!

### Sample-rate ranking matches Automatic mode

When you export several sample rates, the summary ranks them using the same DSP quality penalty as Automatic mode. Scores can differ from earlier versions.

### Text measurements with a byte-order mark

Text measurement files saved with a byte-order mark now keep their first data
row, and UTF-16 files are read instead of rejected.

### Runs with an empty correction band

If the correction band ends up empty, for example when its maximum is below its
minimum, the run now finishes without magnitude correction instead of failing.

### Corrupt WAV measurements are rejected

A float WAV that contains invalid samples is now reported as unreadable. It
used to load as an empty curve and fail later in the run.

### Zero values stay zero

A left IR window of 0 ms and a maximum boost of 0 dB are kept when you reopen
the app. Both used to snap back to their defaults.

### Headroom across the full filter

Final filters keep the requested headroom across the full frequency range,
including output tilt and normalization. Reports show any extra attenuation.

### More reliable timing with limited measurements

Timing estimates use phase within the measured frequency range. Reports explain
when there is too little phase coverage to estimate timing.

### Faster Bass Integration calculations

Bass Integration calculates timing faster. Band analysis uses the correct
response for odd filter orders.

### More dependable correction

Final filter checks catch bass and voice-band peaks in either channel. Bass
analysis recognises nearby resonances across sample rates and filter lengths.
Correction uses measured room decay; magnitude-derived RT60 estimates remain
diagnostic.

### Consistent repeated calculations

Changing measurement data no longer alters earlier cached inputs. RT60 estimates
stay consistent across filter lengths and calculation order. Earlier Automatic
mode results are rebuilt.

### Faster Mixed Phase calculation

Mixed Phase filters calculate faster, with pre-ringing protection and alignment
checks in place.

### Bass Integration for up to four subwoofers

Measure and integrate up to four subs, with previews and results for each one.
Separate filters include individual delay, level and polarity, with a six-output
CamillaDSP export for four subs and two main speakers. Combined level estimates
account for measured level differences and phase cancellation.

## [1.4.0] - 12-10-2026

### Decay before and after correction

New Decay / RT60 and Bass decay views compare measured and predicted decay
for each channel. The Timing view adds an impulse envelope in dB, and reports
include playback headroom recommendations. Decay estimates describe ringing,
not overall sound quality.

### Safer correction

Correction fades smoothly at measurement boundaries, and decay correction
respects its frequency limit. Phase correction avoids correcting the same
bass resonance twice. Pre-ringing protection now works, with a default limit
of -25 dB instead of -35 dB.

For left/right correction without Bass Integration, filters that lengthen bass
decay trigger a check of lighter phase correction. Reports explain any remaining
problems or why the original correction was kept when all checks failed.

### Separate filters for two subwoofers

Bass Integration supports individual delay, level and polarity for two subs,
with separate results views and a four-output CamillaDSP export. Both sub
measurements must share a timing reference. Bypass playback preserves
crossovers, subwoofer protection and bass alignment.

### More reliable Automatic mode

Adaptive targets stay within 1 dB of their base curve. Measured room decay and
distortion discourage extra boost, and measurement gaps no longer influence
adaptation. Search starts consistently on Windows and Linux. Earlier results
are rebuilt and may differ slightly.

### Less waiting

Minimum-phase calculation is faster on Windows and Linux; other filter types
are faster on Windows. Bass decay checks skip unnecessary recalculations.
Progress now tracks all four filters through export and shows accurate search
trial counts.

### Clearer setup

Basic is now Setup. Settings managed by Automatic mode sit in expandable
sections, zero values are preserved, and unreadable measurements prevent runs.
Room dimensions stay correct when switching units.

## [1.3.9] - 4-10-2026

### Easier from the start

Choose your measurement files in one window, with Left and Right side by side.
The sidebar keeps your current setup in view. Updated guides and screenshots
help you through the first run.

### Clearer results

See what changed and why. Correction explanations are easier to read, with more
room for the graphs and extra detail when you want it. Safety notices stay
visible, and the exported report includes the same explanations.

Graph downloads are easier to tell apart, even when you save several versions.

### More from the first run

Automatic mode now fine-tunes its best result more consistently on the first
run. It also makes better use of measured distortion and room decay when
choosing how much correction to apply.

### Less waiting. Less memory.

Filter calculations are faster, especially with long filters and repeated runs.
Automatic mode also uses less memory. Acoustic safety checks remain in place.

### More reliable measurements

Measurements run more smoothly, especially on slower computers. Timing
references are recognised more reliably in noisy recordings. If audio is
interrupted, DecayCore stops the take and tells you to try again.

### More precise bass correction

Temporal Decay Control responds more precisely to the bass decay in your room.
Two-subwoofer timing is also more reliable.

### Advanced, made simpler

Choose Strict, Balanced or Loose, preview the settings, and press Apply.
The new presets bring related correction controls together while keeping your
chosen bass protection and phase settings.

You can now enter effective room volume in cubic feet as well as cubic metres.

### Easier to open

Bookmark DecayCore's browser address and come back to it after a restart.
With LAN access enabled, open it from another computer on your trusted network.
Anyone who can reach the host can use the interface while LAN access is enabled.

Linux installation now includes the launcher needed to open your browser
automatically and checks that it is available.

### License terms included

Every download now includes the license and a list of third-party notices.
DecayCore is free for non-commercial use. Commercial use needs written
permission.

## [1.3.8] - 1-10-2026

### Faster Automatic mode

Automatic mode is much faster. On test machine, a full Automatic mode run
takes **116 seconds in version 1.3.7 and 49 seconds in version 1.3.8**. The
filters and safety checks are the same as before.

### New features

Make every filter type in one go with **START with all four filter types**.
DecayCore runs the complete pipeline once per filter type with the current
settings and saves four export bundles. In Automatic mode, each type gets its
own search.

Tell DecayCore about your room: the Advanced tab now has **Room dimensions**
fields for length, width and height in centimetres or feet. For irregular or
open-plan spaces, you can enter an effective volume directly. The room volume
feeds the Schroeder estimate and Automatic mode's frequency limit.

The Output settings header now shows the custom filter name and the enabled
sample-rate exports, including the optional 352.8 / 384 kHz rates, and updates
as you change settings.

### Plots

Plots are easier to read. Axes use round numbers (1, 2, 5 and their multiples),
the number of ticks fits the panel size, and labels stay distinct when you zoom
in far. Wide frequency axes show whole decades.

Panning and zooming are about 10 times faster on large curves.

### Measurement

Audio dropouts can no longer slip into your measurement. If the stream drops or
inserts samples during the sweep, DecayCore rejects the take and asks you to
measure again, instead of analysing it as valid.

When guided measurement fails, the error view lists which takes were rejected
and why. It adds checks for clipping, recording level, timing reference and
channel routing, so you know what to fix.

The Measurement tab is locked while a run is in progress. Every button and field
is greyed out, so a stray click can't start a measurement or replace your
measurement files mid-run. A note at the top of the tab says why, and the tab
unlocks when the run ends. The reverse also holds: the START buttons stay off
while a measurement is in progress, with a note explaining the wait.

### Uploads

Each speaker channel now has exactly one measurement source: the file shown
under **Speaker measurements**. Uploading a file clears any earlier path, and
entering a path clears the upload, so the run, the RT60 and harmonics data and
the target preview always use the file you see. You can also use a WAV file for
one channel and a TXT file for the other.

Automatic mode now caches runs made from uploaded or in-app measurements, so
reruns no longer start cold.

File pickers show a **Choose file** button and name the loaded file below the
field. Long file names wrap.

>> Thanks to **PO3c** from asr-forum for reporting this bug.

### DSP safety

Bass boost now stops at the speaker's measured −1 dB low-frequency limit in
every mode, including Unsafe Raw. Cuts remain available. A final check on the
finished filter lowers the level when needed, without changing phase or timing.
Export diagnostics report each channel's limit and safety gain. The
low-frequency rolloff detector also now smooths the response as intended.

### Reliability and security

DecayCore now keeps working when conditions get rough. Automatic mode starts
reliably, even on systems where its faster worker setup is unavailable. If
another DecayCore version changes your search history mid-run, the search stops
and tells you, so you never get a result quietly built on default settings.

The browser interface holds up when you leave several tabs open or upload large
files. A full queue now turns an upload away right away, so you don't wait for a
big file to transfer before it fails. A broken request no longer blocks later
connections. Pages shown inside the app are cleaned before they appear, and
every release is built the same way, so you get the same tested software each
time.

### Updating from an older version

Settings, caches and search histories saved by older versions are no longer
carried over. Settings that cannot be read fall back to defaults, and the first
Automatic mode run after updating starts from scratch. Select your measurement
device again.

## [1.3.7] - 20-9-2026

### Local network access

Fixed a blank page when opening DecayCore from another device in `--lan` mode.
Browsers do not provide `crypto.randomUUID()` over an unencrypted, non-local
HTTP connection, which previously stopped the frontend during initialization.

The frontend now falls back to cryptographically secure UUID generation using
`crypto.getRandomValues()`, allowing the interface and authenticated requests
to initialize normally over a trusted local network.

### Installation script on Linux systems

For measurement audio on Linux releases, DecayCore packages a pinned PortAudio build with
ALSA and PulseAudio host APIs. `install.sh` supplies only its host audio
libraries: `libasound2t64 libpulse0` on current Debian/Ubuntu releases (`libasound2` on older releases), or `alsa-lib libpulse pipewire-alsa pipewire-pulse wireplumber`
on an Arch PipeWire audio system.

Thanks to **mkusan** from asr-forum for finding these!

## [1.3.6] - 20-9-2026

### Shutdown

The shutdown button now displays a single persistent notification: “DecayCore
is shutting down. You can close this browser tab.” Genuine HTTP and loading
errors continue to be reported normally.

### Local network access

Local network access must now be enabled explicitly with the `--lan` option,
for example `./run.sh --lan`. DecayCore prints complete LAN addresses, including
the session token, to the console. Copy the entire address when opening
DecayCore on another device.

LAN mode uses an unencrypted HTTP connection and should only be enabled on a
trusted local network. TCP port 8080 may also need to be allowed through the
firewall for private networks.

## [1.3.5] - 19-9-2026

### A small bug, a stronger safety net

This release fixes a small, four-line stereo validation bug with a much longer
list of safeguards around it. The correction engine has not been broadly
changed: the extra work makes the existing safety policy more accurate,
transparent and dependable.

Final FIR validation now evaluates the least favourable left or right channel
for pre-ringing energy and group delay, while reporting both channels
separately. Pre-ringing is measured from the generated FIR itself, so natural
room behaviour no longer causes false rejections. Linear-phase centring delay
is removed before group-delay analysis, eliminating spurious results of around
1.36 seconds, and unreliable spectral-null points are excluded.

The surrounding failure paths are now deliberately conservative. Missing FIR
validation is rejected, new settings default to rejection while explicitly
saved warning policies remain compatible, and an all-rejected fallback is
clearly identified as a hard-gate failure in the report. If the excess-delay
guard cannot be calculated, excess-phase correction is disabled and the
diagnostic is retained instead of silently bypassing the safeguard.

Automatic mode's cache version has been advanced so results created under the
older validation policy cannot be reused. In short: a tiny root cause, a more
trustworthy final check, and clearer evidence when DecayCore chooses the safe
fallback.

## [1.3.4] - 19-9-2026

### Automatic mode

Adaptive-target selection has been redesigned. Automatic mode now evaluates all
built-in targets first and chooses the base curve that best fits the measurement.
The adaptive curve is built from that selection instead of always using Harman6.
Adaptation remains strictly bounded to −1…+1 dB relative to the selected base
target.

### Linux

Fixed automatic browser opening on recent KDE and Arch Linux systems. Packaged
builds now restore the system library path before starting `xdg-open`, preventing
KDE from loading DecayCore's bundled `libstdc++.so.6` and rejecting it because
required GLIBCXX or CXXABI versions are unavailable.

This also ensures that the browser receives the current tokenized DecayCore URL
instead of leaving the user with an old or unauthenticated page that reports
HTTP 403 errors.
