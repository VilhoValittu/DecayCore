---
title: DecayCore Changelog
description: Recent user-visible changes to DecayCore measurement, correction, export, and platform support.
hide_page_heading: true
---

# Changelog

This page contains the five latest stable releases. [Older releases and Finnish translations are preserved in the archive]({{ '/changelog/archive/' | relative_url }}).

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
