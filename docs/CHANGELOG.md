---
title: DecayCore Changelog
description: Recent user-visible changes to DecayCore measurement, correction, export, and platform support.
hide_page_heading: true
---

# Changelog

This page contains the current development notes and three latest stable releases. [Older releases and Finnish translations are preserved in the archive]({{ '/changelog/archive/' | relative_url }}).

## [Unreleased]

### Reliability and security

Automatic-mode process workers now use a thread-safe process start method and
fall back to threads when a worker payload cannot be serialized. Browser
Markdown and ordinary HTML content are sanitized before rendering.

The continuous-integration workflow now runs the complete Python test suite and
Ruff checks. Release builds also pin PyInstaller and Maturin, including the
isolated native-extension build environments.

Lisäsin ohjatun mittauksen virhenäkymään syykohtaiset tarkistusohjeet. Virhe kertoo nyt myös, mitkä otot hylättiin ja miksi. Ohjeet neuvovat tarkistamaan esimerkiksi leikkautumisen, äänitystason, ajoitusreferenssin ja kanavareitityksen.

Advanced-välilehdelle lisättiin huoneen pituuden, leveyden ja korkeuden kentät sekä yksikkövalinta senttimetreille tai jaloille.
Mitoista laskettu tilavuus vaikuttaa Schroeder-arvioon ja automaattitilan taajuusrajaan.

Lisätty erillinen start-nappi joka tekee putkeen jokaisen filtterityypin ajot.

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

## [1.3.3] - 18-9-2026

### UI

The **Modern** style has a new layout. The packaged application now uses
DecayCore's native browser UI engine and no longer includes NiceGUI.

The new **Apply saved auto settings** button in Basic and Advanced modes lets
you apply saved Automatic mode settings for the selected filter type, with the
newest settings listed first. The current target curve and mode safety limits
are preserved.

### Automatic mode

After running Automatic mode with an adaptive target, you can download that
target and use it in Basic and Advanced modes.
