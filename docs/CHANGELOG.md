---
title: DecayCore Changelog
description: Recent user-visible changes to DecayCore measurement, correction, export, and platform support.
hide_page_heading: true
permalink: /changelog/
---

# Changelog

This page contains the five latest stable releases. [Older releases and Finnish translations are preserved in the archive]({{ '/changelog/archive/' | relative_url }}).

## [Unreleased]

## [1.4.7] - 27-11-2026

### Less memory at high sample rates

Filters at 192 kHz and above take about half as much memory on Linux and a fifth less on Windows, also when you export every sample rate. The filters stay the same.

### Closer miniDSP and Hybrid IIR filters

The same number of PEQ filters follows the correction about twice as closely. The Filter graph shows the FIR and PEQ filters together, as your player runs them. Hybrid IIR now runs only when you turn it on. Earlier Automatic results are rebuilt.

### More accurate measurements

DecayCore's own measurements read distortion and signal-to-noise at their true level, keep each speaker's and subwoofer's arrival time, and warn when background noise affects the bass. Re-measure older sessions to use this.

### Start delay for measurements

Set up to 20 seconds on the Measure page, and the sweep waits that long after Start or Continue. You get time to step away from the microphone and let the room go quiet.

### Safer Bass Integration

Bass Integration tunes sub delay and polarity against the finished filters, and asks you to re-measure files that have lost the main/sub timing. Sub gain stops at +6 dB, and the mains keep their level when the sub gets a boost.

### Also new

An HQPlayer convolution folder, bass and treble shelves for the target, a Harman 6 dB · Horn speakers target, and steadier Automatic results.

## [1.4.6] - 18-11-2026

### miniDSP filters that fit the FIR block

With miniDSP ticked, the filters fit the 4096-tap pool of a 2x4 HD or Flex: 2042 taps for two outputs, 1362 with a subwoofer, 1024 with two. Output settings, Setup and START show the limit, and your tap setting stays saved.

## [1.4.5] - 18-11-2026

### Filters for miniDSP

Tick miniDSP files in Output settings. The FIR then corrects phase only, and PEQ filters carry the EQ, including boosts and room-mode cuts. Set how many per output below the box; the default is 10.

The high-pass goes to the plugin's crossover block instead of the FIR. The README gives its type and frequency, and the export adds it as biquad files. Export at the plugin's sample rate. Hybrid IIR and the Equalizer APO, PipeWire, HLC, Roon and JRiver files are unavailable while miniDSP is ticked.

### Clearer START page

Pick Standard or All four filter types, tick Automatic tap count for either, then press the one START button. Output settings has its own page in the sidebar, just above START. START shows your output choices in one line, and Change… opens the page.

### Filters ready for Roon and JRiver

Tick Roon convolution ZIP or JRiver convolution folder in Output settings. Each holds your filters for every exported rate, so the player switches to the one that matches playback. Roon takes the ZIP as it is. The export then skips the CamillaDSP files, so each filter is saved once. Filters with subwoofer outputs still need CamillaDSP.

### Compare filters inside DecayCore

The new Compare FIRs page puts saved filters side by side in the built-in FIR Compare, with each predicted response against the target the run used. Compare filters on the Results page opens it with the run, or the whole batch, selected.

### Targets saved with your filters

Each filter ZIP now holds the target each channel was designed for, as a REW-compatible text file. Compare FIRs lists runs saved from this version on. Add older filters as WAV files.

### Automatic mode goals now pick different filters

Room-safe now favors less boost and gentler correction, Low-ripple the smoothest bass, and Subwoofers the closest match to the target from 20 to 200 Hz. Before, all three picked the same filter as Balanced 90% of the time. Balanced is unchanged. Earlier Automatic results are rebuilt.

### Measurements on macOS record to the end

On macOS the microphone could stop a moment into the sweep while playback went on, and the measurement timed out. DecayCore no longer checks audio devices during a measurement, which is what stopped the recording. Thanks to **PaulZH** from ASR-forum for helping solve this.

### Bass phase correction no longer fades run by run

Automatic mode no longer weakens bass phase correction a little more on every run. It could fade to zero, leaving the FIR with no bass phase correction. Earlier Automatic results are rebuilt.

### Full bass from short filters

Short filters no longer roll off the bass. At 2048 taps and 96 kHz they lost about 4 dB at 20 Hz and 25 dB at 5 Hz. Earlier Automatic results are rebuilt.

### Hybrid IIR only with CamillaDSP alone

Hybrid IIR can't be ticked while Output settings adds Equalizer APO, PipeWire, HLC, Roon or JRiver files. Those play the FIR alone and would miss the IIR cuts. The setup panel shows when Hybrid IIR is on.

### Equalizer APO, PipeWire and HLC files ticked separately

Output settings now has one box each for the Equalizer APO, PipeWire and HLC convolver files. If you had ticked the shared Equalizer APO and PipeWire box, both stay on. HLC files are no longer added unless you tick them.

### Faster Linear Phase calculation

Linear Phase filters calculate about 25% faster. The filters are unchanged.

## [1.4.4] - 13-11-2026

### Intel Macs get their own package

Releases now include a macOS package for Intel Macs next to the Apple silicon one. Download the one that matches your Mac.

### Phase correction works better

Automatic mode keeps the phase correction unless dropping it clearly helps, and the group-delay limiter now covers the bass too, so room modes near 30 Hz no longer slip through. If the limiter's result is still too strong, the correction is scaled down until it fits.

### Resonances are cut deeper

Automatic mode now judges tone at matched level, so room-mode peaks come down 0.5 to 1.6 dB further on a reference measurement. A resonance that is cut until its decay can no longer be measured counts as fixed.

### Left and right stay balanced

With stereo link on, both channels keeps balance better compared to each other.

### Measurement starts with your default devices

The Measure page now selects your system's default microphone and output when you have no saved choice.

### Faster Automatic mode on Windows

Automatic mode runs about 15% faster on Windows.

### Record test for audio problems

If measurement stops recording, start DC with : `--audio-record-test` extension in a terminal. It records three seconds through two audio paths and prints which one works. Send the output with your log file. Command works at every OS.

### Quicker error when the input stops

If the microphone stops delivering audio during a measurement, DecayCore now gives up after a few seconds of silence instead of waiting for each retry to time out. The log file also records why a measurement failed.

## [1.4.3] - 30-10-2026

### Send a log file with bug reports

Tick Save a log file in About DecayCore, repeat the problem, and click Email the log. Your email app opens a draft to the developer. Attach the newest file from Documents/DecayCore/logs, since an email link cannot attach it.

### Audio diagnostics and LAN access on macOS

Run `./Start_Decay.command --audio-diagnostics` to list your audio devices, or `--lan` to open DecayCore from another computer. With `--lan` the addresses print in the terminal, macOS asks for microphone access for Terminal, and closing the terminal stops DecayCore.

### Placement hints for deep bass nulls

Results lists up to three deep, narrow nulls that no filter can fill, each with the distance to a reflecting surface that can cause it. Moving the speaker, subwoofer or seat does more than any filter.

### Settings files for Equalizer APO and PipeWire

Tick Equalizer APO and PipeWire files in Output settings to add a ready-made config for each player to the export, next to the CamillaDSP files.

### Profiles for your settings

Save your settings under a name on the Setup page and load them again, for music and movies for example. A profile holds settings only, not measurement files. Undo restores what a load replaced.

### Results lead with graphs

Results opens with the graphs. The details sit in three groups, and the ZIP download is a small link beside the saved folder.

### Crossover models no longer fail the safety check

Asymmetric and Linear Phase filters for a speaker with crossovers set under XO pass the safety check. Undoing a crossover's phase puts part of the pulse just ahead of its peak, and the check counted that as pre-ringing. It now allows for it. Earlier Automatic results are rebuilt.

### Automatic tap count shortens many more filters

Most Minimum and Mixed Phase filters now shorten by about 60%, Linear Phase filters by about a third, and more Asymmetric filters shorten too. An exported filter stays within 1% (about 0.09 dB) of the design response. The search takes a few seconds longer.

### DecayCore asks before quitting mid-run

The Quit button asks first while a run or measurement is in progress, and the browser warns before you close the tab. A long Automatic search is no longer lost by accident.

### Undo for Apply mode defaults

Undo puts back the values that Apply mode defaults replaced. The message tells you how many values changed.

### Measure says why capture is unavailable

The Measure page shows the reason above the buttons, such as no microphone or output device, and points to loading measurements you already have.

### Edit manually from Automatic mode

Pages that Automatic mode manages have an Edit manually in Advanced mode button. In the Modern and North styles their tabs are labelled Auto. The Target preview says when its curve is for reference only.

### Navigation follows the working order

In the Modern and North styles, tabs are grouped as Prepare, Fine-tune and Run, and each page before START ends with a Next button. The group names show in windows tall enough to fit them.

### Run all four filter types keeps every result

Each filter type's result and download stay on the page, one section per type, with the latest open. Before, only the last result remained.

### A tidier XO page and labels

The XO page shows the crossovers you use plus one empty slot, and its heading no longer repeats the step number. Labels no longer use emoji, so they look the same on every system.
