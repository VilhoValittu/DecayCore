# DecayCore — FIR Room Correction and Acoustic Measurement

Your speakers play the music. Your room leaves its mark. Bass builds up. Some notes linger. Others disappear at the listening seat.

DecayCore helps you measure what is happening and build FIR correction filters for your system. Measure your room, choose a target, and export filters ready to load into your DSP. The packaged app brings measurement and filter generation together in one workflow, free for non-commercial use.

**[Download DecayCore](https://github.com/VilhoValittu/DecayCore/releases)** · [Getting started](https://vilhovalittu.github.io/DecayCore/getting-started/) · [Documentation](https://vilhovalittu.github.io/DecayCore/)

## Correction with restraint

A deep dip in a measurement can tempt you to turn up the gain. Room acoustics rarely makes it that simple.

DecayCore puts controlled cuts first. It keeps correction within defined frequency bands and allows limited boost only where the measurement supports it. Deep nulls and uncertain regions are guarded against excessive correction.

You can use automatic target optimization in the packaged app or take control with Basic and Advanced manual filtering. Linear Phase, Minimum Phase, Mixed Phase and Asymmetric FIR filters let you choose an approach for your system. Phase-aware correction and Temporal Decay Control add tools for addressing timing and low-frequency decay.

## From measurement to playback

1. **Measure your room.** Use the built-in measurement workflow in the packaged release. Compatible REW-style measurement data can also be used.
2. **Build your correction.** Choose your target and filter settings, then generate FIR filters from your measurements.
3. **Load the filters and listen.** Export convolution-ready WAV files for CamillaDSP, Roon convolution, Equalizer APO, MiniDSP, or another FIR-capable DSP engine.

The app runs locally on your computer, with controls in your web browser. No cloud service is involved.

## Choose your version

**The packaged release** includes integrated measurement and the automatic-mode decision engine. Download it from [GitHub Releases](https://github.com/VilhoValittu/DecayCore/releases) for the complete workflow.

**The public source** provides filter generation with Basic and Advanced manual controls for non-commercial use. The measurement engine, automatic-mode decision engine, and related packaged workflows are proprietary and are not included in the public source tree. See the [source repository](https://github.com/VilhoValittu/DecayCore) and the license terms below.

> DecayCore was formerly called CamillaFIR. The name changed to avoid confusion with CamillaDSP; full CamillaDSP compatibility remains.

## Screenshots

![Files tab — load measurement files, inspect metadata, and set output format](docs/pics/ui_1.png)

![Measure tab — configure capture devices and run guided room measurements](docs/pics/ui_2.png)

![Basic tab — choose operating mode, FIR engine, and sample rate](docs/pics/ui_3.png)

![Target tab — shape the target curve, leveling, and gain behavior](docs/pics/ui_4.png)

![Advanced tab — refine correction shaping, bass protection, and confidence controls](docs/pics/ui_5.png)

![IR Window & Decay Control tab — control export windowing and temporal processing](docs/pics/ui_6.png)

![XO tab — define crossover filters between bands](docs/pics/ui_7.png)

![Start / Results tab — launch correction and follow its progress](docs/pics/ui_8.png)

## Documentation

- [Getting started](https://vilhovalittu.github.io/DecayCore/getting-started/)
- [Measurement workflow](https://vilhovalittu.github.io/DecayCore/measurement-workflow/)
- [User manual](https://vilhovalittu.github.io/DecayCore/User_Manual.html)
- [Engineering documentation](https://vilhovalittu.github.io/DecayCore/engineering/)
- [Glossary](https://vilhovalittu.github.io/DecayCore/glossary/)
- [FAQ](https://vilhovalittu.github.io/DecayCore/faq/)

## Download

Ready to hear what correction can do in your room?

[Download the packaged app](https://github.com/VilhoValittu/DecayCore/releases) and follow the [getting started guide](https://vilhovalittu.github.io/DecayCore/getting-started/).

## Contact

Questions, feedback, or something that could work better? Write to vilho.valittu@gmail.com.

## Python and dependency baseline

All DecayCore versions released and documented in this repository are based on Python `3.12.3`.

The main source environment currently documented by `requirements.txt` uses these pinned package versions:

- `numpy==2.4.6`
- `scipy==1.17.1`
- `nicegui==3.13.0`
- `plotly==6.8.0`

> `numba` was removed in v1.1.6. Public source builds may optionally compile the
> `decaycore-dsp` Rust extension for faster manual filtering. Automatic mode uses
> a separate native decision engine that is available only in packaged releases;
> source builds provide the full Basic and Advanced manual-filtering workflow.
> See the [Installation guide](https://vilhovalittu.github.io/DecayCore/installation/) for steps.

## License

DecayCore is source-available for personal, educational, research, and other
non-commercial use under the terms of the LICENSE file.

The measurement engine, automatic-mode decision engine, and related packaged
workflows are not included in this repository and remain proprietary.

Commercial use, integration into commercial audio/DSP products, hosted services,
paid filtering services, or paid measurement/calibration workflows requires
separate written permission.
