# DecayCore - FIR Room Correction and Measurement Tool for Convolution DSP engines

DecayCore is a free FIR room correction, acoustic measurement, and filter generation tool for CamillaDSP, convolution WAV filters, Roon convolution workflows, Equalizer APO, and other FIR-capable DSP engines.

DecayCore includes its own measurement workflow in release builds. The preferred workflow is to measure directly with DecayCore, generate correction filters from those measurements, and export convolution-ready WAV FIR filters.

DecayCore runs through a browser-based user interface. The application starts a local UI that you use in your web browser; it is not a cloud service.

It focuses on physically sane, band-limited room correction instead of simply forcing a flat frequency response. DecayCore supports Linear Phase, Minimum Phase, Mixed Phase and Asymmetric FIR filters, automatic target optimization, phase-aware correction, and Temporal Decay Control for low-frequency room behavior.

>DecayCore was formerly known as CamillaFIR. The project was renamed to avoid confusion with CamillaDSP while keeping full CamillaDSP compatibility.

## Links

- Documentation: https://vilhovalittu.github.io/DecayCore/
- Releases: https://github.com/VilhoValittu/DecayCore/releases
- Source code: https://github.com/VilhoValittu/DecayCore

> Important note about the measurement function :
> The integrated measurement function is available only in the packaged versions published under the GitHub Releases section. It is not included in the public source tree. The source repository contains the filter-generation side, while the measurement/acquisition workflow remains available through the released builds.

REW-style measurement data may also be used in compatible workflows, but DecayCore's own measurement workflow is the preferred path.

## What DecayCore does

- Measures loudspeakers and rooms with the built-in measurement workflow in release builds
- Provides a local browser-based user interface
- Generates FIR room correction filters from measurement data
- Exports convolution-ready WAV FIR filters
- Supports CamillaDSP, Roon convolution, Equalizer APO, and other FIR-capable DSP engines
- Supports Linear Phase, Minimum Phase, Mixed Phase and Asymmetric FIR filters
- Uses conservative correction limits to avoid unsafe boosts and unrealistic room correction
- Includes automatic target optimization and Temporal Decay Control

## Documentation

- [Getting started](https://vilhovalittu.github.io/DecayCore/getting-started/)
- [Measurement workflow](https://vilhovalittu.github.io/DecayCore/measurement-workflow/)
- [CamillaDSP FIR room correction](https://vilhovalittu.github.io/DecayCore/camilladsp-fir-room-correction/)
- [FIR room correction](https://vilhovalittu.github.io/DecayCore/fir-room-correction/)
- [Minimum phase FIR filter generation](https://vilhovalittu.github.io/DecayCore/minimum-phase-fir-generator/)
- [Mixed phase room correction](https://vilhovalittu.github.io/DecayCore/mixed-phase-room-correction/)
- [Temporal Decay Control](https://vilhovalittu.github.io/DecayCore/temporal-decay-control/)
- [FAQ](https://vilhovalittu.github.io/DecayCore/faq/)

## Download

Download DecayCore from the official GitHub releases page:

https://github.com/VilhoValittu/DecayCore/releases

## Contact

Feedback: vilho.valittu@gmail.com

## License

DecayCore is source-available for personal, educational, research, and other
non-commercial use under the terms of the LICENSE file.

The measurement engine and related acquisition workflow are not included in this
repository and remain proprietary.

Commercial use, integration into commercial audio/DSP products, hosted services,
paid filtering services, or paid measurement/calibration workflows requires
separate written permission.
