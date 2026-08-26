---
title: DecayCore Installation Guide
nav_title: Installation
description: Install a packaged DecayCore release on Windows, Linux, or macOS, or run the manual workflow from source.
permalink: /installation/
---

## Recommended installation

Download the latest packaged release, extract it, and run the launcher for your platform. Packaged releases include guided measurement and the native Automatic mode engine.

[Download the latest release](https://github.com/VilhoValittu/DecayCore/releases/latest)

| Platform | Package and launcher | First-launch notes |
|---|---|---|
| Windows | Extract `DecayCore_<version>_windows.7z`, then run `DecayCore.exe`. | If SmartScreen appears, select **More info → Run anyway**. Allow private firewall access if asked. |
| Ubuntu / Debian | Extract `DecayCore_<version>_linux.7z`, run `./install.sh`, then start with `./run.sh`. | The installer adds the required host ALSA/Pulse client libraries. Use `./install.sh --check` for a read-only dependency check. |
| Arch Linux | Extract `DecayCore_<version>_linux.7z`, run `./install.sh`, then start with `./run.sh`. | On an existing PipeWire audio system, the installer adds the matching ALSA/Pulse integration packages. |
| Raspberry Pi / Linux ARM64 | Extract `DecayCore_<version>_linux_arm64.7z`, run `./install.sh`, then start with `./run.sh`. | Intended for Raspberry Pi 4/5 and other 64-bit ARM Linux systems. It does not support 32-bit Raspberry Pi OS. |
| macOS Apple Silicon | Extract `DecayCore_<version>_macos_arm64.7z`, then open `Start_Decay.command`. | If blocked, use **System Settings → Privacy & Security → Open Anyway**. Allow microphone access if requested. |

For measurement audio, DecayCore packages a pinned PortAudio build with ALSA and PulseAudio host APIs. `install.sh` supplies only its host audio libraries: `libasound2 libpulse0` on Debian/Ubuntu, or `alsa-lib libpulse pipewire-alsa pipewire-pulse wireplumber` on an Arch PipeWire audio system.

To print the packaged process's PortAudio host APIs and raw device list without opening an audio stream, run `./run.sh --audio-diagnostics`.

DecayCore normally opens its local browser interface automatically. If it does not, open `http://127.0.0.1:8080`. The interface runs on your computer; it is not a cloud service.

See [Measurement]({{ '/measurement-workflow/' | relative_url }}) for current platform support and routing requirements.

## Run from source

Use the source workflow for development or the public manual filtering engine. A source checkout supports **Basic** and **Advanced** modes. Guided measurement and the native Automatic mode decision engine are available only in packaged releases.

The documented Python baseline is `3.12.3`. Current pinned dependencies are listed in `requirements.txt`.

### 1. Clone the repository

```bash
git clone https://github.com/VilhoValittu/DecayCore.git
cd DecayCore
```

### 2. Create an environment and install dependencies

#### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate the environment again.

#### Ubuntu / Debian Linux

```bash
sudo apt update
sudo apt install git python3 python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### macOS

Install Python 3 and Git first, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Optional Rust DSP acceleration

Packaged releases already include the public `decaycore-dsp` extension. Source runs can use the pure-Python fallback, but the Rust extension speeds up manual filtering.

Install a Rust toolchain from [rustup](https://rustup.rs/), activate the Python environment, and run:

```bash
python -m pip install ./decaycore-dsp
```

On Windows, Rust also needs the MSVC build tools offered by the `rustup-init.exe` installer. When the extension is unavailable, DecayCore logs a fallback warning and continues with Python DSP.

### 4. Start DecayCore

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m decaycore
```

Linux and macOS:

```bash
PYTHONPATH=src python -m decaycore
```

## Output location

Filter ZIP files are saved by default under:

```text
Documents/DecayCore/filters/<version>/
```

If the directory is not writable, DecayCore chooses a writable application-data location and shows it on **START / Results**.

## Browser notes

Use a current Chrome, Edge, Firefox, or Safari release. If Vivaldi on Windows causes a NumPy `MemoryError` during a demanding run, close other tabs or repeat the run in another supported browser.
