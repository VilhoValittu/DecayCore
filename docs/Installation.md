---
title: DecayCore Installation Guide
nav_title: Installation
description: Install DecayCore from a packaged release or run it from Python source on Windows, Linux, and macOS.
permalink: /installation/
---

## Recommended path

For most users, the recommended option is:

1. Download the latest packaged release
2. Extract it
3. Run DecayCore
4. Open `http://127.0.0.1:8080` if the browser does not open automatically

Latest release:
- [Latest DecayCore release](https://github.com/VilhoValittu/DecayCore/releases/latest)

All releases:
- [DecayCore releases](https://github.com/VilhoValittu/DecayCore/releases)

## Python and dependency baseline

All DecayCore versions released and documented in this repository are based on Python `3.12.3`.

Current dependency baselines from the repository requirement files are:

- `requirements.txt`: `numpy==2.4.6`, `scipy==1.17.1`, `nicegui==3.13.0`, `plotly==6.8.0`, `optuna==4.9.0`, `numba==0.65.1`

## Run from release package

---

### Windows

1. Download `DecayCore_<version>_windows.7z` from Releases.
2. Extract the ZIP.
3. Run `DecayCore.exe`.
4. If SmartScreen appears, choose `More info` -> `Run anyway`.
5. Allow private firewall access if prompted. DecayCore runs on internal server on your computer.
6. Open `http://127.0.0.1:8080` if the browser does not open automatically.

---

### Ubuntu / Debian Linux

1. Download `DecayCore_<version>_linux.7z` from Releases.
2. Extract the archive.
3. Open Terminal in the extracted folder and run:

```bash
./run.sh
```

4. **Built-in measurement:** Measurement has been verified to work on Windows. Linux has been verified to work at least on Ubuntu 22.04. macOS could not be tested due to unavailable test hardware. On platforms where measurement is unavailable, compatible external measurements can be used.

5. Built-in measurement audio requires the system PortAudio library. If measurement audio reports a PortAudio/backend error, install it first:

```bash
sudo apt install libportaudio2
```

5. Open `http://127.0.0.1:8080` if the browser does not open automatically.

---

### Raspberry Pi / Linux ARM64

1. Download `DecayCore_<version>_linux_arm64.7z` from Releases.
2. Extract the archive.
3. Open Terminal in the extracted folder and run:

```bash
./run.sh
```

4. Use this build for Raspberry Pi 4/5 running a 64-bit operating system, Debian/Ubuntu Linux ARM64 systems, and other 64-bit ARM Linux machines.
5. This build does not support 32-bit Raspberry Pi OS.
6. Open `http://127.0.0.1:8080` if the browser does not open automatically.

---

### macOS (Apple Silicon)

1. Download `DecayCore_<version>_macos_arm64.7z` from Releases.
2. Extract the archive.
3. Double-click `Start_Decay.command` to launch DecayCore through Terminal.
4. If macOS blocks the first launch, open `System Settings -> Privacy & Security -> Open Anyway`.
5. If you prefer the bundle directly, you can also double-click `DecayCore_<version>.app`.
6. If you prefer Terminal, open Terminal in the extracted folder and run:

```bash
./Start_Decay.command
```

7. **Built-in measurement:** Measurement has been verified to work on Windows. Linux has been verified to work at least on Ubuntu 22.04. macOS could not be tested due to unavailable test hardware. macOS users can use compatible external measurements.
8. If macOS asks for microphone access, allow DecayCore. Measurement files are saved under `Documents/DecayCore/measurement` by default and fall back to a writable app-data location if needed.
9. Open `http://127.0.0.1:8080` if the browser does not open automatically.

---

## Run directly from Python source

Use this path if you want to run DecayCore from a cloned source tree instead of a packaged release. The documented baseline is Python `3.12.3`.

Repository:
- [DecayCore source repository](https://github.com/VilhoValittu/DecayCore)

### Get the source with Git

Install Git first:
- Windows: https://git-scm.com/download/win
- Linux: use your distribution package manager, for example `sudo apt install git`
- macOS: install Xcode Command Line Tools with `xcode-select --install`, or install Git with Homebrew

Clone the repository:

```bash
git clone https://github.com/VilhoValittu/DecayCore.git
cd DecayCore
```

To update an existing clone later:

```bash
cd DecayCore
git pull
```

---

### Windows

1. Install Python from https://www.python.org/downloads/windows/ and enable `Add python.exe to PATH`.
2. Open PowerShell in the cloned DecayCore source folder.
3. Create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. Start DecayCore:

```powershell
$env:PYTHONPATH = "src"
python -m decaycore
```

6. Open `http://127.0.0.1:8080` if the browser does not open automatically.

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the virtual environment again.

---

### Ubuntu / Debian Linux

1. Install Git, Python, venv support, pip :

```bash
sudo apt update
sudo apt install git python3 python3-venv python3-pip
```

2. Open Terminal in the cloned DecayCore source folder.
3. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. Start DecayCore:

```bash
PYTHONPATH=src python -m decaycore
```

6. Open `http://127.0.0.1:8080` if the browser does not open automatically.

---

### macOS

1. Install Python from https://www.python.org/downloads/macos/ or with Homebrew.
2. Install Git with Xcode Command Line Tools or Homebrew:

```bash
xcode-select --install
```

or:

```bash
brew install git
```

3. Open Terminal in the cloned DecayCore source folder.
4. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

5. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

6. Start DecayCore:

```bash
PYTHONPATH=src python -m decaycore
```

7. Open `http://127.0.0.1:8080` if the browser does not open automatically.

---

## Output path

Output ZIP files are saved by default to:

```text
Documents/DecayCore/filters/<version>/
```

If that path is not writable, DecayCore falls back to a safe writable directory and reports the final path in Results.

## Browser and PNG notes

- DecayCore UI is browser-based.
- Interactive graphs can be saved from the graph download button in the UI.
- ZIP export is focused on filter artifacts and summary data.
- Dashboard image inclusion can be disabled in performance mode.

## Known issue: Windows + Vivaldi

In some Windows setups, using Vivaldi can trigger NumPy `MemoryError` under browser memory pressure.

Workarounds:

- use Chrome, Edge, or Firefox
- close extra Vivaldi tabs or extensions
- re-run the process in another browser if needed
