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

## Run from release package

### Windows

1. Download `DecayCore_<version>_windows.zip` from Releases.
2. Extract the ZIP.
3. Run `DecayCore.exe`.
4. If SmartScreen appears, choose `More info` -> `Run anyway`.
5. Allow private firewall access if prompted. DecayCore runs on internal server on your computer.
6. Open `http://127.0.0.1:8080` if the browser does not open automatically.

### Ubuntu / Debian Linux

1. Download `DecayCore_<version>_linux.tar.gz` from Releases.
2. Extract the archive.
3. Open Terminal in the extracted folder and run:

```bash
./run.sh
```

4. Built-in measurement audio requires the system PortAudio library. If measurement audio reports a PortAudio/backend error, install it first:

```bash
sudo apt install libportaudio2
```

5. Open `http://127.0.0.1:8080` if the browser does not open automatically.

### macOS (Intel + Apple Silicon)

1. Download `DecayCore_<version>_macos.tar.gz` from Releases.
2. Extract the archive.
3. Open Terminal in the extracted folder and run:

```bash
chmod +x DecayCore
./DecayCore
```

4. If macOS blocks first launch, open `System Settings -> Privacy & Security -> Open Anyway`.
5. Open `http://127.0.0.1:8080` if the browser does not open automatically.

## Run directly from Python source

Use this path if you want to run DecayCore from a cloned source tree instead of a packaged release. Python 3.11 or newer is recommended.

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

### Ubuntu / Debian Linux

1. Install Git, Python, venv support, pip, and the optional PortAudio system library used by built-in measurement audio:

```bash
sudo apt update
sudo apt install git python3 python3-venv python3-pip libportaudio2
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

3. If you use the built-in measurement audio, install PortAudio:

```bash
brew install portaudio
```

4. Open Terminal in the cloned DecayCore source folder.
5. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

6. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

7. Start DecayCore:

```bash
PYTHONPATH=src python -m decaycore
```

8. Open `http://127.0.0.1:8080` if the browser does not open automatically.

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
