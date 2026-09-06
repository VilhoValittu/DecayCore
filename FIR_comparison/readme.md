# FIR Compare – Startup Instructions

Choose the `.7z` package that matches your operating system and processor architecture. Each package contains a standalone application; no separate Python or other runtime installation is required.

## Windows (x86_64)

Package: `FIR-Compare-windows-x86_64.7z`

1. Install 7-Zip from <https://www.7-zip.org/>.
2. Extract the package, for example to the Desktop or Downloads folder.
3. Open the extracted `FIR-Compare` folder.
4. Start the application by double-clicking `FIR-Compare.exe`.

If Windows SmartScreen displays a warning, verify that the package was downloaded from a trusted source. If necessary, select **More info** → **Run anyway**.

## macOS (Apple Silicon / arm64)

Package: `FIR-Compare-macos-arm64.7z`

1. Extract the `.7z` package using an application such as Keka or 7-Zip.
2. Open the extracted `FIR-Compare` folder.
3. Start the application by double-clicking `FIR-Compare`.

If macOS blocks the application, open **System Settings → Privacy & Security** and select **Open Anyway** for the application. Alternatively, start it from Terminal:

```bash
cd /path/to/the/extracted/FIR-Compare/folder
./FIR-Compare
```

This package is intended for Apple Silicon Macs (M1, M2, M3, M4, etc.). There is no separate Intel Mac package in this folder.

## Linux (x86_64)

Package: `FIR-Compare-linux-x86_64.7z`

1. Install 7-Zip through your distribution's package manager, for example on Debian or Ubuntu: `sudo apt install 7zip`.
2. Extract the package.
3. Open a terminal in the extracted `FIR-Compare` folder.
4. Start the application:

```bash
./FIR-Compare
```

The executable permission is preserved during extraction. If the operating system reports a permission error, run `chmod +x FIR-Compare` and start the application again.

## Linux (ARM64)

Package: `FIR-Compare-linux-arm64.7z`

Extract and start the application as described in the Linux x86_64 section, using this package on an ARM64 device:

```bash
./FIR-Compare
```

## Extracting from the command line

After installing 7-Zip, a package can also be extracted with:

```bash
7z x FIR-Compare-linux-x86_64.7z
```

Replace the filename with the package for your operating system.
