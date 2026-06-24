# DecayCore Configuration and Data Paths

This document describes where DecayCore stores configuration files, Optuna optimization databases, measurements, and other data across different operating systems.

## Main Configuration File

The main `config.json` contains UI preferences, DSP settings, and user-configurable parameters.

### Linux / Unix-like Systems (XDG-compliant)
```
$XDG_CONFIG_HOME/DecayCore/config.json
```
- If `XDG_CONFIG_HOME` is not set, defaults to `~/.config/DecayCore/config.json`

**Example:** `/home/username/.config/DecayCore/config.json`

### macOS
```
~/Library/Application Support/DecayCore/config.json
```

**Example:** `/Users/username/Library/Application Support/DecayCore/config.json`

### Windows
```
%APPDATA%\DecayCore\config.json
```

**Example:** `C:\Users\username\AppData\Roaming\DecayCore\config.json`

---

## Data Directory (Measurements, Filters, Optuna Database)

The data directory contains measurement files, exported filters, and Optuna optimization journals.

### Linux / Unix-like Systems (XDG-compliant)
```
$XDG_DATA_HOME/DecayCore/
```
- If `XDG_DATA_HOME` is not set, defaults to `~/.local/share/DecayCore/`

**Example:** `/home/username/.local/share/DecayCore/`

### macOS
```
~/Library/Application Support/DecayCore/
```

**Example:** `/Users/username/Library/Application Support/DecayCore/`

### Windows
```
%APPDATA%\DecayCore\
```

**Example:** `C:\Users\username\AppData\Roaming\DecayCore\`

### Legacy Fallback (Linux/macOS/Windows)
If the primary data directory cannot be created, DecayCore falls back to the legacy directory:
```
~/.camillafir/
```

---

## Optuna Optimization Database

When using automatic mode, DecayCore uses Optuna to optimize filter parameters across trials. Optimization data is stored in journal files (SQLite databases) within the data directory.

### File Naming

Optuna journal filenames follow the pattern:
```
decaycore_optuna_[FILTER_TYPE]_[MEASUREMENT_ID]_[VERSION].log
```

**Parameters:**
- `[FILTER_TYPE]`: `asymmetric`, `linear`, `minimum`, or `mixed`
- `[MEASUREMENT_ID]`: Measurement identity or `nomeasurement` if no measurement is active
- `[VERSION]`: Program compatibility version (e.g., `v1_2_0`), omitted if current version

**Examples:**
- `decaycore_optuna_mixed_nomeasurement.log` — Mixed filter, no measurement context
- `decaycore_optuna_asymmetric_headphone_micdistance1cm.log` — Asymmetric filter, specific measurement
- `decaycore_optuna_linear_nomeasurement_v1_0_0.log` — Historical version-tagged database

### Target Curve Optimization

When running automatic mode with a specific target curve selected:
```
decaycore_optuna_target_[FILTER_TYPE]_[MEASUREMENT_ID]_[VERSION].log
```

**Example:**
- `decaycore_optuna_target_asymmetric_headphone.log` — Target curve optimization for asymmetric filters

### Path Examples

#### Linux
```
~/.local/share/DecayCore/decaycore_optuna_mixed_nomeasurement.log
~/.local/share/DecayCore/decaycore_optuna_target_asymmetric_headphone.log
```

#### macOS
```
~/Library/Application Support/DecayCore/decaycore_optuna_mixed_nomeasurement.log
~/Library/Application Support/DecayCore/decaycore_optuna_target_asymmetric_headphone.log
```

#### Windows
```
%APPDATA%\DecayCore\decaycore_optuna_mixed_nomeasurement.log
%APPDATA%\DecayCore\decaycore_optuna_target_asymmetric_headphone.log
```

#### Legacy Fallback (if primary directory unavailable)
```
~/.camillafir/decaycore_optuna_mixed_nomeasurement.log
```

---

## Resetting Caches and Configuration (Troubleshooting)

If DecayCore behaves strangely — for example automatic mode returns unexpected
results, reuses stale optimization data, or the app fails to start cleanly after
a settings change — **deleting the automatic-mode disk caches and `config.json`
is a recommended first step.** These files are regenerated automatically on the
next run, so removing them is safe and only forces a fresh recomputation.

What to remove:

- **Optuna journals:** `decaycore*optuna*.log` (and any `.log.lock` files)
- **Auto-mode result cache:** `decaycore_auto_mode_cache_*.json`
- **Filter priors:** `auto_mode_filter_priors.json`
- **Main config:** `config.json`

from the data directory (and the legacy `~/.camillafir/` fallback), as listed in
the tables above. Your saved measurements and exported filters under
`Documents/DecayCore/` are **not** affected.

### Helper scripts

The repository ships ready-made cleanup scripts under
[`config_delete/`](https://github.com/VilhoValittu/DecayCore/tree/main/config_delete)
that locate and remove exactly these files for you. Run the one for your OS:

| OS      | Script                                |
|---------|---------------------------------------|
| Windows | `delete_decaycore_data_windows.bat`   |
| macOS   | `delete_decaycore_data_macos.sh`      |
| Linux   | `delete_decaycore_data_linux.sh`      |

Each script lists the files it will delete and asks for confirmation first. The
shell scripts also accept `--dry-run` to preview without deleting, and
`-y`/`--yes` to skip the prompt.

---

## Migration from Legacy Paths

If you have DecayCore installed from a legacy version, configuration and Optuna databases are automatically migrated to the new platform-specific paths on first use:

- **Old Optuna path** (`~/.camillafir/`) → New platform-specific data directory
- **Old config path** → New platform-specific config directory (no auto-migration; starts fresh)

The legacy `~/.camillafir/` directory is not deleted; you may remove it manually after confirming the migration was successful.

---

## Summary Table

| Component | Linux | macOS | Windows |
|-----------|-------|-------|---------|
| **Config** | `~/.config/DecayCore/config.json` | `~/Library/Application Support/DecayCore/config.json` | `%APPDATA%\DecayCore\config.json` |
| **Data/Optuna** | `~/.local/share/DecayCore/` | `~/Library/Application Support/DecayCore/` | `%APPDATA%\DecayCore\` |
| **Legacy** | `~/.camillafir/` | `~/.camillafir/` | `~/.camillafir/` |
| **Measurements** | `~/Documents/DecayCore/measurement/` | `~/Documents/DecayCore/measurement/` | `%USERPROFILE%\Documents\DecayCore\measurement\` |
| **Filters** | `~/Documents/DecayCore/filters/` | `~/Documents/DecayCore/filters/` | `%USERPROFILE%\Documents\DecayCore\filters\` |

---

## Opening Paths from DecayCore

To locate these directories on your system:

### Linux
```bash
# Config
cat ~/.config/DecayCore/config.json

# Data/Optuna
ls ~/.local/share/DecayCore/

# Measurements
ls ~/Documents/DecayCore/measurement/
```

### macOS
```bash
# Config
cat ~/Library/Application\ Support/DecayCore/config.json

# Data/Optuna
ls ~/Library/Application\ Support/DecayCore/

# Measurements
ls ~/Documents/DecayCore/measurement/
```

### Windows (PowerShell)
```powershell
# Config
type "$env:APPDATA\DecayCore\config.json"

# Data/Optuna
dir "$env:APPDATA\DecayCore\"

# Measurements
dir "$env:USERPROFILE\Documents\DecayCore\measurement\"
```

Or open File Explorer and navigate to:
- **Config/Data:** `%APPDATA%\DecayCore\`
- **Measurements:** `%USERPROFILE%\Documents\DecayCore\measurement\`
