# config_delete

Maintenance scripts that remove DecayCore's **automatic-mode disk caches**
(Optuna journals, the auto-mode result cache, and filter priors) and its
**`config.json`** from the default per-user storage locations. Use these to
force automatic mode to recompute from scratch or to reset the app to a clean
state.

Run the script for your operating system:

| OS      | Script                                |
|---------|---------------------------------------|
| Windows | `delete_decaycore_data_windows.bat`   |
| macOS   | `delete_decaycore_data_macos.sh`      |
| Linux   | `delete_decaycore_data_linux.sh`      |

Each script asks for confirmation before deleting. Pass `-y` / `--yes`
(or answer the prompt) to delete without further questions. The shell scripts
also support `--dry-run` to print what would be removed without deleting.

## What gets deleted

These paths mirror `src/decaycore/app_paths.py` and
`src/decaycore/auto_mode/optuna_backend_storage.py`.

**Auto-mode disk caches** from the user data directory and the legacy
`~/.camillafir` fallback:

- `decaycore*optuna*.log` — Optuna journal files (plus any `.lock` files)
- `decaycore_auto_mode_cache_*.json` — auto-mode result cache
- `auto_mode_filter_priors.json` — filter priors

| OS      | Data directory                                  |
|---------|-------------------------------------------------|
| Windows | `%APPDATA%\DecayCore`                            |
| macOS   | `~/Library/Application Support/DecayCore`        |
| Linux   | `$XDG_DATA_HOME/DecayCore` or `~/.local/share/DecayCore` |

**Config file** `config.json` from the config directory:

| OS      | Config directory                                |
|---------|-------------------------------------------------|
| Windows | `%APPDATA%\DecayCore`                            |
| macOS   | `~/Library/Application Support/DecayCore`        |
| Linux   | `$XDG_CONFIG_HOME/DecayCore` or `~/.config/DecayCore` |

## What is NOT touched

Saved measurements and exported filters live under `Documents/DecayCore` and are
**not** removed.
