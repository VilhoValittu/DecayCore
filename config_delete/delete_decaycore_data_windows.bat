@echo off
rem DecayCore - remove the Optuna disk cache and config.json (Windows).
rem
rem Mirrors src/decaycore/app_paths.py and
rem src/decaycore/auto_mode/optuna_backend_storage.py.
setlocal enableextensions

set "PROG=DecayCore"

rem On Windows data and config share %APPDATA%\DecayCore.
set "DATA_DIR=%APPDATA%\%PROG%"
set "LEGACY_DIR=%USERPROFILE%\.camillafir"

set "ASSUME_YES=0"
if /i "%~1"=="-y" set "ASSUME_YES=1"
if /i "%~1"=="--yes" set "ASSUME_YES=1"

echo The following will be deleted if present:
echo   Optuna cache:     "%DATA_DIR%\decaycore*optuna*.log"
echo   Optuna cache:     "%LEGACY_DIR%\decaycore*optuna*.log"
echo   Auto-mode cache:  "%DATA_DIR%\decaycore_auto_mode_cache_*.json"
echo   Filter priors:    "%DATA_DIR%\auto_mode_filter_priors.json"
echo   Config file:      "%DATA_DIR%\config.json"
echo.

if "%ASSUME_YES%"=="0" (
    set /p "REPLY=Delete these files? [y/N] "
    if /i not "%REPLY%"=="y" if /i not "%REPLY%"=="yes" (
        echo Aborted.
        exit /b 1
    )
)

rem del does not fail the script if files are missing; quote each pattern.
del /q "%DATA_DIR%\decaycore*optuna*.log" 2>nul && echo Deleted Optuna cache in "%DATA_DIR%"
del /q "%DATA_DIR%\decaycore*optuna*.log.lock" 2>nul
del /q "%LEGACY_DIR%\decaycore*optuna*.log" 2>nul && echo Deleted Optuna cache in "%LEGACY_DIR%"
del /q "%LEGACY_DIR%\decaycore*optuna*.log.lock" 2>nul
del /q "%DATA_DIR%\decaycore_auto_mode_cache_*.json" 2>nul && echo Deleted auto-mode cache in "%DATA_DIR%"
del /q "%LEGACY_DIR%\decaycore_auto_mode_cache_*.json" 2>nul
del /q "%DATA_DIR%\auto_mode_filter_priors.json" 2>nul && echo Deleted filter priors in "%DATA_DIR%"
del /q "%LEGACY_DIR%\auto_mode_filter_priors.json" 2>nul
del /q "%DATA_DIR%\config.json" 2>nul && echo Deleted "%DATA_DIR%\config.json"

echo Done.
endlocal
