@echo off
rem DecayCore - remove Python bytecode cache directories from the source tree
rem (Windows).
setlocal enableextensions

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
if not exist "%REPO_ROOT%\pyproject.toml" goto invalid_root
if not exist "%REPO_ROOT%\src\decaycore\" goto invalid_root

set "ASSUME_YES=0"
set "DRY_RUN=0"

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="-y" (
    set "ASSUME_YES=1"
    shift
    goto parse_args
)
if /i "%~1"=="--yes" (
    set "ASSUME_YES=1"
    shift
    goto parse_args
)
if /i "%~1"=="-n" (
    set "DRY_RUN=1"
    shift
    goto parse_args
)
if /i "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto parse_args
)
if /i "%~1"=="-h" goto help
if /i "%~1"=="--help" goto help
echo Unknown option: %~1 1>&2
exit /b 2

:args_done
set "ACTION=LIST"
set "TARGET_COUNT=0"
call :for_each_target

if "%TARGET_COUNT%"=="0" (
    echo Nothing to delete - no project __pycache__ directories found.
    exit /b 0
)

if "%DRY_RUN%"=="1" (
    echo ^(dry run - nothing deleted^)
    exit /b 0
)

if "%ASSUME_YES%"=="1" goto confirmed
set "REPLY="
set /p "REPLY=Delete these directories? [y/N] "
if /i "%REPLY%"=="y" goto confirmed
if /i "%REPLY%"=="yes" goto confirmed
echo Aborted.
exit /b 1

:confirmed
set "ACTION=DELETE"
set "DELETE_FAILED=0"
call :for_each_target
if "%DELETE_FAILED%"=="1" exit /b 1
exit /b 0

:for_each_target
if /i "%ACTION%"=="LIST" echo The following Python cache directories will be deleted:
call :visit_target "%REPO_ROOT%\__pycache__"
call :scan_root "%REPO_ROOT%\src"
call :scan_root "%REPO_ROOT%\tests"
call :scan_root "%REPO_ROOT%\scripts"
call :scan_root "%REPO_ROOT%\pyinstaller_hooks"
call :scan_root "%REPO_ROOT%\decaycore-dsp"
call :scan_root "%REPO_ROOT%\decaycore-scoring"
exit /b 0

:scan_root
if not exist "%~1\" exit /b 0
for /d /r "%~1" %%D in (__pycache__) do call :visit_target "%%~fD"
exit /b 0

:visit_target
if not exist "%~1\" exit /b 0
if /i "%ACTION%"=="LIST" (
    echo   %~1
    set /a TARGET_COUNT+=1
    exit /b 0
)
rmdir /s /q "%~1"
if exist "%~1\" (
    echo Failed:  %~1 1>&2
    set "DELETE_FAILED=1"
) else (
    echo Deleted: %~1
)
exit /b 0

:help
echo Usage: %~nx0 [-y^|--yes] [-n^|--dry-run]
exit /b 0

:invalid_root
echo Refusing to continue: DecayCore source root was not found. 1>&2
exit /b 2
