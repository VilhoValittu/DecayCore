# DecayCore Headless Scripts

Developer shell scripts for exercising `python -m decaycore.headless` without starting the GUI.

## Location

All scripts live in `tests/headless/scripts/`. They resolve the repository root automatically and export:

```bash
PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
```

Default stereo fixtures:

```text
tests/headless/speaker_data/measurement_left__20260422_154122__ir.wav
tests/headless/speaker_data/measurement_right__20260422_154153__ir.wav
```

Default bass-integration fixtures:

```text
tests/headless/speaker_data/subs_inc/left_final__ir.wav
tests/headless/speaker_data/subs_inc/right_final__ir.wav
tests/headless/speaker_data/subs_inc/sub1_final__ir.wav
tests/headless/speaker_data/subs_inc/sub2_final__ir.wav
```

Default output root is `/tmp/camillafir-headless-runs`, unless `--output` is passed.

Grafana does not read these output folders directly. The Docker worker writes Grafana data only after it sees a session under its `/data/in` mount. Use `--worker-in src/decaycore/worker/data/in` to publish completed metrics for ingestion.

## Scripts

- `run_headless_basic.sh`: simple stereo auto-mode run, Harman10, 20-250 Hz, no plots, metrics and summary.
- `run_headless_safe_basic.sh`: conservative user-safe run, 20-200 Hz, 4 dB boost, 12 dB cut, TDC and confidence pull.
- `run_headless_aggressive_bass.sh`: stronger bass correction stress test, 15-300 Hz, 6 dB boost, 18 dB cut, stronger TDC.
- `run_headless_phase_focus.sh`: phase and group-delay focused run with 600 Hz phase limit and GD limiter config.
- `run_headless_mixed_phase.sh`: mixed-phase run with 200 Hz split, transition width, and pre-ringing guard config.
- `run_headless_minimum_phase.sh`: minimum-phase magnitude-focused workflow with safe boost limits.
- `run_headless_linear_phase.sh`: linear-phase workflow with longer taps and Tukey windowing config.
- `run_headless_bass_integration_direct_dac.sh`: direct DAC main/sub run using main L/R and sub L/R inputs.
- `run_headless_rt60_harmonics.sh`: metadata-oriented run; accepts measurement metadata, RT60 JSON, and harmonics JSON where available.
- `run_headless_batch_folder.sh`: runs every child folder containing `config.json` and continues on failures.
- `run_headless_compare_presets.sh`: runs safe, default, aggressive bass, phase focus, and mixed phase presets for the same L/R input.
- `run_headless_optuna_sweep.sh`: compares 24, 48, 82, and 120 trial settings.
- `run_headless_tdc_sweep.sh`: compares TDC strengths 0.25, 0.50, 0.75, and 1.00, represented as 25, 50, 75, and 100.
- `run_headless_confidence_sweep.sh`: compares confidence pull off, 0.25, 0.50, and 0.75.

## Examples

```bash
bash tests/headless/scripts/run_headless_basic.sh

bash tests/headless/scripts/run_headless_compare_presets.sh \
  --left tests/headless/speaker_data/measurement_left__20260422_154122__ir.wav \
  --right tests/headless/speaker_data/measurement_right__20260422_154153__ir.wav \
  --output /tmp/camillafir-compare

bash tests/headless/scripts/run_headless_tdc_sweep.sh \
  --output /tmp/camillafir-tdc-sweep

bash tests/headless/scripts/run_headless_bass_integration_direct_dac.sh \
  --main-left tests/headless/speaker_data/subs_inc/left_final__ir.wav \
  --main-right tests/headless/speaker_data/subs_inc/right_final__ir.wav \
  --sub-left tests/headless/speaker_data/subs_inc/sub1_final__ir.wav \
  --sub-right tests/headless/speaker_data/subs_inc/sub2_final__ir.wav

bash tests/headless/scripts/run_headless_aggressive_bass.sh \
  --worker-in src/decaycore/worker/data/in
```

## Outputs

Each run writes an output-local `config.json`, then runs:

```bash
python -m decaycore.headless --config "$OUT/config.json" --output "$OUT" --mode auto --no-plots --no-png --write-summary --write-metrics
```

Expected files:

```text
metrics.json
summary.txt
run.log
```

Sweep and comparison scripts also write an aggregate JSON file in the requested output directory.

## Docker Worker Use

Start the worker stack from `src/decaycore/worker`:

```bash
docker compose -f src/decaycore/worker/docker_compose.yml up -d
```

Publish completed headless metrics into the worker input folder:

Example:

```bash
bash tests/headless/scripts/run_headless_compare_presets.sh \
  --output /tmp/camillafir-worker-input \
  --worker-in src/decaycore/worker/data/in
```

The scripts copy `metrics.json`, `summary.txt`, and `run.log` into `src/decaycore/worker/data/in/<session>/`. The worker waits for the folder to be stable, parses the metrics, writes one `camillafir` point to InfluxDB, and marks the input with `.done`.

For worker-side batch runs, prepare `src/decaycore/worker/data/in/<session>/config.json` folders, then run:

```bash
bash tests/headless/scripts/run_headless_batch_folder.sh \
  --input src/decaycore/worker/data/in \
  --output /tmp/camillafir-batch
```

If Grafana remains empty, check:

```bash
docker compose -f src/decaycore/worker/docker_compose.yml logs worker --tail=100
ls src/decaycore/worker/data/in
ls src/decaycore/worker/data/out
```
