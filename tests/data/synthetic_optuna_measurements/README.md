# Synthetic Optuna Measurement Set

This folder contains five synthetic stereo measurement pairs in the plain text format accepted by CamillaFIR.

Files:
- `pair_01_neutral_room_L.txt`, `pair_01_neutral_room_R.txt`
- `pair_02_bass_mode_heavy_L.txt`, `pair_02_bass_mode_heavy_R.txt`
- `pair_03_crossover_misaligned_L.txt`, `pair_03_crossover_misaligned_R.txt`
- `pair_04_bright_resonant_L.txt`, `pair_04_bright_resonant_R.txt`
- `pair_05_asymmetric_room_L.txt`, `pair_05_asymmetric_room_R.txt`

What each pair simulates:
- `pair_01_neutral_room`: a mostly neutral in-room response with mild bass support and small left/right variation.
- `pair_02_bass_mode_heavy`: strong low-frequency room modes and a deep bass null, useful for testing bass control and regularization.
- `pair_03_crossover_misaligned`: a crossover-region dip with stronger phase rotation, useful for mixed/linear/asymmetric phase strategies.
- `pair_04_bright_resonant`: elevated treble energy plus narrow resonances and notches, useful for testing smoothing and boost restraint.
- `pair_05_asymmetric_room`: strongly different left/right room interaction, useful for stereo-link and robust target fitting behavior.

Notes:
- These are synthetic fixtures for Optuna prior exploration, not physically perfect loudspeaker models.
- The goal is to give Optuna a broader spread of plausible cases so filter-type defaults can be compared more systematically.
- Each file uses `freq_hz magnitude_db phase_deg` columns and can be loaded through the normal local path inputs.

Suggested usage:
1. Generate config variants:
   `python tools/build_synthetic_optuna_run_matrix.py`
2. Activate one generated config:
   `python tools/build_synthetic_optuna_run_matrix.py --activate pair_01_neutral_room__linear_phase.json`
3. Launch CamillaFIR normally and run AUTO mode.
4. Repeat for the remaining configs and let Optuna journal accumulate cross-case history.
