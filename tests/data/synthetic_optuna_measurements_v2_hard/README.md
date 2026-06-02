# Synthetic Optuna Measurement Set v2 Realistic/Hard

This folder is the harder follow-up set for Optuna prior experiments.

The goal is not to be physically perfect. The goal is to stop AUTO mode from trivially saturating to `100/100` and force more realistic tradeoffs.

What is different compared to v1:
- deeper and partially uncorrectable low-frequency nulls
- more overlapping room modes
- stronger crossover-region interference
- higher-Q treble resonances and comb-like roughness
- larger left/right asymmetry
- deterministic measurement-like roughness so the curves are less "clean"

Pairs:
- `pair_01_low_end_trap`: stacked bass modes and deep LF nulls
- `pair_02_crossover_disaster`: crossover interference with multiple nearby dips
- `pair_03_treble_hostile`: bright/resonant top end with combing
- `pair_04_room_asymmetry`: strong L/R mismatch in both bass and top-end tilt
- `pair_05_noisy_compromise`: broadband roughness and compromise-only correction case

Suggested usage:
1. Generate or refresh the dataset:
   `python tools/generate_synthetic_optuna_measurements.py --dataset hard`
2. Build run configs for the hard dataset:
   `python tools/build_synthetic_optuna_run_matrix.py --dataset synthetic_optuna_measurements_v2_hard`
3. Activate one config:
   `python tools/build_synthetic_optuna_run_matrix.py --dataset synthetic_optuna_measurements_v2_hard --activate pair_01_low_end_trap__linear_phase.json`
4. Run CamillaFIR normally and let Optuna accumulate history.
