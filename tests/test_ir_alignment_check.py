import numpy as np

from decaycore.dsp.ir_alignment_check import run_ir_alignment_check


def _pad_signal(values: np.ndarray, *, total: int = 128, left: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    right = max(0, int(total) - int(left) - int(values.size))
    return np.pad(values, (int(left), right), mode="constant")


def test_ir_alignment_does_not_flag_polarity_from_peak_sign_only():
    left = _pad_signal(
        np.array(
            [
                0.0334, 0.0290, 0.0555, 0.1271, 0.0899, 0.1936, 0.1887, 0.2015,
                0.1872, 0.1843, 0.2344, 0.2305, 0.2972, 0.3860, 0.5048, 0.5548,
                0.6811, 0.7240, 0.8361, 0.8520, 0.8488, 0.8256, 0.8197, 0.7174,
                0.6718, 0.5444, 0.4060, 0.3972, -0.8632, 0.2659, 0.2505, 0.2293,
                0.1467, 0.1675, 0.0874, 0.1568, 0.0782, 0.0740, 0.0376, 0.0664,
                0.0303,
            ]
        )
    )
    right = _pad_signal(
        np.array(
            [
                0.0555, -0.0085, 0.0345, 0.1026, 0.1527, 0.1595, 0.0981, 0.1545,
                0.1375, 0.2471, 0.2280, 0.2479, 0.3371, 0.4309, 0.4505, 0.5625,
                0.6643, 0.7322, 0.8273, 0.8803, 0.8593, 0.8831, 0.8228, 0.8309,
                0.6809, 0.5821, 0.4725, 0.3164, 0.2356, 0.2824, 0.2307, 0.2185,
                0.1778, 0.1878, 0.1605, 0.1240, 0.0553, 0.0722, 0.0878, 0.0210,
                0.0485,
            ]
        )
    )

    stats = run_ir_alignment_check(left, 48_000, right, 48_000)

    assert bool(stats["ir_align_xcorr_polarity_flip"]) is False
    assert bool(stats["ir_align_polarity_inverted"]) is False
    assert 0.0 <= float(stats["ir_align_xcorr_confidence"]) <= 1.0


def test_ir_alignment_normalizes_xcorr_confidence_and_detects_true_inversion():
    signal = _pad_signal(np.array([0.0, 0.15, 0.8, -0.3, 0.1, 0.04, 0.0]))
    inverted = -signal

    stats = run_ir_alignment_check(signal, 48_000, inverted, 48_000)

    assert bool(stats["ir_align_xcorr_polarity_flip"]) is True
    assert bool(stats["ir_align_polarity_inverted"]) is True
    assert float(stats["ir_align_xcorr_confidence"]) == 1.0
