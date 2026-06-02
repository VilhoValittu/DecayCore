import numpy as np

from decaycore.config.results import FilterResult
from decaycore.ui import decaycore_export as export
from decaycore.ui.ng_results_sections import _build_auto_polish_lines


def _result(fs: int, left_score: float, right_score: float, boost_db: float = 0.0, events: int = 0) -> FilterResult:
    zeros = np.zeros(8, dtype=float)
    refs = [{} for _ in range(int(events))]
    return FilterResult(
        fs=fs,
        taps=2048,
        l_ir=zeros,
        r_ir=zeros,
        l_mag=zeros,
        r_mag=zeros,
        l_phase=zeros,
        r_phase=zeros,
        freq_axis=zeros,
        l_st={"score": float(left_score), "net_boost_peak_db": float(boost_db), "reflections": list(refs)},
        r_st={"score": float(right_score), "net_boost_peak_db": float(boost_db), "reflections": list(refs)},
    )


def test_export_winner_summary_uses_official_rank_score():
    data = {
        "camillafir_automatic_mode": True,
        "_auto_mode_meta": {
            "trials_ok": 4,
            "trials_total": 6,
            "search_fs": 44100,
            "search_taps": 2048,
            "best_metrics": {
                "rank_score": 69.505,
                "rank_score_official": 91.264,
                "rank_score_components": {"rank_score": 91.264, "context": {"score_kind": "preset_objective_score"}},
                "avg_score": 81.700,
                "dsp_penalty": 0.80,
                "exc_penalty": 0.20,
                "bass_boost_20_200_db": 1.75,
                "max_net_boost_db": 2.40,
                "events_total": 1,
                "events_severity": 0.25,
            },
            "best_preset": {},
        },
    }

    text = export._append_dsp_effective_params("", data, 44100)

    assert export._export_winner_rank_score(data) == 91.264
    assert "Score breakdown: 91.264/100" in text
    assert "bass_boost=1.75 dB" in text
    assert "Best rank score: 69.505/100" not in text


def test_export_winner_summary_reports_bass_integration_hard_gate():
    data = {
        "camillafir_automatic_mode": True,
        "bass_integration_enable": True,
        "_auto_mode_meta": {
            "trials_ok": 4,
            "trials_total": 6,
            "search_fs": 44100,
            "search_taps": 2048,
            "best_metrics": {
                "rank_score": 66.0,
                "rank_score_official": 82.0,
                "rank_score_components": {"rank_score": 66.0, "context": {"score_kind": "preset_objective_score"}},
                "avg_score": 80.0,
                "bass_integration_hard_gate_failed": True,
                "bass_integration_hard_gate_reason": "Right channel remains limiting.",
                "hard_gate_failures": ["bass_integration_infeasible_hard_gate"],
                "bass_feasibility_class": "infeasible",
                "bass_feasibility_reason": "Right channel remains limiting.",
                "bass_cancellation_risk": 0.24,
                "bass_overlap_ripple": 8.5,
                "bass_sub_dominance": 9.0,
            },
            "best_preset": {},
        },
    }

    text = export._append_dsp_effective_params("", data, 44100)

    assert "Auto-mode safety gate: Bass Integration infeasible: Right channel remains limiting." in text


def test_export_run_ranking_uses_distinct_label(monkeypatch):
    monkeypatch.setattr(
        export.plots,
        "calc_ai_summary_from_stats",
        lambda stats: {"score": float(dict(stats or {}).get("score", 0.0))},
    )

    ranking = export._build_export_ranking(
        [
            _result(44100, 82.0, 81.0, boost_db=0.5, events=0),
            _result(48000, 78.0, 77.5, boost_db=3.5, events=2),
        ]
    )

    winner = dict(ranking["entries"][0])
    block = export._append_export_ranking("", 44100, ranking)

    assert winner["run_ranking_score"] == winner["run_ranking_score_components"]["rank_score"]
    assert "run_ranking_score=" in block
    assert "rank_score=" not in block
    assert "separate from automatic-mode Best rank score" in block


def test_export_winner_summary_includes_lf_winner_polish_lines():
    data = {
        "camillafir_automatic_mode": True,
        "_auto_mode_meta": {
            "trials_ok": 4,
            "trials_total": 6,
            "search_fs": 44100,
            "search_taps": 2048,
            "best_metrics": {
                "rank_score": 91.264,
                "rank_score_official": 91.264,
                "rank_score_components": {"rank_score": 91.264, "context": {"score_kind": "preset_objective_score"}},
                "avg_score": 81.700,
            },
            "best_preset": {},
            "low_bass_cut_winner_polish": {
                "applicable": True,
                "applied": True,
                "start_low_bass_cut_hz": 34.0,
                "final_low_bass_cut_hz": 36.0,
                "rank_before": 38.690,
                "rank_after": 46.150,
                "tested_low_bass_cut_hz": [32.0, 36.0, 38.0],
            },
            "hpf_winner_polish": {
                "applicable": True,
                "applied": True,
                "start_enabled": True,
                "start_freq_hz": 27.5,
                "start_slope_db_oct": 24,
                "final_enabled": True,
                "final_freq_hz": 24.5,
                "final_slope_db_oct": 24,
                "rank_before": 46.150,
                "rank_after": 46.168,
                "tested_candidates": [
                    {"label": "HPF off"},
                    {"label": "HPF 24.5 Hz/24 dB/oct"},
                    {"label": "HPF 30.5 Hz/24 dB/oct"},
                ],
            },
        },
    }

    text = export._append_dsp_effective_params("", data, 44100)

    assert "Low-bass-cut winner polish: applied (34.0 -> 36.0 Hz, rank 68.326 -> 76.650, tested [32.0, 36.0, 38.0] Hz)" in text
    assert "HPF winner polish: applied (HPF 27.5 Hz/24 dB/oct -> HPF 24.5 Hz/24 dB/oct, rank 76.650 -> 76.668, tested [HPF off, HPF 24.5 Hz/24 dB/oct, HPF 30.5 Hz/24 dB/oct])" in text


def test_results_polish_lines_use_display_rank_system():
    lines = _build_auto_polish_lines(
        {
            "hpf_winner_polish": {
                "applicable": True,
                "applied": True,
                "start_enabled": True,
                "start_freq_hz": 18.0,
                "start_slope_db_oct": 24,
                "final_enabled": False,
                "final_freq_hz": 18.0,
                "final_slope_db_oct": 24,
                "rank_before": 38.690,
                "rank_after": 46.150,
            },
            "excess_phase_strength_winner_polish": {
                "applicable": True,
                "applied": True,
                "start_value": 0.9,
                "final_value": 0.8,
                "rank_before": 46.150,
                "rank_after": 46.168,
            },
        }
    )

    text = "\n".join(lines)
    assert "rank 68.326" in text
    assert "rank 76.650" in text
    assert "→ 76.668" in text
    assert "rank 38.690" not in text
    assert "rank 46.150" not in text


def test_export_winner_summary_hides_exc_penalty_text_for_optuna():
    data = {
        "camillafir_automatic_mode": True,
        "_auto_exc_seed_freq_hz": 31.5,
        "_auto_exc_freq_hz": 31.5,
        "_auto_mode_meta": {
            "optimizer_backend": "optuna",
            "trials_ok": 4,
            "trials_total": 6,
            "search_fs": 44100,
            "search_taps": 2048,
            "auto_exc_seed_freq_hz": 31.5,
            "best_auto_exc_freq_hz": 31.5,
            "best_metrics": {
                "rank_score": 91.264,
                "rank_score_official": 91.264,
                "rank_score_components": {"rank_score": 91.264, "context": {"score_kind": "preset_objective_score"}},
                "avg_score": 81.700,
                "dsp_penalty": 0.80,
                "exc_penalty": 0.20,
                "bass_boost_20_200_db": 1.75,
                "max_net_boost_db": 2.40,
                "events_total": 1,
                "events_severity": 0.25,
            },
            "best_preset": {},
        },
    }

    text = export._append_dsp_effective_params("", data, 44100)

    assert "exc_pen=" not in text
    assert "Excursion protection (AUTO):" not in text
