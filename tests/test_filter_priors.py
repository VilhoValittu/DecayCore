import json

import numpy as np

from decaycore.auto_mode import filter_priors


def _measurements(offset: float = 0.0) -> dict:
    return {
        "f_l": np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float),
        "m_l": np.asarray([1.0 + offset, -2.0, 0.5, 0.0], dtype=float),
        "f_r": np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float),
        "m_r": np.asarray([0.8 + offset, -1.8, 0.4, -0.1], dtype=float),
    }


def _factory_payload() -> dict:
    return {
        "version": 2,
        "source": "factory",
        "filters": {
            "linear": {
                "filter_type": "Linear Phase",
                "auto_defaults": {
                    "filter_type_str": "Linear Phase",
                    "fdw_cycles": 7.0,
                    "max_boost_db": 4.0,
                    "enable_tdc": True,
                },
                "seed_preset": {
                    "fdw_cycles": 7.0,
                    "max_boost": 4.0,
                    "enable_tdc": True,
                },
            },
            "asym": {
                "filter_type": "Asymmetric (low-latency)",
                "auto_defaults": {
                    "filter_type_str": "Asymmetric (low-latency)",
                    "fdw_cycles": 6.0,
                    "max_boost_db": 3.5,
                    "enable_tdc": True,
                },
                "seed_preset": {
                    "fdw_cycles": 6.0,
                    "max_boost": 3.5,
                    "enable_tdc": True,
                },
            },
            "mixed": {
                "filter_type": "Mixed Phase",
                "auto_defaults": {
                    "filter_type_str": "Mixed Phase",
                    "fdw_cycles": 5.0,
                    "max_boost_db": 3.0,
                    "mixed_split_freq": 120.0,
                    "enable_tdc": True,
                },
                "seed_preset": {
                    "fdw_cycles": 5.0,
                    "max_boost": 3.0,
                    "mixed_freq": 120.0,
                    "enable_tdc": True,
                },
            },
        },
    }


def _write_json(path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_filter_priors_are_scoped_by_measurement_signature(monkeypatch, tmp_path):
    factory_path = tmp_path / "factory.json"
    user_path = tmp_path / "user.json"
    _write_json(factory_path, _factory_payload())

    m1 = _measurements(0.0)
    m2 = _measurements(0.25)
    sig1 = filter_priors._measurement_signature(m1)
    sig2 = filter_priors._measurement_signature(m2)
    _write_json(
        user_path,
        {
            "version": 3,
            "source": "auto_run",
            "measurements": {
                sig1: {
                    "filters": {
                        "linear": {
                            "filter_type": "Linear Phase",
                            "auto_defaults": {"fdw_cycles": 8.0},
                            "seed_preset": {"fdw_cycles": 8.0},
                        }
                    }
                },
                sig2: {
                    "filters": {
                        "linear": {
                            "filter_type": "Linear Phase",
                            "auto_defaults": {"fdw_cycles": 9.0},
                            "seed_preset": {"fdw_cycles": 9.0},
                        }
                    }
                },
            },
        },
    )
    monkeypatch.setattr(filter_priors, "_priors_path", lambda: factory_path)
    monkeypatch.setattr(filter_priors, "_user_priors_path", lambda: user_path)
    filter_priors.clear_auto_mode_filter_priors_cache()

    assert filter_priors.get_auto_mode_filter_seed_preset("Linear Phase", measurements=m1)["fdw_cycles"] == 8
    assert filter_priors.get_auto_mode_filter_seed_preset("Linear Phase", measurements=m2)["fdw_cycles"] == 9
    assert filter_priors.get_auto_mode_filter_auto_defaults("Linear Phase", measurements=m1)["max_boost_db"] == 4


def test_filter_priors_cache_and_force_reload(monkeypatch, tmp_path):
    factory_path = tmp_path / "factory.json"
    user_path = tmp_path / "user.json"
    _write_json(factory_path, _factory_payload())
    _write_json(user_path, {"version": 3, "measurements": {}})
    monkeypatch.setattr(filter_priors, "_priors_path", lambda: factory_path)
    monkeypatch.setattr(filter_priors, "_user_priors_path", lambda: user_path)
    filter_priors.clear_auto_mode_filter_priors_cache()

    first = filter_priors.load_auto_mode_filter_priors()
    _write_json(factory_path, {"version": 2, "filters": {}})
    cached = filter_priors.load_auto_mode_filter_priors()
    reloaded = filter_priors.load_auto_mode_filter_priors(force_reload=True)

    assert "linear" in first["filters"]
    assert "linear" in cached["filters"]
    assert "linear" not in reloaded["filters"]


def test_filter_prior_lookup_cache_clears_on_force_reload(monkeypatch, tmp_path):
    factory_path = tmp_path / "factory.json"
    user_path = tmp_path / "user.json"
    _write_json(factory_path, _factory_payload())
    measurements = _measurements()
    sig = filter_priors._measurement_signature(measurements)
    _write_json(
        user_path,
        {
            "version": 3,
            "measurements": {
                sig: {
                    "filters": {
                        "linear": {
                            "filter_type": "Linear Phase",
                            "seed_preset": {"fdw_cycles": 8.0},
                        }
                    }
                }
            },
        },
    )
    monkeypatch.setattr(filter_priors, "_priors_path", lambda: factory_path)
    monkeypatch.setattr(filter_priors, "_user_priors_path", lambda: user_path)
    filter_priors.clear_auto_mode_filter_priors_cache()

    first = filter_priors.get_auto_mode_filter_seed_preset("Linear Phase", measurements=measurements)
    _write_json(user_path, {"version": 3, "measurements": {}})
    filter_priors.load_auto_mode_filter_priors(force_reload=True)
    second = filter_priors.get_auto_mode_filter_seed_preset("Linear Phase", measurements=measurements)

    assert first["fdw_cycles"] == 8
    assert second["fdw_cycles"] == 7


def test_missing_measurement_prior_falls_back_to_factory(monkeypatch, tmp_path):
    factory_path = tmp_path / "factory.json"
    user_path = tmp_path / "user.json"
    _write_json(factory_path, _factory_payload())
    _write_json(user_path, {"version": 3, "measurements": {}})
    monkeypatch.setattr(filter_priors, "_priors_path", lambda: factory_path)
    monkeypatch.setattr(filter_priors, "_user_priors_path", lambda: user_path)
    filter_priors.clear_auto_mode_filter_priors_cache()

    seed = filter_priors.get_auto_mode_filter_seed_preset("Linear Phase", measurements=_measurements())

    assert seed["fdw_cycles"] == 7
    assert seed["max_boost"] == 4


def test_filter_prior_update_writes_user_data_not_factory(monkeypatch, tmp_path):
    factory_path = tmp_path / "factory.json"
    user_path = tmp_path / "user.json"
    factory = _factory_payload()
    _write_json(factory_path, factory)
    monkeypatch.setattr(filter_priors, "_priors_path", lambda: factory_path)
    monkeypatch.setattr(filter_priors, "_user_priors_path", lambda: user_path)
    filter_priors.clear_auto_mode_filter_priors_cache()

    measurements = _measurements()
    sig = filter_priors._measurement_signature(measurements)
    filter_priors.update_auto_mode_filter_priors_from_winner(
        "Linear Phase",
        {"fdw_cycles": 8.25, "max_boost": 5.5},
        {"filter_type": "Linear Phase", "enable_tdc": False},
        measurements=measurements,
    )

    assert json.loads(factory_path.read_text(encoding="utf-8")) == factory
    user = json.loads(user_path.read_text(encoding="utf-8"))
    entry = user["measurements"][sig]["filters"]["linear"]
    assert entry["seed_preset"]["fdw_cycles"] == 8.25
    assert entry["seed_preset"]["max_boost"] == 5.5
    assert entry["seed_preset"]["enable_tdc"] is False
    assert entry["auto_defaults"]["max_boost_db"] == 5.5
    assert entry["auto_defaults"]["enable_tdc"] is False
    assert set(user["measurements"][sig]["filters"]) == {"asym", "linear", "mixed"}
    assert user["measurements"][sig]["filters"]["asym"]["seed_preset"]["fdw_cycles"] == 6
    assert user["measurements"][sig]["filters"]["mixed"]["seed_preset"]["mixed_freq"] == 120


def test_filter_prior_update_uses_known_template_when_factory_missing(monkeypatch, tmp_path):
    factory_path = tmp_path / "missing_factory.json"
    user_path = tmp_path / "user.json"
    monkeypatch.setattr(filter_priors, "_priors_path", lambda: factory_path)
    monkeypatch.setattr(filter_priors, "_user_priors_path", lambda: user_path)
    filter_priors.clear_auto_mode_filter_priors_cache()

    measurements = _measurements()
    filter_priors.update_auto_mode_filter_priors_from_winner(
        "Asymmetric (low-latency)",
        {
            "fdw_cycles": 8.0,
            "max_boost": 4.5,
            "phase_limit": 110.0,
        },
        {
            "filter_type": "Asymmetric (low-latency)",
            "enable_tdc": True,
            "enable_afdw": True,
            "comparison_mode": True,
            "hpf_enable": True,
        },
        measurements=measurements,
    )

    user = json.loads(user_path.read_text(encoding="utf-8"))
    sig = filter_priors._measurement_signature(measurements)
    entry = user["measurements"][sig]["filters"]["asym"]
    assert set(user["measurements"][sig]["filters"]) == {"asym"}
    assert entry["seed_preset"]["fdw_cycles"] == 8
    assert entry["seed_preset"]["max_boost"] == 4.5
    assert entry["seed_preset"]["phase_limit"] == 110
    assert entry["auto_defaults"]["max_boost_db"] == 4.5
    assert entry["auto_defaults"]["filter_type_str"] == "Asymmetric (low-latency)"
    assert entry["auto_defaults"]["enable_tdc"] is True


def test_filter_priors_read_legacy_v2_user_filters(monkeypatch, tmp_path):
    factory_path = tmp_path / "factory.json"
    user_path = tmp_path / "user.json"
    _write_json(factory_path, _factory_payload())
    _write_json(
        user_path,
        {
            "version": 2,
            "source": "legacy",
            "filters": {
                "linear": {
                    "filter_type": "Linear Phase",
                    "auto_defaults": {"fdw_cycles": 10.0},
                    "seed_preset": {"fdw_cycles": 10.0},
                }
            },
        },
    )
    monkeypatch.setattr(filter_priors, "_priors_path", lambda: factory_path)
    monkeypatch.setattr(filter_priors, "_user_priors_path", lambda: user_path)
    filter_priors.clear_auto_mode_filter_priors_cache()

    assert filter_priors.get_auto_mode_filter_seed_preset("Linear Phase")["fdw_cycles"] == 10


def test_auto_search_prior_seed_uses_measurement_scope(monkeypatch, tmp_path):
    from decaycore.auto_mode import api  # noqa: F401
    from decaycore.auto_mode.search_v2.seeds import _apply_explicit_seed

    factory_path = tmp_path / "factory.json"
    user_path = tmp_path / "user.json"
    _write_json(factory_path, _factory_payload())
    measurements = _measurements(0.5)
    sig = filter_priors._measurement_signature(measurements)
    _write_json(
        user_path,
        {
            "version": 3,
            "measurements": {
                sig: {
                    "filters": {
                        "linear": {
                            "filter_type": "Linear Phase",
                            "seed_preset": {"fdw_cycles": 11.0},
                        }
                    }
                }
            },
        },
    )
    monkeypatch.setattr(filter_priors, "_priors_path", lambda: factory_path)
    monkeypatch.setattr(filter_priors, "_user_priors_path", lambda: user_path)
    filter_priors.clear_auto_mode_filter_priors_cache()

    seed = _apply_explicit_seed(
        search_base_data={"filter_type": "Linear Phase"},
        cache_base_data={},
        measurements=measurements,
    )

    assert seed["fdw_cycles"] == 11
