import decaycore.config.decaycore_config as decaycore_config
from decaycore.ui.decaycore_modes import MODE_DEFAULTS


def test_first_launch_default_filter_type_is_asymmetric(monkeypatch, tmp_path):
    monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(tmp_path / "missing_config.json"))
    cfg = decaycore_config.load_config()
    assert str(cfg["filter_type"]) == "Asymmetric"


def test_all_mode_defaults_use_asymmetric_filter():
    for mode in ("BASIC", "ADVANCED", "AUTO"):
        assert str(MODE_DEFAULTS[mode]["filter_type_str"]) == "Asymmetric (low-latency)"
