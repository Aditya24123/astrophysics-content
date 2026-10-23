import json
from pathlib import Path
import pytest

from hii_sim import cli, audit
from hii_sim.config_defaults import DEFAULT_CONFIG, LEGACY_CONFIG
from hii_sim.schema import load_config, Instrument, Grid
from hii_sim.instrument import operator, coverage_diagnostics


@pytest.mark.parametrize("command,tail", [
    ("generate", ["--out", "unused", "--sources", "1"]),
    ("plan", ["--sources", "1"]),
    ("validate", ["--out", "unused"]),
])
def test_cli_defaults_to_guarded_compact_scope(command, tail):
    args = cli.build_parser().parse_args([command, *tail])
    assert args.config == DEFAULT_CONFIG
    config = load_config(args.config)
    inst = Instrument(**config["instrument"])
    assert inst.name == "compact_fourier_proxy_v2"
    assert inst.max_short_spacing_ratio == 1.5
    assert set(inst.bandwidths_mhz) == {8}
    report = coverage_diagnostics(Grid(**config["grid"]), inst)
    assert all(channel["actual_min_nonzero_baseline_m"] <=
               channel["requested_min_baseline_m"]*inst.max_short_spacing_ratio
               for channel in report["channels"])
    # Building the configured operator enforces the same sampling guard.
    operator(Grid(**config["grid"]), inst)


@pytest.mark.parametrize("command,tail", [
    ("generate", ["--out", "unused", "--sources", "1"]),
    ("plan", ["--sources", "1"]),
    ("validate", ["--out", "unused"]),
])
def test_cli_explicit_config_overrides_default(command, tail, tmp_path):
    config = json.loads(Path(LEGACY_CONFIG).read_text(encoding="utf-8"))
    explicit = tmp_path/"legacy.json"
    explicit.write_text(json.dumps(config), encoding="utf-8")
    args = cli.build_parser().parse_args([command, *tail, "--config", str(explicit)])
    assert args.config == str(explicit)
    inst = Instrument(**load_config(args.config)["instrument"])
    assert inst.max_short_spacing_ratio is None
    assert tuple(inst.bandwidths_mhz) == (128.,)*6


def test_generate_dispatch_uses_default_and_explicit_config(monkeypatch, capsys, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "generate", lambda *args,**kwargs: calls.append(args) or {"ok": True})
    cli.main(["generate", "--out", "unused", "--sources", "1"])
    assert calls[-1][1] == DEFAULT_CONFIG
    explicit = tmp_path/"chosen.json"
    explicit.write_text("{}", encoding="utf-8")
    cli.main(["generate", "--out", "unused", "--sources", "1", "--config", str(explicit)])
    assert calls[-1][1] == str(explicit)
    capsys.readouterr()


def test_plan_dispatch_uses_default_and_explicit_config(monkeypatch, capsys, tmp_path):
    paths = []
    monkeypatch.setattr(cli, "load_config", lambda path: paths.append(path) or {
        "challenge_sources": 0, "realizations_per_source": 1,
        "sampling": {"ranges": {key: [0, 1, "linear"] for key in
            ("em0_pc_cm6", "te_k", "major_fwhm_arcsec", "axis_ratio", "pa_deg")}},
        "reported_targets": ["te_k"]})
    monkeypatch.setattr(cli, "make_plan", lambda config, count: [
        {"split": "train", "source": {key: 0. for key in config["sampling"]["ranges"]}},
        {"split": "test", "source": {key: 1. for key in config["sampling"]["ranges"]}}])
    cli.main(["plan", "--sources", "2"])
    assert paths[-1] == DEFAULT_CONFIG
    explicit = tmp_path/"chosen.json"
    cli.main(["plan", "--sources", "2", "--config", str(explicit)])
    assert paths[-1] == str(explicit)
    capsys.readouterr()


def test_validate_dispatch_uses_default_and_explicit_config(monkeypatch, capsys, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "validate", lambda *args: calls.append(args) or {"ok": True})
    cli.main(["validate", "--out", "unused"])
    assert calls[-1][0] == DEFAULT_CONFIG
    explicit = tmp_path/"chosen.json"
    cli.main(["validate", "--out", "unused", "--config", str(explicit)])
    assert calls[-1][0] == str(explicit)
    capsys.readouterr()


def test_audit_default_and_explicit_config_dispatch(monkeypatch, capsys, tmp_path):
    calls = []
    monkeypatch.setattr(audit, "audit", lambda *args: calls.append(args) or {"ok": True})
    audit.main(["--out", "unused"])
    assert calls[-1][0] == DEFAULT_CONFIG
    explicit = tmp_path/"chosen.json"
    explicit.write_text("{}", encoding="utf-8")
    audit.main(["--out", "unused", "--config", str(explicit)])
    assert calls[-1][0] == str(explicit)
    capsys.readouterr()
