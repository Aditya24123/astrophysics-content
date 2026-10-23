"""Admission and overrun reporting, with tiny fixtures and an injected clock."""
import json
from pathlib import Path
import pytest

from hii_sim import dataset, cli
from hii_sim.io import atomic_json


@pytest.fixture
def config_path(tmp_path):
    config=json.loads(Path("configs/base.json").read_text())
    config["challenge_sources"]=0
    config["grid"]={"npix":16,"pixel_arcsec":10.,"subpixels":2}
    config["instrument"].update(frequencies_ghz=[1.31],bandwidths_mhz=[8.],min_baseline_m=300.)
    path=tmp_path/"config.json"
    atomic_json(path,config)
    return path


def test_initialization_time_counts_and_reports_overrun(tmp_path,config_path,monkeypatch):
    clock=[0.]
    monkeypatch.setattr(dataset.time,"perf_counter",lambda:clock[0])
    original=dataset.prepare
    def slow_prepare(*args):
        result=original(*args)
        clock[0]+=5.
        return result
    monkeypatch.setattr(dataset,"prepare",slow_prepare)
    def forbidden(*args):raise AssertionError("no parent may start after startup exhausts time")
    monkeypatch.setattr(dataset,"generate_source",forbidden)
    root=tmp_path/"bank"
    stats=dataset.generate(root,config_path,1,max_seconds=1,max_gib=1)
    assert stats["initialization_seconds"]==5. and stats["wall_seconds"]==5.
    assert stats["status"]=="budget_exceeded" and stats["new_sources"]==0
    assert not (root/"protected/generation.lock").exists()
    assert json.loads((root/"protected/last_run.json").read_text())==stats


@pytest.mark.parametrize("reservation",[{"source_headroom_seconds":121.},{"source_headroom_mib":1024.}])
def test_one_parent_headroom_prevents_admission(tmp_path,config_path,monkeypatch,reservation):
    def forbidden(*args):raise AssertionError("reserved parent must not start")
    monkeypatch.setattr(dataset,"generate_source",forbidden)
    stats=dataset.generate(tmp_path/"bank",config_path,1,max_seconds=120,max_gib=1,**reservation)
    assert stats["status"]=="bounded_checkpoint" and stats["new_sources"]==0
    assert all(stats["limits"][key]==value for key,value in reservation.items())


def test_parent_overrun_keeps_commit_and_is_not_reported_complete(tmp_path,config_path,monkeypatch):
    clock=[0.]
    monkeypatch.setattr(dataset.time,"perf_counter",lambda:clock[0])
    original=dataset.generate_source
    def slow_parent(*args):
        result=original(*args)
        clock[0]+=2.
        return result
    monkeypatch.setattr(dataset,"generate_source",slow_parent)
    root=tmp_path/"bank"
    stats=dataset.generate(root,config_path,1,max_seconds=1,max_gib=1)
    assert stats["status"]=="budget_exceeded" and stats["complete_sources"]==1
    plan=json.loads((root/"protected/plan.json").read_text())
    assert dataset.valid_commit(root,plan[0])
    assert not (root/"protected/generation.lock").exists()


@pytest.mark.parametrize("budget",[{"max_seconds":float("nan")},{"max_gib":0},
                                    {"source_headroom_mib":-1},{"source_headroom_seconds":True}])
def test_invalid_budget_rejected_before_initialization(tmp_path,config_path,budget):
    root=tmp_path/"absent"
    with pytest.raises(ValueError,match="budget"):
        dataset.generate(root,config_path,1,**budget)
    assert not root.exists()


def test_cli_forwards_reservations_and_fails_on_overrun(monkeypatch,capsys):
    calls=[]
    def overrun(*args,**kwargs):
        calls.append((args,kwargs))
        return {"status":"budget_exceeded"}
    monkeypatch.setattr(cli,"generate",overrun)
    with pytest.raises(SystemExit) as error:
        cli.main(["generate","--out","unused","--sources","1",
                  "--source-headroom-seconds","7","--source-headroom-mib","3"])
    assert error.value.code==1
    assert calls[0][1]=={"source_headroom_seconds":7.,"source_headroom_mib":3.}
    assert json.loads(capsys.readouterr().out)["status"]=="budget_exceeded"
