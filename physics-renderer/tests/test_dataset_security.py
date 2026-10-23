"""Regression attacks and integrity failures, using tiny independently grouped sets."""
import json
from pathlib import Path
import numpy as np
import pytest

from hii_sim import dataset
from hii_sim.dataset import generate, make_plan, prepare, subset_membership, valid_commit
from hii_sim.export import export_v2
from hii_sim.io import atomic_json, canonical_hash, digest, write_npz
from hii_sim.loaders import PredictorDataset, PairedDataset


@pytest.fixture
def config_path(tmp_path):
    c=json.loads(Path("configs/base.json").read_text())
    c["challenge_sources"]=0
    c["grid"]={"npix":16,"pixel_arcsec":10.,"subpixels":2}
    c["instrument"]["frequencies_ghz"]=[1.31,1.82]
    c["instrument"]["bandwidths_mhz"]=[128.,128.]
    c["instrument"]["min_baseline_m"]=300.
    path=tmp_path/"config.json"; atomic_json(path,c)
    return path


def read(path):
    return json.loads(Path(path).read_text())


def recover_by_public_order(public_subsets, known_seed_plan):
    """The actual original attack: zip full public lists against same-split ranks."""
    recovered={}
    for split,levels in public_subsets.items():
        if not levels:
            continue
        ids=levels[str(max(map(int,levels)))]
        predicted=[x for x in known_seed_plan if x["split"]==split]
        recovered.update({sid:guess["source"] for sid,guess in zip(ids,predicted)})
    return recovered


def test_reproduce_original_attack_and_block_v2(tmp_path,config_path):
    requested=read(config_path)
    legacy=make_plan(requested,64,id_salt="ab"*32)
    public_seed_plan=make_plan(requested,64)
    recovered=recover_by_public_order(subset_membership(legacy),public_seed_plan)
    assert len(recovered)==64
    assert all(recovered[item["source_id"]]==item["source"] for item in legacy)

    root=tmp_path/"v2"
    generate(root,config_path,64,max_seconds=120,max_gib=1)
    actual=read(root/"protected/plan.json")
    public=read(root/"public/subsets.json")
    assert public["schema_version"]=="2.0"
    for split,levels in public["nested_source_ids"].items():
        private=subset_membership(actual)[split]
        for count,ids in levels.items():
            assert ids==sorted(ids)  # opaque-ID ordering only
            assert set(ids)==set(private[count])
    attacked=recover_by_public_order(public["nested_source_ids"],public_seed_plan)
    assert sum(attacked.get(x["source_id"])==x["source"] for x in actual)==0
    private_seed=read(root/"protected/provenance.json")["config"]["seed"]
    assert private_seed!=requested["seed"]
    saved_config=read(root/"protected/provenance.json")["config"]
    saved_key=read(root/"protected/id_key.json")["id_salt"]
    assert make_plan(saved_config,64,saved_key)==actual
    for p in (root/"public").rglob("*.json"):
        text=p.read_text()
        assert str(private_seed) not in text
        for key in ('"source_index":','"seed":','"sampling_seed":','"parameters":'):
            assert key not in text
    predictor=PredictorDataset(root/"public",verify=True)
    paired=PairedDataset(root,verify=True)
    inputs=predictor[0]
    assert set(inputs)=={"observations","metadata","bookkeeping"}
    assert set(paired[0][1]["parameters"])==set(requested["reported_targets"])
    assert "parameters" not in inputs["metadata"]
    assert all(Path(r[k]).stem==r["realization_id"] for r in predictor.records for k in ("image","measurement"))


def test_interrupt_then_resume_retains_history_and_groups(tmp_path,config_path,monkeypatch):
    root=tmp_path/"interrupt"
    original=dataset.generate_source
    calls=0
    def interrupted(*args):
        nonlocal calls
        calls+=1
        if calls==2:
            (root/"public/observations/orphan.tmp").write_bytes(b"uncommitted")
            raise KeyboardInterrupt("injected after first committed physical source")
        return original(*args)
    monkeypatch.setattr(dataset,"generate_source",interrupted)
    with pytest.raises(KeyboardInterrupt):
        generate(root,config_path,4,max_seconds=120,max_gib=1)
    assert not (root/"protected/generation.lock").exists()
    assert len(list((root/"protected/commits").glob("*.json")))==1
    first_commit=next((root/"protected/commits").glob("*.json"))
    before=digest(first_commit)
    monkeypatch.setattr(dataset,"generate_source",original)
    result=generate(root,config_path,4,max_seconds=120,max_gib=1)
    assert result["status"]=="complete" and result["resumed_sources"]==1
    assert digest(first_commit)==before
    history=[read(p) for p in (root/"protected/runs").glob("*.json")]
    assert sorted(x["status"] for x in history)==["complete","interrupted"]
    records=read(root/"public/manifest.json")["records"]
    assert len(records)==8
    for sid in {r["source_id"] for r in records}:
        related=[r for r in records if r["source_id"]==sid]
        assert len(related)==2 and len({r["split"] for r in related})==1
    assert not any("orphan" in str(r) for r in records)


@pytest.mark.parametrize("change",["config","code","runtime","count"])
def test_incompatible_resume_is_rejected_without_writes(tmp_path,config_path,monkeypatch,change):
    root=tmp_path/"resume"
    prepare(root,config_path,4)
    before={p.relative_to(root).as_posix():digest(p) for p in root.rglob("*") if p.is_file()}
    n=4
    if change=="config":
        config=read(config_path); config["seed"]+=1; atomic_json(config_path,config)
    elif change=="code":
        monkeypatch.setattr(dataset,"code_digest",lambda:"changed-code")
    elif change=="runtime":
        monkeypatch.setattr(dataset,"versions",lambda:{"python":"changed-runtime"})
    else:
        n=5
    with pytest.raises(ValueError,match="Resume refused"):
        prepare(root,config_path,n)
    after={p.relative_to(root).as_posix():digest(p) for p in root.rglob("*") if p.is_file()}
    assert before==after


@pytest.mark.parametrize("array_name",["weights","primary_beam","psf"])
def test_shared_instrument_tampering_rejected(tmp_path,config_path,array_name):
    root=tmp_path/array_name
    generate(root,config_path,1,max_seconds=120,max_gib=1)
    shared=root/"public/instruments/base.npz"
    with np.load(shared) as f:
        arrays={k:f[k].copy() for k in f.files}
    arrays[array_name].flat[0]+=1.
    write_npz(shared,**arrays)
    with pytest.raises(RuntimeError,match="Checksum mismatch"):
        PredictorDataset(root/"public",verify=True)
    with pytest.raises(RuntimeError,match="Checksum mismatch"):
        prepare(root,config_path,1)


def test_missing_checksum_coverage_rejected(tmp_path,config_path):
    root=tmp_path/"coverage"
    generate(root,config_path,1,max_seconds=120,max_gib=1)
    item=read(root/"protected/plan.json")[0]
    path=root/"protected/commits"/(item["source_id"]+".json")
    commit=read(path)
    del commit["checksums"]["public/"+commit["records"][0]["measurement"]]
    atomic_json(path,commit)
    with pytest.raises(RuntimeError,match="Incomplete checksum"):
        valid_commit(root,item)


def test_versioned_export_rotates_ids_preserves_membership_and_source(tmp_path,config_path):
    source=tmp_path/"old"; target=tmp_path/"export_v2"
    config=read(config_path); config["realizations_per_source"]=3; atomic_json(config_path,config)
    generate(source,config_path,8,max_seconds=120,max_gib=1)
    # Represent the precise original public ordering flaw on a small fixture.
    plan=read(source/"protected/plan.json")
    manifest=read(source/"public/manifest.json"); manifest["schema_version"]="1.0"
    atomic_json(source/"public/manifest.json",manifest)
    atomic_json(source/"public/subsets.json",{"schema_version":"1.0","nested_source_ids":subset_membership(plan)})
    before={p.relative_to(source).as_posix():digest(p) for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError,match="Legacy"):
        PredictorDataset(source/"public")
    assert export_v2(source,target)["sources"]==8
    after={p.relative_to(source).as_posix():digest(p) for p in source.rglob("*") if p.is_file()}
    assert before==after
    mapping=read(target/"protected/export_mapping.json")["source_ids"]
    assert not (set(mapping)&set(mapping.values()))
    before_subsets=read(source/"public/subsets.json")["nested_source_ids"]
    after_subsets=read(target/"public/subsets.json")["nested_source_ids"]
    for split,levels in before_subsets.items():
        for size,ids in levels.items():
            assert set(after_subsets[split][size])=={mapping[x] for x in ids}
    assert len(PairedDataset(target,verify=True))==24
    PairedDataset(target,verify=True)[0]
    for record in read(target/"public/manifest.json")["records"]:
        assert record["source_id"] in mapping.values()
    with pytest.raises(ValueError,match="outside"):
        export_v2(source,source/"other")


def test_predictor_rejects_arbitrary_metadata_and_traversal(tmp_path,config_path):
    root=tmp_path/"metadata"
    generate(root,config_path,1,max_seconds=120,max_gib=1)
    path=root/"public/instruments/base.json"
    meta=read(path); meta["parameters"]={"te_k":8000}; atomic_json(path,meta)
    with pytest.raises(ValueError,match="potential protected"):
        PredictorDataset(root/"public")[0]
    manifest=read(root/"public/manifest.json")
    manifest["records"][0]["measurement"]="../protected/plan.json"
    atomic_json(root/"public/manifest.json",manifest)
    with pytest.raises(ValueError,match="escapes"):
        PredictorDataset(root/"public")


def test_protected_plan_tampering_and_partial_initialization_rejected(tmp_path,config_path):
    root=tmp_path/"plan"
    prepare(root,config_path,1)
    plan=read(root/"protected/plan.json")
    plan[0]["source"]["te_k"]+=1
    atomic_json(root/"protected/plan.json",plan)
    with pytest.raises(RuntimeError,match="Checksum mismatch"):
        prepare(root,config_path,1)
    partial=tmp_path/"partial"
    partial.mkdir(); (partial/".initializing").write_text("interrupted")
    with pytest.raises(ValueError,match="Nonempty output"):
        prepare(partial,config_path,1)


@pytest.mark.parametrize("artifact",["plan","provenance"])
def test_export_rejects_changed_protected_provenance(tmp_path,config_path,artifact):
    source=tmp_path/"source"
    generate(source,config_path,2,max_seconds=120,max_gib=1)
    if artifact=="plan":
        path=source/"protected/plan.json"; value=read(path); value[0]["source"]["te_k"]+=1
    else:
        path=source/"protected/provenance.json"; value=read(path); value["config"]["seed"]+=1
    atomic_json(path,value)
    with pytest.raises(RuntimeError,match="[Cc]hecksum mismatch"):
        export_v2(source,tmp_path/"rejected")
    assert not (tmp_path/"rejected").exists()


def test_export_uses_anchored_membership_and_legacy_validates_nesting(tmp_path,config_path):
    source=tmp_path/"source"
    generate(source,config_path,4,max_seconds=120,max_gib=1)
    original=read(source/"protected/subset_membership.json")
    # Public indices are rebuildable; v2 export trusts the checksum-anchored
    # protected membership instead of a modified public subset file.
    altered=read(source/"public/subsets.json")
    altered["nested_source_ids"]={"train":{},"validation":{},"test":{}}
    atomic_json(source/"public/subsets.json",altered)
    target=tmp_path/"reindexed"
    export_v2(source,target)
    mapping=read(target/"protected/export_mapping.json")["source_ids"]
    revised=read(target/"public/subsets.json")["nested_source_ids"]
    for split,levels in original.items():
        for n,ids in levels.items():
            assert set(revised[split][n])=={mapping[x] for x in ids}
    # A real v1 source has no anchored private membership. Validate its public
    # membership against its plan's documented prefix, groups and nesting.
    provenance=read(source/"protected/provenance.json")
    legacy={"config":provenance["config"],"n_sources":4,"code_sha256":"legacy-no-snapshot"}
    legacy["spec_sha256"]=canonical_hash(legacy)
    atomic_json(source/"protected/provenance.json",legacy)
    altered["nested_source_ids"]["train"]={"1":["f"*24]}
    atomic_json(source/"public/subsets.json",altered)
    with pytest.raises(ValueError,match="subset"):
        export_v2(source,tmp_path/"bad_legacy")


def test_public_subset_integrity_and_membership(tmp_path,config_path):
    root=tmp_path/"subset"
    generate(root,config_path,4,max_seconds=120,max_gib=1)
    path=root/"public/subsets.json"
    subset=read(path)
    subset["nested_source_ids"]["train"]={"1":["f"*24]}
    atomic_json(path,subset)
    with pytest.raises(RuntimeError,match="Checksum mismatch: subsets"):
        PredictorDataset(root/"public",verify=True)
    with pytest.raises(ValueError,match="Invalid public subset"):
        PredictorDataset(root/"public",verify=False)
