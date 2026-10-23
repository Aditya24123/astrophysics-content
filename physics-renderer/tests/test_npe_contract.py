"""Tiny contract fixtures, not a release candidate or a noise qualification."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

from hii_sim.dataset import generate
from hii_sim.io import atomic_json, digest, write_npz
from hii_sim.loaders import NPEInputDataset, PairedNPEDataset, PairedDataset
from hii_sim.npe_contract import canonical_fourier_data, ThermalCovariance, CalibrationPrior


@pytest.fixture
def bank(tmp_path):
    def build(challenges=0, self_modes=False):
        c=json.loads(Path("configs/base.json").read_text())
        c.update(challenge_sources=challenges,reported_targets=["em0_pc_cm6","te_k","major_fwhm_arcsec","axis_ratio"])
        c["grid"]={"npix":16,"pixel_arcsec":10.,"subpixels":2}
        c["instrument"].update(frequencies_ghz=[1.31,1.82],bandwidths_mhz=[8.,8.],
                               min_baseline_m=0. if self_modes else 300., sampling_fraction=1. if self_modes else .5)
        config=tmp_path/"config.json";atomic_json(config,c)
        root=tmp_path/"bank"
        generate(root,config,2,max_seconds=120,max_gib=1)
        return root
    return build


def read(path):
    return json.loads(Path(path).read_text())


def feature_keys(value):
    if is_dataclass(value):value=asdict(value)
    if isinstance(value,dict):
        return set(value)|set().union(*(feature_keys(v) for v in value.values()))
    return set()


def test_public_only_features_do_not_open_images_or_protected_files(bank,tmp_path,monkeypatch):
    root=bank();public=tmp_path/"public_only"
    shutil.copytree(root/"public",public)
    for path in (public/"observations").glob("*.fits"):path.unlink()
    import hii_sim.loaders as loaders
    def no_image(*args):raise AssertionError("NPE must not open dirty/CLEAN images")
    monkeypatch.setattr(loaders,"read_image",no_image)
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert "protected" not in path.parts
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    view=NPEInputDataset(public)
    assert len(view)==4
    example=view[0]
    assert set(example)=={"features","bookkeeping"}
    assert set(example["features"])=={"measurements","observing"}
    assert not (feature_keys(example["features"])&{"profile","parameters","truth","dirty_image","source_id","split","seed","source_index"})
    metadata=example["features"]["observing"]
    assert metadata.taper_role=="imaging_weight_not_inverse_variance"
    assert "synthetic" in metadata.thermal.provenance and "not_true" in metadata.calibration.provenance
    with pytest.raises(ValueError):example["features"]["measurements"].values_jy[0]=0.


def test_canonical_packing_retains_all_and_only_independent_real_information():
    n=4
    image=np.arange(n*n,dtype=float).reshape(1,n,n)**2
    vis=np.fft.fft2(image)
    w=np.ones(vis.shape)
    packed=canonical_fourier_data(vis[w>0],w)
    assert len(packed.values_jy)==n*n  # Not twice that count.
    assert int(packed.self_conjugate.sum())==4
    assert not np.any(packed.self_conjugate & (packed.component==1))
    rebuilt=np.zeros_like(vis)
    for value,f,y,x,component in zip(packed.values_jy,packed.frequency_index,packed.y_index,packed.x_index,packed.component):
        rebuilt[f,y,x]+=value*(1 if component==0 else 1j)
    for f,y,x in set(zip(packed.frequency_index,packed.y_index,packed.x_index)):
        rebuilt[f,(-y)%n,(-x)%n]=rebuilt[f,y,x].conjugate()
    np.testing.assert_allclose(rebuilt,vis,rtol=0,atol=1e-12)
    bad=vis.copy();bad[0,0,1]+=1.j
    with pytest.raises(ValueError,match="Hermitian"):
        canonical_fourier_data(bad[w>0],w)


def test_loader_handles_self_modes_and_frequency_availability(bank):
    root=bank(self_modes=True)
    example=NPEInputDataset(root/"public")[0]["features"]
    values=example["measurements"];metadata=example["observing"]
    assert values.self_conjugate.sum()==8  # Four per even square band.
    assert len(values.values_jy)==2*16*16
    assert np.all(values.component[values.self_conjugate]==0)
    base=metadata.thermal.covariance([0,1])
    np.testing.assert_array_equal(metadata.thermal.covariance([0,1],self_conjugate=True),2*base)


def test_typed_covariance_and_shared_calibration_limiting_cases():
    thermal=ThermalCovariance(np.array([2.,3.]),.25)
    np.testing.assert_array_equal(thermal.covariance(),[[4.,1.5],[1.5,9.]])
    assert thermal.covariance_unit=="Jy^2"
    with pytest.raises(ValueError):thermal.covariance([0,0])
    common=CalibrationPrior(np.array([1.,2.,4.]),.05,0.)
    mean,cov=common.gain_moments()
    np.testing.assert_allclose(mean,1.)
    np.testing.assert_allclose(cov,np.full((3,3),.05**2),rtol=1e-14)
    slope=CalibrationPrior(np.array([1.,2.,4.]),0.,.1)
    mean,cov=slope.gain_moments()
    assert mean[0]>1 and mean[2]>1 and mean[1]==pytest.approx(1.)
    np.testing.assert_allclose(cov[1],0.,atol=1e-18)
    assert cov[0,2]<0  # Reciprocal slope-only edge gains, not independent errors.


def test_original_targets_profiles_and_explicit_challenge_evaluation(bank):
    root=bank(challenges=3);plan={r["source_id"]:r for r in read(root/"protected/plan.json")}
    plain=NPEInputDataset(root/"public")
    assert len(plain)==4
    assert len(NPEInputDataset(root/"public",include_challenges=True))==10
    with pytest.raises(ValueError,match="explicit"):
        NPEInputDataset(root/"public",split="challenge")
    with pytest.raises(ValueError,match="explicit"):
        PairedNPEDataset(root,split="challenge")
    base=PairedNPEDataset(root)
    for inputs,labels in (base[i] for i in range(len(base))):
        original=plan[inputs["bookkeeping"]["source_id"]]["source"]
        assert labels["parameters"]=={k:original[k] for k in labels["target_order"]}
        assert labels["parameter_definitions"]["size_definition"]=="full_major_axis_projected_EM_FWHM"
    challenge=PairedNPEDataset(root,split="challenge",challenge_evaluation=True)
    assert len(challenge)==6
    for i in range(len(challenge)):
        inputs,labels=challenge[i];original=plan[inputs["bookkeeping"]["source_id"]]["source"]
        assert labels["parameters"]["major_fwhm_arcsec"]==original["major_fwhm_arcsec"]
        assert labels["base_training_eligible"] is False
        assert "profile" not in feature_keys(inputs["features"])
        stored=read(root/"protected/labels"/(inputs["bookkeeping"]["source_id"]+".json"))
        assert stored["parameter_definitions"]==labels["parameter_definitions"]
    profile_sid=next(sid for sid,item in plan.items() if item["source"]["profile"]=="ellipsoid_em")
    generic=PairedDataset(root,split="challenge")
    index=next(i for i,r in enumerate(generic.predictor.records) if r["source_id"]==profile_sid)
    with pytest.raises(ValueError,match="Boundary-diameter"):
        generic[index]
    explicit=PairedDataset(root,split="challenge",allow_profile_challenges=True)[index][1]
    assert explicit["parameter_definitions"]["size_definition"]=="full_major_axis_projected_EM_boundary_diameter"


def test_paired_targets_reject_rehashed_label_changes_against_original_plan(bank):
    root=bank();view=PairedNPEDataset(root);record=view.predictor.records[0];sid=record["source_id"]
    path=root/"protected/labels"/(sid+".json");label=read(path);label["parameters"]["te_k"]+=1;atomic_json(path,label)
    commitpath=root/"protected/commits"/(sid+".json");commit=read(commitpath)
    commit["checksums"][path.relative_to(root).as_posix()]=digest(path);atomic_json(commitpath,commit)
    with pytest.raises(ValueError,match="original generating"):
        view[0]


@pytest.mark.parametrize("defect",["extra_profile","missing_correlation","seed","grid_extra","uv_convention","operator","negative_rms"])
def test_malformed_or_hidden_metadata_is_not_silently_defaulted(bank,defect):
    root=bank();path=root/"public/instruments/base.json";meta=read(path)
    if defect=="extra_profile":meta["profile"]="ellipsoid_em"
    if defect=="missing_correlation":meta["instrument"].pop("noise_frequency_correlation")
    if defect=="seed":meta["instrument"]["sampling_seed"]=5
    if defect=="grid_extra":meta["grid"]["true_size"]=30.
    if defect=="uv_convention":meta["uv_grid_convention"]="unknown"
    if defect=="operator":meta["sampling_report"]["model"]="VLA"
    if defect=="negative_rms":meta["instrument"]["rms_jy_dirty_beam"]=-1.
    atomic_json(path,meta)
    with pytest.raises(ValueError):NPEInputDataset(root/"public",verify=False)[0]


@pytest.mark.parametrize("defect",["weight_sign","weight_symmetry","pb","psf_shape","sigma","packing","unknown_array"])
def test_bad_array_and_uncertainty_contracts_rejected(bank,defect):
    root=bank();record=read(root/"public/manifest.json")["records"][0]
    if defect in {"sigma","packing","unknown_array"}:
        path=root/"public"/record["measurement"]
        with np.load(path) as f:arrays={k:f[k].copy() for k in f.files}
        if defect=="sigma":arrays["sigma_real_jy"]*=2.
        if defect=="packing":arrays["vis_jy"]=arrays["vis_jy"][:-1]
        if defect=="unknown_array":arrays["te_k"]=np.array(8000.)
    else:
        path=root/"public/instruments/base.npz"
        with np.load(path) as f:arrays={k:f[k].copy() for k in f.files}
        if defect=="weight_sign":arrays["weights"][0,1,1]=-1.
        if defect=="weight_symmetry":arrays["weights"][0,1,1]+=1.
        if defect=="pb":arrays["primary_beam"][:]=1.
        if defect=="psf_shape":arrays["psf"]=arrays["psf"][:,:,:-1]
    write_npz(path,**arrays)
    with pytest.raises(ValueError):NPEInputDataset(root/"public",verify=False)[0]


def test_parent_grouping_and_duplicate_lineage_cannot_be_hidden_by_split_filter(bank):
    root=bank();path=root/"public/manifest.json";manifest=read(path)
    source=manifest["records"][0]["source_id"]
    variants=[r for r in manifest["records"] if r["source_id"]==source]
    assert len(variants)==2 and len({r["split"] for r in variants})==1
    variants[1]["split"]="validation" if variants[0]["split"]!="validation" else "train"
    atomic_json(path,manifest)
    with pytest.raises(ValueError,match="crosses"):
        NPEInputDataset(root/"public",verify=False)
    variants[1]["split"]=variants[0]["split"]
    manifest["records"][1]["realization_id"]=manifest["records"][0]["realization_id"]
    atomic_json(path,manifest)
    with pytest.raises(ValueError,match="Duplicate"):
        NPEInputDataset(root/"public",split="train",verify=False)
