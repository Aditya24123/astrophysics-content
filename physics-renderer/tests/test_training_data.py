"""Leakage, artifact binding and batching tests; no model training."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hii_sim.dataset import generate
from hii_sim.io import atomic_json
from hii_sim.loaders import NPEInputDataset, PairedNPEDataset
import hii_sim.training_data as td


@pytest.fixture(scope="module")
def small_bank(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("training_contract")
    config = json.loads(Path("configs/base.json").read_text())
    config.update(challenge_sources=0,reported_targets=list(td.TARGET_ORDER))
    config["sampling"]["split_fractions"] = [.5,.2,.3]
    config["grid"] = {"npix":16,"pixel_arcsec":10.,"subpixels":2}
    config["instrument"].update(frequencies_ghz=[1.31,1.82],bandwidths_mhz=[8.,8.],
                                 min_baseline_m=300.,sampling_fraction=.5)
    path = workspace/"config.json"
    atomic_json(path,config)
    root = workspace/"bank"
    generate(root,path,16,max_seconds=120,max_gib=1)
    return root


@pytest.fixture(scope="module")
def transform(small_bank):
    return td.fit_training_transform(small_bank)


def test_fit_forces_train_and_never_decodes_holdout_labels_or_images(small_bank,monkeypatch):
    manifest = json.loads((small_bank/"public/manifest.json").read_text())
    forbidden = {small_bank/"protected/labels"/(r["source_id"]+".json")
                 for r in manifest["records"] if r["split"] != "train"}
    opened, called = set(), []
    original_open, original_dataset = Path.open, td.PairedNPEDataset
    def guarded_open(path,*args,**kwargs):
        assert path not in forbidden
        opened.add(path)
        return original_open(path,*args,**kwargs)
    def guarded_dataset(root,*,split,verify):
        called.append(split)
        assert split == "train" and verify is True
        return original_dataset(root,split=split,verify=verify)
    def no_image(*args,**kwargs):
        raise AssertionError("Training preparation cannot decode FITS")
    monkeypatch.setattr(Path,"open",guarded_open)
    monkeypatch.setattr(td,"PairedNPEDataset",guarded_dataset)
    monkeypatch.setattr("hii_sim.loaders.read_image",no_image)
    result = td.fit_training_transform(small_bank)
    assert called == ["train"] and result.metadata["fit_split"] == "train"
    assert not (opened & forbidden)


def test_target_scaling_gives_one_vote_per_parent_with_unequal_variants(small_bank,monkeypatch):
    original = PairedNPEDataset(small_bank,split="train")[0]
    inputs, labels = original
    vectors = [[1e4,5000,12,.5],[5e5,12000,60,1.]]
    rows, records = [], []
    for i,parent in enumerate((0,0,0,1)):
        book = {"source_id":str(parent),"realization_id":str(i),"split":"train"}
        m = inputs["features"]["measurements"]
        changed = replace(m,values_jy=m.values_jy+i*.001)
        x = {"features":{**inputs["features"],"measurements":changed},"bookkeeping":book}
        y = {**labels,"parameters":dict(zip(td.TARGET_ORDER,vectors[parent]))}
        rows.append((x,y)); records.append(book)
    class Fake:
        predictor = SimpleNamespace(records=records)
        def __len__(self): return len(rows)
        def __getitem__(self,i): return rows[i]
    monkeypatch.setattr(td,"PairedNPEDataset",lambda *args,**kwargs:Fake())
    result = td.fit_training_transform(small_bank)
    expected = np.asarray(vectors,dtype=float)
    expected[:,[0,2]] = np.log10(expected[:,[0,2]])
    np.testing.assert_allclose(result.target_mean,expected.mean(axis=0))
    assert result.metadata["fit_unique_parents"] == 2
    assert result.metadata["fit_observations"] == 4


def test_roundtrip_original_units_float32_and_immutable_stats(small_bank,transform):
    batch = next(td.iter_batches(small_bank,transform,split="train",batch_size=3,include_targets=True))
    assert set(batch["features"]) == {"values"}
    assert batch["features"]["values"].dtype == np.float32
    assert batch["targets"]["original"].dtype == np.float64
    assert batch["targets"]["order"] == td.TARGET_ORDER
    np.testing.assert_allclose(transform.original_targets(batch["targets"]["standardized"]),
                               batch["targets"]["original"],rtol=2e-6)
    public = NPEInputDataset(small_bank/"public",split="train")[0]
    raw = public["features"]["measurements"].values_jy
    np.testing.assert_allclose(transform.original_features(transform.features(raw)),raw,rtol=0,atol=2e-7)
    with pytest.raises(ValueError): transform.feature_scale_jy[0] = 0
    assert batch["observing"].thermal.covariance_unit == "Jy^2"
    assert not transform.metadata["is_noise_whitening"]


def test_public_only_test_batches_never_open_protected_files(small_bank,transform,monkeypatch):
    original = Path.open
    def public_only(path,*args,**kwargs):
        assert "protected" not in path.parts and path.suffix != ".fits"
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",public_only)
    batches = list(td.iter_batches(small_bank,transform,split="test",batch_size=2))
    assert batches and all("targets" not in b for b in batches)
    assert all(r["split"] == "test" for b in batches for r in b["bookkeeping"])


def test_shuffle_is_reproducible_complete_and_keeps_last_partial_batch(small_bank,transform):
    def order(seed):
        return [r["realization_id"] for b in td.iter_batches(small_bank,transform,split="train",
                batch_size=3,shuffle=True,order_seed=seed) for r in b["bookkeeping"]]
    first = order(123)
    assert first == order(123)
    assert first != order(456)
    assert len(set(first)) == len(first) == transform.metadata["fit_observations"]


@pytest.mark.parametrize("split",[None,"","all","challenge","Train"])
def test_implicit_or_mixed_split_requests_fail(small_bank,transform,split):
    with pytest.raises(ValueError,match="explicit"):
        next(td.iter_batches(small_bank,transform,split=split))


def test_test_target_access_requires_later_explicit_evaluation(small_bank,transform):
    with pytest.raises(ValueError,match="sealed"):
        next(td.iter_batches(small_bank,transform,split="test",include_targets=True))


def test_metadata_change_not_measurement_change_controls_layout_signature(small_bank,transform,monkeypatch):
    inputs = NPEInputDataset(small_bank/"public",split="train")[0]
    f = inputs["features"]
    altered_measurements = replace(f["measurements"],values_jy=f["measurements"].values_jy*1.1)
    assert td.input_signature({**inputs,"features":{**f,"measurements":altered_measurements}}) == td.input_signature(inputs)
    altered_instrument = replace(f["observing"],pixel_arcsec=f["observing"].pixel_arcsec*2)
    bad = {**inputs,"features":{**f,"observing":altered_instrument}}
    assert td.input_signature(bad) != td.input_signature(inputs)
    monkeypatch.setattr(td,"NPEInputDataset",lambda *a,**k:[bad])
    with pytest.raises(ValueError,match="Incompatible"):
        next(td.iter_batches(small_bank,transform,split="train"))


def test_saved_artifact_binding_no_overwrite_and_corruption_detection(small_bank,transform,tmp_path):
    output = tmp_path/"prepared"
    td.save_transform(transform,output)
    loaded = td.load_transform(output,small_bank/"public")
    np.testing.assert_array_equal(loaded.feature_mean_jy,transform.feature_mean_jy)
    with pytest.raises(FileExistsError): td.save_transform(transform,output)
    with (output/"normalization.npz").open("ab") as stream: stream.write(b"tamper")
    with pytest.raises(ValueError,match="checksum"):
        td.load_transform(output,small_bank/"public")


def test_different_manifest_rejects_normalization(small_bank,transform,tmp_path):
    output = tmp_path/"prepared"
    td.save_transform(transform,output)
    public = tmp_path/"public";public.mkdir()
    (public/"manifest.json").write_text("{}")
    with pytest.raises(ValueError,match="different dataset"):
        td.load_transform(output,public)


@pytest.mark.parametrize("bad",[np.nan,np.inf])
def test_nonfinite_input_and_targets_rejected(transform,bad):
    x = transform.feature_mean_jy.copy();x[0] = bad
    with pytest.raises(ValueError): transform.features(x)
    with pytest.raises(ValueError): transform.targets([1e4,bad,20,.8])
