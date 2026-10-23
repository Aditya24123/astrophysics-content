"""Recheck an existing candidate and prepare batches without training/generation.

Run from the project root with PYTHONPATH=src. Results go into a NEW directory;
the dataset is read only. Prior expensive numerical/noise evidence is reused
only with its recorded dataset and unchanged core code/configuration.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from hii_sim.dataset import verify_checksums
from hii_sim.io import atomic_json, canonical_hash, digest, source_manifest, versions
from hii_sim.loaders import NPEInputDataset, PairedNPEDataset
from hii_sim.training_data import (TARGET_ORDER, fit_training_transform, input_signature,
                                   iter_batches, load_transform, save_transform)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory(root):
    prov = read(root/"protected/provenance.json")
    require(canonical_hash({k:v for k,v in prov.items() if k != "spec_sha256"}) == prov["spec_sha256"],
            "Dataset provenance self-hash mismatch")
    verify_checksums(root, prov["initial_checksums"])
    frozen = read(root/"protected/source_manifest.json")
    require(frozen == prov["source_manifest"] and canonical_hash(frozen) == prov["resume_spec"]["code_sha256"],
            "Frozen generating code manifest mismatch")
    current = source_manifest()
    critical = [p for p in frozen if p.startswith("src/hii_sim/") or p.startswith("requirements") or p == "pyproject.toml"]
    require(all(current.get(p) == frozen[p] for p in critical), "Core generator/loader/dependencies changed since the candidate")
    require(all(versions()[k] == v for k,v in prov["resume_spec"]["versions"].items() if k != "platform"),
            "Scientific runtime versions differ from the frozen run")
    manifest = read(root/"public/manifest.json")
    last = read(root/"protected/last_run.json")
    require(last["status"] == "complete", "Generation is partial")
    # The plan is checksum verified for lineage. Holdout parameter values are
    # not returned, summarized, or used for normalization/experiment choices.
    plan = read(root/"protected/plan.json")
    expected = set(prov["initial_checksums"]) | {
        "protected/provenance.json", "public/manifest.json", "public/subsets.json", "protected/last_run.json"}
    records, parent_splits = [], {}
    expected_labels = set()
    for item in plan:
        sid = item["source_id"]
        require(sid not in parent_splits, "Duplicate physical parent")
        parent_splits[sid] = item["split"]
        cp = root/"protected/commits"/(sid+".json")
        commit = read(cp)
        expected.add(cp.relative_to(root).as_posix())
        verify_checksums(root, commit["checksums"])
        expected.update(commit["checksums"])
        expected_labels.add("protected/labels/"+sid+".json")
        require(commit["source_id"] == sid, "Parent commit mismatch")
        require({r["realization_id"] for r in commit["records"]} ==
                {r["realization_id"] for r in item["realizations"]}, "Parent realization lineage mismatch")
        require(all(r["source_id"] == sid and r["split"] == item["split"] for r in commit["records"]),
                "Realization crosses parent/split")
        records.extend(commit["records"])
    del plan
    expected.update(p.relative_to(root).as_posix() for p in (root/"protected/runs").glob("*.json"))
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == expected, "Missing/unexpected dataset files: " + str(sorted(actual ^ expected)[:8]))
    require(expected_labels <= actual, "Missing original labels")
    by_rid = {r["realization_id"]: r for r in records}
    require(len(by_rid) == len(records) == len(manifest["records"]), "Duplicate/missing observations")
    for r in manifest["records"]:
        require({k:v for k,v in r.items() if k != "sha256"} == by_rid[r["realization_id"]],
                "Public manifest differs from committed products")
    groups = Counter(r["source_id"] for r in records)
    require(len(groups) == len(parent_splits), "Committed parents differ from plan")
    require(last["complete_sources"] == len(groups) == last["planned_sources"], "Completion count mismatch")
    require(last["realizations"] == len(records), "Completion observation count mismatch")
    return {"status":"PASS", "files_hashed":len(actual),
            "bytes":sum((root/p).stat().st_size for p in actual),
            "parent_counts":dict(Counter(parent_splits.values())),
            "observation_counts":dict(Counter(r["split"] for r in records)),
            "realizations_per_parent":dict(Counter(groups.values())),
            "dataset_generation_code_sha256":prov["resume_spec"]["code_sha256"],
            "core_files_identical_to_frozen_generation":critical,
            "noncore_changed_frozen_files":[p for p in frozen if current.get(p) != frozen[p]],
            "current_added_files":sorted(set(current)-set(frozen)),
            "current_code_sha256":canonical_hash(current),
            "public_manifest_sha256":digest(root/"public/manifest.json"),
            "provenance_sha256":digest(root/"protected/provenance.json"),
            "heldout_target_handling":"protected plan parsed for lineage; holdout label/truth files hashed only; holdout parameter values not summarized, selected on, or used for normalization; prior full original-plan equality audit reused"}


def scan_public(root):
    """Enforce no protected or FITS open even for challenges during this scan."""
    guard = {"active":True, "attempts":[]}
    def hook(event, args):
        if event != "open" or not guard["active"] or not isinstance(args[0], (str, bytes)):
            return
        name = str(args[0]).replace("\\", "/").lower()
        if "/protected/" in name or name.endswith(".fits"):
            guard["attempts"].append(name)
            raise PermissionError("Public feature scan forbids protected/FITS access")
    sys.addaudithook(hook)
    layouts = defaultdict(Counter)
    signatures = defaultdict(set)
    try:
        view = NPEInputDataset(root/"public", verify=True, include_challenges=True)
        for inputs in view:
            require(set(inputs) == {"features","bookkeeping"} and
                    set(inputs["features"]) == {"measurements","observing"}, "Unexpected feature fields")
            m, o = inputs["features"]["measurements"], inputs["features"]["observing"]
            require(m.unit == "Jy" and np.isfinite(m.values_jy).all(), "Invalid measurement values/units")
            require(not m.values_jy.flags.writeable, "Measurements are not immutable")
            require(o.taper_role == "imaging_weight_not_inverse_variance", "Weight semantics changed")
            split = inputs["bookkeeping"]["split"]
            layouts[split][str(m.values_jy.size)] += 1
            signatures[split].add(input_signature(inputs))
        require(not guard["attempts"], "Protected/FITS access during feature scan")
    finally:
        guard["active"] = False
    base_signatures = set.union(*(signatures[s] for s in ("train","validation","test")))
    require(len(base_signatures) == 1, "Base splits cannot use one fixed-vector input contract")
    require(len(NPEInputDataset(root/"public")) == sum(sum(layouts[s].values()) for s in ("train","validation","test")),
            "Default NPE view contains challenges or omits base records")
    return {"status":"PASS", "all_public_observations_checked":sum(sum(v.values()) for v in layouts.values()),
            "feature_length_counts_by_split":{k:dict(v) for k,v in layouts.items()},
            "operator_layout_signatures_per_split":{k:len(v) for k,v in signatures.items()},
            "base_input_signature_sha256":next(iter(base_signatures)),
            "private_or_FITS_feature_access_attempts":guard["attempts"],
            "challenge_policy":"24 separate diagnostic observations; no challenge normalization, fitting, or fixed-vector batching"}


def inherited_evidence(root, evidence):
    old_inventory = read(evidence/"candidate_inventory_01.json")
    require(old_inventory["status"] == "pass" and
            old_inventory["provenance_sha256"] == digest(root/"protected/provenance.json") and
            old_inventory["public_manifest_sha256"] == digest(root/"public/manifest.json"), "Prior audit is for a different candidate")
    require(old_inventory["all_original_targets_equal_generating_plan"] and old_inventory["all_variant_groups_preserved"],
            "Prior original-label/split audit failed")
    accuracy = read(evidence/"candidate_accuracy_01.json")
    prov = read(root/"protected/provenance.json")
    require(accuracy["all_parent_checks_pass"] and
            accuracy["source_digest_at_start"] == accuracy["source_digest_at_finish"] == prov["resume_spec"]["code_sha256"],
            "Prior all-parent accuracy audit failed or used different code")
    require(len(accuracy["cases"]) == old_inventory["physical_parents"], "Prior accuracy omitted parents")
    precision = read(evidence/"candidate_storage_precision_01.json")
    require(precision["status"] == "pass", "Prior storage precision audit failed")
    noise = read(evidence/"proxy_noise_audit_01.json")
    protocol = evidence/"proxy_noise_protocol_v1.json"
    require(noise["status"] == "pass" and all(g["pass"] for g in noise["primary_statistics"]) and
            noise["protocol_sha256"] == digest(protocol), "Prior independent proxy-noise evidence failed")
    # This experiment used compact_proxy_v2. GL4 changes spatial integration,
    # not grid geometry, the observing operator or stochastic noise/gain law.
    noise_config = read(Path(__file__).resolve().parents[1]/"configs/compact_proxy_v2.json")
    require(noise_config["instrument"] == prov["config"]["instrument"], "Noise-evidence instrument configuration differs")
    require(all(noise_config["grid"][k] == prov["config"]["grid"][k] for k in ("npix","pixel_arcsec")),
            "Noise-evidence grid geometry differs")
    names = ["candidate_inventory_01.json", "candidate_accuracy_01.json", "candidate_storage_precision_01.json",
             "proxy_noise_audit_01.json", "proxy_noise_protocol_v1.json", "audit_proxy_noise.py"]
    return {"status":"PASS", "reuse_reason":"fresh hashes show identical candidate and frozen core generator/loader; no repeated expensive forward/CASA runs",
            "evidence_sha256":{n:digest(evidence/n) for n in names},
            "all_parent_numerical_checks":len(accuracy["cases"]),
            "max_combined_numerical_and_storage_whitened_noise_norm":precision["release_accuracy_max_plus_global_roundoff_bound"],
            "predeclared_numerical_budget":precision["budget"],
            "numerical_metric_meaning":"dimensionless joint whitened forward-discretization plus storage error, NOT image error or parameter inference error",
            "independent_proxy_noise_draws":noise["draws"], "noise_statistical_gates":len(noise["primary_statistics"]),
            "noise_applicability":"same six-band proxy grid/operator/stochastic law; GL4 affects noiseless source integration only",
            "not_claimed":"CASA-to-proxy transfer, real-data validity, posterior coverage or trained-parameter accuracy"}


def smoke(root, transform, output):
    reloaded = load_transform(output, root/"public")
    require(np.array_equal(reloaded.feature_mean_jy, transform.feature_mean_jy), "Saved normalization changed")
    results = {}
    for split in ("train", "validation", "test"):
        count, batches, parents, seen = 0, 0, set(), set()
        sum_x = np.zeros(transform.metadata["feature_count"])
        sum_x2 = sum_x.copy()
        max_y_roundtrip = 0.
        for batch in iter_batches(root, reloaded, split=split, batch_size=32,
                                  shuffle=split == "train", order_seed=22022, include_targets=split != "test"):
            x = batch["features"]["values"]
            require(set(batch["features"]) == {"values"} and x.dtype == np.float32 and np.isfinite(x).all(),
                    "Invalid model features")
            count += len(x)
            batches += 1
            for book in batch["bookkeeping"]:
                require(book["split"] == split and book["realization_id"] not in seen, "Mixed split/duplicate batch row")
                parents.add(book["source_id"])
                seen.add(book["realization_id"])
            if split == "test":
                require("targets" not in batch, "Starter unsealed test targets")
            else:
                y = batch["targets"]
                require(y["original"].shape == y["standardized"].shape == (len(x),len(TARGET_ORDER)), "Invalid targets")
                require(np.isfinite(y["original"]).all() and np.isfinite(y["standardized"]).all(), "Nonfinite targets")
                err = np.max(abs(reloaded.original_targets(y["standardized"])/y["original"]-1))
                max_y_roundtrip = max(max_y_roundtrip,float(err))
                require(err < 2e-6, "Float32 target transform roundtrip exceeded tolerance")
            if split == "train":
                sum_x += x.astype(float).sum(axis=0)
                sum_x2 += (x.astype(float)**2).sum(axis=0)
        require(count > 0, "Empty split")
        result = {"observations":count,"unique_parents":len(parents),"batches":batches,
                  "feature_shape":["B",transform.metadata["feature_count"]], "dtype":"float32",
                  "target_shape":None if split == "test" else ["B",len(TARGET_ORDER)],
                  "max_target_relative_roundtrip_error":None if split == "test" else max_y_roundtrip}
        if split == "train":
            result["max_abs_standardized_feature_mean"] = float(np.max(abs(sum_x/count)))
            result["max_abs_standardized_feature_variance_minus_one"] = float(np.max(abs(sum_x2/count-(sum_x/count)**2-1)))
            require(result["max_abs_standardized_feature_mean"] < 2e-7 and
                    result["max_abs_standardized_feature_variance_minus_one"] < 2e-7, "Normalization smoke failed")
        results[split] = result
    return {"status":"PASS", "splits":results, "training_executed":False,
            "test_labels_loaded_by_starter":False, "normalization_fit_split":"train",
            "target_scaling_fit_unique_parents":transform.metadata["fit_unique_parents"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--evidence", type=Path, default=Path("output/training_readiness_20261021"))
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    require(root.is_dir(), "Dataset root does not exist")
    require(out != root and root not in out.parents and out not in root.parents,
            "Preparation output must be separate from the immutable dataset")
    require(not out.exists(), "Output already exists; choose a new version directory")
    started = time.perf_counter()
    report = {"created_utc":datetime.now(timezone.utc).isoformat(), "dataset_root":str(root),
              "intended_use":"initial four-target experiment on the declared six-band compact Fourier proxy",
              "gates":{}, "training_executed":False}
    def checkpoint(name, result):
        report["gates"][name] = result
        print(json.dumps({"gate":name,"status":result["status"]}), flush=True)
    try:
        checkpoint("immutable_inventory_lineage_runtime", inventory(root))
        checkpoint("public_features_and_source_splits", scan_public(root))
        checkpoint("existing_numerical_and_noise_evidence", inherited_evidence(root,args.evidence))
        transform = fit_training_transform(root)
        save_transform(transform,out)
        checkpoint("train_only_normalization", {"status":"PASS", **transform.metadata})
        checkpoint("end_to_end_batches_original_targets", smoke(root,transform,out))
        report["status"] = "PASS_FOR_INITIAL_DECLARED_PROXY_EXPERIMENT"
        report["remaining_scientific_boundaries"] = [
            "512 base parents support a small initial experiment; sample sufficiency/generalization unmeasured until later learning curves.",
            "48 test parents are only 48 independent skies, not 96 independent source tests; keep test targets sealed until final evaluation.",
            "Four original targets do not imply four identifiable quantities; EM/Te and calibration degeneracies remain part of the problem.",
            "Artificial Fourier lattice, simple smooth pure-H source family, Gaussian thermal law and global real gains are not a validated multi-band real telescope.",
            "Challenges are separate diagnostics; profile size semantics/layout changes prohibit silently mixing them into this fixed-vector base training.",
            "Standardization is empirical scaling, not noise whitening; thermal covariance alone omits signal-dependent shared calibration covariance.",
            "CASA image reconstruction errors and separately tested CASA photometry are not candidate label errors or NPE posterior coverage.",
            "This controlled local release has protected labels/plan; distribute only the public subtree to measurement-only evaluators."
        ]
    except Exception as exc:
        report["status"] = "FAIL"
        report["failure"] = {"type":type(exc).__name__,"message":str(exc)}
        # Store failure evidence outside the bank even if fitting did not start.
        out.mkdir(parents=True,exist_ok=True)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["wall_seconds"] = time.perf_counter()-started
        report["executed_source_sha256"] = source_manifest()
        if out.exists():
            atomic_json(out/"readiness_report.json",report)
    print(json.dumps({"status":report["status"],"report":str(out/"readiness_report.json"),
                      "wall_seconds":report["wall_seconds"]}),flush=True)


if __name__ == "__main__":
    main()
