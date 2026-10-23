"""NumPy batching for a fixed Fourier-proxy experiment; no model or training.

Labels remain original simulator parameters. Empirical standardization is not
noise whitening. Test inputs are available without opening their label files.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .io import atomic_json, canonical_hash, digest, write_npz
from .loaders import NPEInputDataset, PairedNPEDataset
from .npe_contract import immutable_array


TARGET_ORDER = ("em0_pc_cm6", "te_k", "major_fwhm_arcsec", "axis_ratio")
TARGET_TRANSFORMS = ("log10", "identity", "log10", "identity")
LAYOUT_KEYS = ("frequency_index", "y_index", "x_index", "component", "self_conjugate")


def _fingerprint(value):
    """Include array shape/dtype/bytes, never measurement values or labels."""
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": value.dtype.str,
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    if isinstance(value, dict):
        return {k: _fingerprint(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_fingerprint(v) for v in value]
    return value


def input_signature(inputs):
    f = inputs["features"]
    m = f["measurements"]
    return canonical_hash({"unit": m.unit,
                           "layout": {k: _fingerprint(getattr(m, k)) for k in LAYOUT_KEYS},
                           "observing": _fingerprint(asdict(f["observing"]))})


def _membership(records):
    return canonical_hash(sorted((r["source_id"], r["realization_id"], r["split"]) for r in records))


def _target_values(labels):
    if (tuple(labels["target_order"]) != TARGET_ORDER or
            not labels["base_training_eligible"] or labels["challenge_kind"] is not None or
            labels["parameter_definitions"]["size_definition"] != "full_major_axis_projected_EM_FWHM"):
        raise ValueError("Starter targets require the original base Gaussian parameter contract")
    v = np.asarray([labels["parameters"][k] for k in TARGET_ORDER], dtype=np.float64)
    if not np.isfinite(v).all():
        raise ValueError("Nonfinite original target")
    return v


def _target_forward(values):
    v = np.array(values, dtype=np.float64, copy=True)
    if v.ndim < 1 or v.shape[-1] != len(TARGET_ORDER) or not np.isfinite(v).all():
        raise ValueError("Invalid target shape or values")
    for j, transform in enumerate(TARGET_TRANSFORMS):
        if transform == "log10":
            if np.any(v[..., j] <= 0):
                raise ValueError("Log targets must be positive")
            v[..., j] = np.log10(v[..., j])
    return v


@dataclass(frozen=True)
class TrainingTransform:
    metadata: dict
    feature_mean_jy: np.ndarray
    feature_scale_jy: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray

    def __post_init__(self):
        for name in ("feature_mean_jy", "feature_scale_jy", "target_mean", "target_scale"):
            a = np.asarray(getattr(self, name), dtype=np.float64)
            expected = self.metadata["feature_count"] if name.startswith("feature_") else len(TARGET_ORDER)
            if a.shape != (expected,) or not np.isfinite(a).all():
                raise ValueError("Invalid normalization array: " + name)
            if "scale" in name and np.any(a <= 0):
                raise ValueError("Normalization scales must be positive")
            object.__setattr__(self, name, immutable_array(a))
        if (self.metadata["fit_split"] != "train" or
                tuple(self.metadata["target_order"]) != TARGET_ORDER or
                tuple(self.metadata["target_transforms"]) != TARGET_TRANSFORMS):
            raise ValueError("Unsupported normalization contract")

    def features(self, values_jy):
        v = np.asarray(values_jy)
        if v.shape[-1:] != self.feature_mean_jy.shape or not np.isfinite(v).all():
            raise ValueError("Invalid feature shape or values")
        result = np.asarray((v-self.feature_mean_jy)/self.feature_scale_jy, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite normalized features")
        return result

    def original_features(self, values):
        return np.asarray(values, dtype=np.float64)*self.feature_scale_jy+self.feature_mean_jy

    def targets(self, original):
        return np.asarray((_target_forward(original)-self.target_mean)/self.target_scale, dtype=np.float32)

    def original_targets(self, values):
        v = np.asarray(values, dtype=np.float64)*self.target_scale+self.target_mean
        if v.shape[-1:] != self.target_mean.shape or not np.isfinite(v).all():
            raise ValueError("Invalid normalized target values")
        for j, transform in enumerate(TARGET_TRANSFORMS):
            if transform == "log10":
                v[..., j] = np.power(10., v[..., j])
        return v


def fit_training_transform(dataset_root):
    """Fit from train observations and unique train parents only, by construction."""
    root = Path(dataset_root)
    dataset = PairedNPEDataset(root, split="train", verify=True)
    if len(dataset) < 2:
        raise ValueError("At least two training observations are needed")
    count = 0
    mean = m2 = signature = units = None
    parents = {}
    for inputs, labels in dataset:
        if inputs["bookkeeping"]["split"] != "train":
            raise ValueError("Normalization may only fit the training split")
        this_signature = input_signature(inputs)
        if signature is None:
            signature = this_signature
            shape = inputs["features"]["measurements"].values_jy.shape
            mean, m2 = np.zeros(shape), np.zeros(shape)
            units = labels["parameter_definitions"]["units"]
        if this_signature != signature:
            raise ValueError("Incompatible fixed-vector observing/layout contract")
        x = np.asarray(inputs["features"]["measurements"].values_jy, dtype=np.float64)
        count += 1
        delta = x-mean
        mean += delta/count
        m2 += delta*(x-mean)
        y = _target_values(labels)
        source_id = inputs["bookkeeping"]["source_id"]
        if source_id in parents and not np.array_equal(y, parents[source_id]):
            raise ValueError("Targets disagree across source realizations")
        parents[source_id] = y
    scale = np.sqrt(m2/count)
    transformed = _target_forward(np.stack(list(parents.values())))
    target_scale = transformed.std(axis=0)
    if np.any(scale <= 0) or np.any(target_scale <= 0):
        raise ValueError("Constant feature or target requires an explicit new policy")
    metadata = {"schema": "fixed_proxy_training_transform_v1", "fit_split": "train",
                "fit_observations": count, "fit_unique_parents": len(parents),
                "feature_count": len(mean), "input_signature_sha256": signature,
                "train_membership_sha256": _membership(dataset.predictor.records),
                "public_manifest_sha256": digest(root/"public/manifest.json"),
                "provenance_sha256": digest(root/"protected/provenance.json"),
                "target_order": list(TARGET_ORDER), "target_transforms": list(TARGET_TRANSFORMS),
                "target_units": {k: units[k] for k in TARGET_ORDER},
                "feature_input_unit": "Jy", "feature_output_unit": "dimensionless",
                "feature_fit_weighting": "one vote per training observation",
                "target_fit_weighting": "one vote per unique training parent",
                "variance_ddof": 0, "is_noise_whitening": False}
    return TrainingTransform(metadata, mean, scale, transformed.mean(axis=0), target_scale)


def save_transform(transform, output_dir):
    """Create a new versioned artifact; never replace an existing result."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    arrays = {k: getattr(transform, k) for k in
              ("feature_mean_jy", "feature_scale_jy", "target_mean", "target_scale")}
    write_npz(out/"normalization.npz", **arrays)
    atomic_json(out/"normalization.json", {**transform.metadata,
                "arrays_sha256": digest(out/"normalization.npz")})


def load_transform(output_dir, public_root):
    out, public = Path(output_dir), Path(public_root)
    metadata = json.loads((out/"normalization.json").read_text(encoding="utf-8"))
    if (metadata.get("schema") != "fixed_proxy_training_transform_v1" or
            digest(out/"normalization.npz") != metadata.get("arrays_sha256")):
        raise ValueError("Normalization artifact checksum/schema mismatch")
    if digest(public/"manifest.json") != metadata["public_manifest_sha256"]:
        raise ValueError("Normalization belongs to a different dataset manifest")
    view = NPEInputDataset(public, split="train", verify=True)
    if (_membership(view.records) != metadata["train_membership_sha256"] or
            len(view) != metadata["fit_observations"] or
            len({r["source_id"] for r in view.records}) != metadata["fit_unique_parents"]):
        raise ValueError("Normalization training membership mismatch")
    with np.load(out/"normalization.npz", allow_pickle=False) as archive:
        expected = {"feature_mean_jy", "feature_scale_jy", "target_mean", "target_scale"}
        if set(archive.files) != expected:
            raise ValueError("Unexpected normalization arrays")
        arrays = {k: archive[k].copy() for k in expected}
    return TrainingTransform(metadata, **arrays)


def iter_batches(dataset_root, transform, *, split, batch_size=32,
                 shuffle=False, order_seed=0, include_targets=False):
    """Yield original and normalized targets separately from float32 features.

    split is mandatory. Test is public-only; open its protected targets later
    through the explicit evaluation API after freezing the experiment.
    Observing metadata and its thermal covariance retain Jy/Jy^2 units. In
    normalized coordinates Cov(z_i,z_j)=Cov(v_i,v_j)/(scale_i*scale_j).
    Calibration prior moments still require signal-dependent covariance.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("Starter supports explicit train, validation or test splits only")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if include_targets and split == "test":
        raise ValueError("Starter keeps test targets sealed; use the explicit final evaluation API")
    root = Path(dataset_root)
    if digest(root/"public/manifest.json") != transform.metadata["public_manifest_sha256"]:
        raise ValueError("Normalization belongs to a different dataset manifest")
    if include_targets:
        if digest(root/"protected/provenance.json") != transform.metadata["provenance_sha256"]:
            raise ValueError("Target provenance differs from normalization provenance")
        dataset = PairedNPEDataset(root, split=split, verify=True)
    else:
        dataset = NPEInputDataset(root/"public", split=split, verify=True)
    if not len(dataset):
        raise ValueError("Requested split is empty")
    order = np.arange(len(dataset))
    if shuffle:
        np.random.default_rng(order_seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        rows, targets, books = [], [], []
        observing = None
        for index in order[start:start+batch_size]:
            if include_targets:
                inputs, labels = dataset[int(index)]
                targets.append(_target_values(labels))
            else:
                inputs = dataset[int(index)]
            if input_signature(inputs) != transform.metadata["input_signature_sha256"]:
                raise ValueError("Incompatible fixed-vector observing/layout contract")
            rows.append(inputs["features"]["measurements"].values_jy)
            books.append(inputs["bookkeeping"])
            observing = inputs["features"]["observing"]
        batch = {"features": {"values": transform.features(np.stack(rows))},
                 "observing": observing, "feature_scale_jy": transform.feature_scale_jy,
                 "bookkeeping": tuple(books)}
        if include_targets:
            original = np.stack(targets)
            batch["targets"] = {"original": original, "standardized": transform.targets(original),
                                "order": TARGET_ORDER, "units": transform.metadata["target_units"]}
        yield batch
