"""Predictor never reads a protected record. Paired loading is explicit."""
from pathlib import Path
import json
import warnings
import re
from dataclasses import fields
import numpy as np
from .io import read_image, digest
from .schema import Instrument
from .dataset import PUBLIC_SCHEMA, safe_relative, verify_checksums
from .npe_contract import (canonical_fourier_data, source_parameter_definitions,
                           immutable_array, ProxyObservingMetadata, ThermalCovariance, CalibrationPrior)


class PredictorDataset:
    """Construct with ONLY the public export directory; it can be copied separately."""
    def __init__(self, public_root, split=None, verify=False, allow_legacy=False):
        self.root=Path(public_root)
        self.verify=verify
        manifest=json.loads((self.root/"manifest.json").read_text())
        self.legacy=manifest.get("schema_version") != PUBLIC_SCHEMA
        if self.legacy and not allow_legacy:
            raise ValueError("Legacy public export exposes a recoverable target bank; use a versioned v2 export or explicitly set allow_legacy=True for audit only")
        if self.legacy:
            warnings.warn("Legacy v1 export has known label leakage; do not use for evaluation",RuntimeWarning)
        self.shared=manifest.get("shared_sha256",{})
        if not self.legacy:
            required_shared={"instruments/"+r["instrument_id"]+suffix
                             for r in manifest["records"] for suffix in (".json",".npz")}
            if not required_shared <= self.shared.keys():
                raise ValueError("Manifest lacks shared instrument checksum coverage")
        if self.verify:
            verify_checksums(self.root,self.shared)
        allowed={"source_id","realization_id","split","instrument_id","measurement","image","sha256"}
        for r in manifest["records"]:
            if set(r)!=allowed:
                raise ValueError("Unexpected predictor manifest fields")
            if any(re.fullmatch(r"[0-9a-f]{24}",r[key]) is None for key in ("source_id","realization_id")):
                raise ValueError("Invalid opaque bookkeeping ID")
            if re.fullmatch(r"[a-z0-9_-]+",r["instrument_id"]) is None:
                raise ValueError("Invalid instrument ID")
            for key in ("measurement","image"):
                safe_relative(self.root,r[key])
                if r[key] not in r["sha256"]:
                    raise ValueError("Manifest lacks observation checksum coverage")
            if set(r["sha256"])!={r["measurement"],r["image"]}:
                raise ValueError("Unexpected observation checksum entries")
        if not self.legacy:
            if not manifest.get("subsets_sha256"):
                raise ValueError("Manifest lacks public subset checksum coverage")
            if self.verify and digest(self.root/"subsets.json")!=manifest["subsets_sha256"]:
                raise RuntimeError("Checksum mismatch: subsets.json")
            subsets=json.loads((self.root/"subsets.json").read_text())
            if subsets.get("schema_version")!=PUBLIC_SCHEMA:
                raise ValueError("Subset schema mismatch")
            self.subsets=subsets["nested_source_ids"]
            if set(self.subsets)!={"train","validation","test"}:
                raise ValueError("Invalid subset split keys")
            source_splits={}
            for r in manifest["records"]:
                if r["source_id"] in source_splits and source_splits[r["source_id"]]!=r["split"]:
                    raise ValueError("Physical source crosses dataset splits")
                source_splits[r["source_id"]]=r["split"]
            for subset_split,levels in self.subsets.items():
                previous=set()
                for count,ids in sorted(levels.items(),key=lambda p:int(p[0])):
                    current=set(ids)
                    if (len(ids)!=int(count) or len(current)!=len(ids) or ids!=sorted(ids) or
                        not previous<=current or any(source_splits.get(sid)!=subset_split for sid in ids)):
                        raise ValueError("Invalid public subset membership, ordering or nesting")
                    previous=current
        self._all_records=manifest["records"]
        self.records=[r for r in self._all_records if split is None or r["split"]==split]
        self._instruments={}

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        r=self.records[index]
        if self.verify:
            verify_checksums(self.root,r["sha256"])
        mode=r["instrument_id"]
        if mode not in self._instruments:
            meta=json.loads((self.root/"instruments"/(mode+".json")).read_text())
            allowed_meta={"instrument","grid","metadata_status","uv_grid_convention","not_available_to_predictor","sampling_report"}
            allowed_instrument={f.name for f in fields(Instrument)}-{"sampling_seed"}
            if set(meta)-allowed_meta or set(meta["instrument"])-allowed_instrument:
                raise ValueError("Unexpected instrument metadata; potential protected data exposure")
            if set(meta["grid"])-{"npix","pixel_arcsec"}:
                raise ValueError("Unexpected grid metadata")
            with np.load(self.root/"instruments"/(mode+".npz")) as f:
                if set(f.files)!={"weights","primary_beam","psf"}:
                    raise ValueError("Unexpected shared instrument arrays")
                arrays={k:f[k].copy() for k in f.files}
            self._instruments[mode]=(meta,arrays)
        meta,arrays=self._instruments[mode]
        image,image_meta=read_image(self.root/r["image"])
        with np.load(self.root/r["measurement"]) as f:
            if set(f.files)!={"vis_jy","sigma_real_jy"}:
                raise ValueError("Unexpected measurement arrays")
            measurements={k:f[k].copy() for k in f.files}
        # IDs/splits are returned as bookkeeping, NOT model feature tensors.
        return {"observations":{"dirty_image":image,**measurements},
                "metadata":{"image":image_meta,**meta,**{k:v.copy() for k,v in arrays.items()}},
                "bookkeeping":{"source_id":r["source_id"],"realization_id":r["realization_id"],"split":r["split"]}}


def _paired_plan(root, provenance, verify):
    path=root/"protected/plan.json"
    if verify:
        expected=provenance.get("initial_checksums",{}).get("protected/plan.json")
        if not expected or digest(path)!=expected:
            raise RuntimeError("Original protected plan is not checksum-anchored")
    return {item["source_id"]:item for item in json.loads(path.read_text())}


def _original_label(root, record, targets, plan, verify, allow_profile_challenges):
    sid=record["source_id"]
    if verify:
        commit=json.loads((root/"protected/commits"/(sid+".json")).read_text())
        verify_checksums(root,commit["checksums"])
    label=json.loads((root/"protected/labels"/(sid+".json")).read_text())
    if (label.get("source_id")!=sid or sid not in plan or
            label.get("parameters")!=plan[sid]["source"] or record["split"]!=plan[sid]["split"]
            or label.get("challenge_kind")!=plan[sid].get("challenge_kind")):
        raise ValueError("Paired labels differ from the original generating parent/parameters")
    definitions=source_parameter_definitions(label["parameters"]["profile"])
    if label.get("parameter_definitions",definitions)!=definitions:
        raise ValueError("Original profile/size definition mismatch")
    if label.get("units")!=definitions["units"]:
        raise ValueError("Original target units mismatch")
    if ("major_fwhm_arcsec" in targets and definitions["profile"]!="gaussian_em"
            and not allow_profile_challenges):
        raise ValueError("Boundary-diameter profile requires explicit allow_profile_challenges evaluation; cannot mix with Gaussian FWHM")
    return label,{"parameters":{k:label["parameters"][k] for k in targets},
                  "target_order":list(targets), "parameter_definitions":definitions,
                  "definition_provenance":("stored_protected_record" if "parameter_definitions" in label
                                           else "original_protected_profile_legacy_view_only"),
                  "challenge_kind":label.get("challenge_kind"),
                  "base_training_eligible":record["split"]!="challenge" and definitions["profile"]=="gaussian_em"}


class PairedDataset:
    def __init__(self,dataset_root,split=None,targets=None,verify=False,allow_legacy=False,*,allow_profile_challenges=False):
        self.root=Path(dataset_root)
        self.predictor=PredictorDataset(self.root/"public",split,verify,allow_legacy)
        provenance=json.loads((self.root/"protected/provenance.json").read_text())
        config=provenance["config"]
        self.targets=config["reported_targets"] if targets is None else list(targets)
        allowed=set(config["sampling"]["ranges"])
        if not set(self.targets)<=allowed or len(set(self.targets))!=len(self.targets):
            raise ValueError("Unknown target")
        self.plan=_paired_plan(self.root,provenance,verify)
        self.allow_profile_challenges=allow_profile_challenges

    def __len__(self):
        return len(self.predictor)

    def __getitem__(self,index):
        inputs=self.predictor[index]
        label,targets=_original_label(self.root,self.predictor.records[index],self.targets,self.plan,
                                      self.predictor.verify,self.allow_profile_challenges)
        truth,meta=read_image(self.root/label["truth"])
        return inputs,{**targets,"truth_sky_jy_sr":truth,"truth_metadata":meta}


class NPEInputDataset(PredictorDataset):
    """Strict public-only features for the declared proxy, not a fitted model.

    Dirty/CLEAN FITS are never opened. Challenges require explicit evaluation
    opt-in; IDs/splits remain outside the returned features object.
    """
    def __init__(self,public_root,split=None,verify=True,*,include_challenges=False):
        super().__init__(public_root,split,verify,allow_legacy=False)
        if split=="challenge" and not include_challenges:
            raise ValueError("Challenge inputs require explicit include_challenges evaluation")
        if any(r["split"] not in {"train","validation","test","challenge"} for r in self._all_records):
            raise ValueError("Invalid source split")
        if len({r["realization_id"] for r in self._all_records})!=len(self._all_records):
            raise ValueError("Duplicate realization ID")
        if not include_challenges:
            self.records=[r for r in self.records if r["split"]!="challenge"]
        self._operators={}

    def _observing(self,mode,sigma):
        if mode not in self._operators:
            path=self.root/"instruments"/(mode+".json")
            meta=json.loads(path.read_text())
            allowed={"instrument","grid","metadata_status","uv_grid_convention","not_available_to_predictor","sampling_report"}
            instrument_fields={f.name for f in fields(Instrument)}-{"sampling_seed"}
            if (set(meta)!=allowed or set(meta["instrument"])!=instrument_fields or
                    set(meta["grid"])!={"npix","pixel_arcsec"}):
                raise ValueError("Unexpected or incomplete observing metadata")
            if meta["sampling_report"].get("model")!="artificial_Hermitian_Fourier_lattice_not_antenna_tracks":
                raise ValueError("NPE input view supports only the declared Fourier proxy")
            if meta["uv_grid_convention"]!="unshifted fftfreq cycles/radian; x coordinate west, conjugate to west; y north":
                raise ValueError("Unsupported Fourier coordinate convention")
            inst=Instrument(**meta["instrument"],sampling_seed=0)
            n,cell=meta["grid"]["npix"],meta["grid"]["pixel_arcsec"]
            if (isinstance(n,bool) or not isinstance(n,int) or n<16 or n%2 or
                    not np.isfinite(cell) or cell<=0):
                raise ValueError("Invalid observed grid")
            with np.load(self.root/"instruments"/(mode+".npz"),allow_pickle=False) as archive:
                if set(archive.files)!={"weights","primary_beam","psf"}:
                    raise ValueError("Unexpected shared observing arrays")
                arrays={k:archive[k].copy() for k in archive.files}
            shape=(len(inst.frequencies_ghz),n,n)
            if any(a.shape!=shape or not np.issubdtype(a.dtype,np.floating) or
                   not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Invalid observing array dimensions/dtype/values")
            w,pb,psf=(arrays[k] for k in ("weights","primary_beam","psf"))
            ids=(-np.arange(n))%n
            if (np.any(w<0) or not np.all(w.sum(axis=(1,2))>0) or
                    not np.array_equal(w,w[:,ids[:,None],ids[None,:]]) or
                    np.any(pb<0) or np.any(pb>1)):
                raise ValueError("Invalid Hermitian weights or primary-beam response")
            from .reconstruction import point_spread_function
            eps=np.finfo(psf.dtype).eps
            if not np.allclose(psf,point_spread_function(w),rtol=8*eps,atol=8*eps):
                raise ValueError("PSF disagrees with observing weights")
            from .instrument import primary_beam
            from .schema import Grid
            expected_pb=np.asarray([primary_beam(Grid(n,cell),f,inst.pb_fwhm_arcmin_ghz)
                                    for f in inst.frequencies_ghz])
            eps_pb=np.finfo(pb.dtype).eps
            if not np.allclose(pb,expected_pb,rtol=8*eps_pb,atol=8*eps_pb):
                raise ValueError("Primary beam disagrees with declared proxy geometry")
            expected_sigma=(inst.rms_jy_dirty_beam*w.mean(axis=(1,2))/
                            np.sqrt((w*w).mean(axis=(1,2)))*n/np.sqrt(2))
            self._operators[mode]=(inst,n,float(cell),w,pb,expected_sigma)
        inst,n,cell,w,pb,expected_sigma=self._operators[mode]
        if (sigma.shape!=expected_sigma.shape or not np.issubdtype(sigma.dtype,np.floating)
                or not np.isfinite(sigma).all() or np.any(sigma<=0)
                or not np.allclose(sigma,expected_sigma,rtol=8*np.finfo(sigma.dtype).eps,atol=0)):
            raise ValueError("Thermal sigma does not match the declared proxy uncertainty law")
        frequencies=immutable_array(inst.frequencies_ghz,dtype=float)
        return ProxyObservingMetadata(n,cell,frequencies,immutable_array(inst.bandwidths_mhz,dtype=float),
            immutable_array(w>0),immutable_array(w),immutable_array(pb),
            ThermalCovariance(immutable_array(sigma),float(inst.noise_frequency_correlation)),
            CalibrationPrior(frequencies,float(inst.calibration_fraction),float(inst.calibration_slope_std)))

    def __getitem__(self,index):
        r=self.records[index]
        if self.verify:
            verify_checksums(self.root,{r["measurement"]:r["sha256"][r["measurement"]]})
        with np.load(self.root/r["measurement"],allow_pickle=False) as archive:
            if set(archive.files)!={"vis_jy","sigma_real_jy"}:
                raise ValueError("Unexpected measurement arrays")
            vis,sigma=archive["vis_jy"].copy(),archive["sigma_real_jy"].copy()
        observing=self._observing(r["instrument_id"],sigma)
        return {"features":{"measurements":canonical_fourier_data(vis,observing.imaging_taper),
                            "observing":observing},
                "bookkeeping":{k:r[k] for k in ("source_id","realization_id","split")}}


class PairedNPEDataset:
    """Original targets paired explicitly; protected metadata stays in labels."""
    def __init__(self,dataset_root,split=None,targets=None,verify=True,*,challenge_evaluation=False):
        self.root=Path(dataset_root)
        self.predictor=NPEInputDataset(self.root/"public",split,verify,include_challenges=challenge_evaluation)
        provenance=json.loads((self.root/"protected/provenance.json").read_text())
        config=provenance["config"]
        self.targets=list(config["reported_targets"] if targets is None else targets)
        if (not set(self.targets)<=set(config["sampling"]["ranges"]) or
                len(set(self.targets))!=len(self.targets)):
            raise ValueError("Unknown or duplicate original target")
        self.plan=_paired_plan(self.root,provenance,verify)
        self.challenge_evaluation=challenge_evaluation

    def __len__(self):
        return len(self.predictor)

    def __getitem__(self,index):
        inputs=self.predictor[index]
        _,labels=_original_label(self.root,self.predictor.records[index],self.targets,self.plan,
                                 self.predictor.verify,self.challenge_evaluation)
        return inputs,labels
