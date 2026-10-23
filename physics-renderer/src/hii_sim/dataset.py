"""Immutable source plans, atomic per-source commits and bounded resume."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import hmac
import json
import math
import secrets
import shutil
import time
import traceback
import os
import uuid
from datetime import datetime, timezone
import numpy as np
from scipy.stats import qmc
from .schema import Source, Grid, Sky, Instrument, load_config, render_grid
from .physics import render
from .instrument import observe, operator, coverage_diagnostics
from .reconstruction import dirty_image, point_spread_function
from .io import atomic_json, canonical_hash, digest, versions, code_digest, source_manifest, write_npz, write_image
from .npe_contract import source_parameter_definitions

PUBLIC_SCHEMA = "2.0"


def safe_relative(root, relative):
    """Reject absolute paths, traversal and symlinks escaping an export root."""
    root = Path(root).resolve()
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes dataset: {relative}")
    return path


def verify_checksums(root, checksums):
    for relative, expected in checksums.items():
        path = safe_relative(root, relative)
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}; preserve the product and investigate")


def subset_membership(plan):
    """Protected prefix membership; NEVER expose its generation ordering."""
    nested = {}
    for split in ("train", "validation", "test"):
        ids = [x["source_id"] for x in sorted(plan, key=lambda x: x["source_index"])
               if x["split"] == split]
        sizes = []
        n = 16
        while n < len(ids):
            sizes.append(n)
            n *= 2
        if ids:
            sizes.append(len(ids))
        nested[split] = {str(n): ids[:n] for n in sizes}
    return nested


def opaque(seed, tag, id_salt=None):
    message=f"hii-v1:{seed}:{tag}".encode()
    if id_salt is not None:
        return hmac.new(bytes.fromhex(id_salt),message,hashlib.sha256).hexdigest()[:24]
    return hashlib.sha256(message).hexdigest()[:24]


def seed_for(seed, tag):
    return int(opaque(seed,tag)[:16],16)


def make_plan(config, n_sources, id_salt=None):
    if n_sources < 1 or n_sources > 131072:
        raise ValueError("Use a bounded count between 1 and 131072")
    seed = config["seed"]
    ranges = config["sampling"]["ranges"]
    keys = list(ranges)
    unit = qmc.Sobol(5,scramble=True,seed=seed).random_base2(math.ceil(math.log2(n_sources)))[:n_sources]
    fractions = np.cumsum(config["sampling"]["split_fractions"])
    plan=[]
    total = n_sources+config["challenge_sources"]
    challenge_unit = np.random.default_rng(seed+121).uniform(size=(config["challenge_sources"],5))
    for index in range(total):
        challenge = index >= n_sources
        v = challenge_unit[index-n_sources] if challenge else unit[index]
        pars={}
        for key,x in zip(keys,v):
            lo,hi,space = ranges[key]
            pars[key] = float(np.exp(np.log(lo)+x*np.log(hi/lo)) if space=="log" else lo+x*(hi-lo))
        group = (index-n_sources)%3 if challenge else -1
        profile = config["profile"]
        if group == 0:
            profile = "ellipsoid_em"
        if group == 1:
            pars["em0_pc_cm6"] = float(np.exp(np.log(5e5)+v[0]*np.log(20)))
        source = Source(**pars,profile=profile)
        rng = np.random.default_rng(seed_for(seed,f"sky:{index}"))
        offset=config["sampling"]["offset_limit_arcsec"]
        sky=Sky(**config["sky"], east_arcsec=float(rng.uniform(-offset,offset)),
                north_arcsec=float(rng.uniform(-offset,offset)))
        split_draw=seed_for(seed,f"split:{index}")/2**64
        split = "challenge" if challenge else ("train","validation","test")[min(2,int(np.searchsorted(fractions,split_draw)))]
        plan.append({"source_id":opaque(seed,f"source:{index}",id_salt),"source_index":index,
                     "source":asdict(source),"sky":asdict(sky),"split":split,
                     "challenge_kind":("profile","turnover","instrument")[group] if challenge else None,
                     "realizations":[{"realization_id":opaque(seed,f"realization:{index}:{r}",id_salt),
                                      "seed":seed_for(seed,f"noise:{index}:{r}")}
                                     for r in range(config["realizations_per_source"])]})
    return plan


def public_instrument(inst):
    d=asdict(inst)
    d.pop("sampling_seed")
    return d


def prepare(root, config_path, n_sources):
    root=Path(root); requested=load_config(config_path)
    spec={"requested_config":requested,"n_sources":n_sources,
          "code_sha256":code_digest(),"versions":versions()}
    provenance=root/"protected"/"provenance.json"
    if provenance.exists():
        old=json.loads(provenance.read_text())
        if old.get("schema_version") != PUBLIC_SCHEMA or old.get("resume_spec") != spec:
            raise ValueError("Resume refused: source count, config, code, or runtime changed; use a new output directory")
        if canonical_hash({k:v for k,v in old.items() if k!="spec_sha256"})!=old.get("spec_sha256"):
            raise RuntimeError("Provenance checksum mismatch")
        verify_checksums(root, old["initial_checksums"])
        return old["config"],json.loads((root/"protected"/"plan.json").read_text())
    if root.exists() and any(root.iterdir()):
        raise ValueError("Nonempty output without provenance; choose a new directory")
    root.mkdir(parents=True,exist_ok=True)
    initialization=root/".initializing"
    try:
        with initialization.open("x") as handle:
            handle.write(str(os.getpid())+"\n")
    except FileExistsError as error:
        raise RuntimeError("Another process is initializing this output directory") from error
    id_salt=secrets.token_hex(32)
    # A checked-in example seed is public knowledge. It must not define a private
    # target bank, even if HMAC IDs or unordered subsets hide its exact ordering.
    config=json.loads(json.dumps(requested))
    config["seed"]=secrets.randbits(63)
    plan=make_plan(config,n_sources,id_salt)
    atomic_json(root/"protected"/"id_key.json",{"id_salt":id_salt,
                "purpose":"opaque source/realization IDs only; not a parameter or noise seed"})
    atomic_json(root/"protected"/"plan.json",plan)
    atomic_json(root/"protected"/"subset_membership.json",subset_membership(plan))
    manifest=source_manifest()
    project_root=Path(__file__).resolve().parents[2]
    for relative in manifest:
        destination=root/"protected/source_snapshot"/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(project_root/relative,destination)
    atomic_json(root/"protected"/"source_manifest.json",manifest)
    grid=Grid(**config["grid"])
    for mode in ("base","challenge_instrument"):
        inst=Instrument(**config["instrument"])
        if mode!="base":
            inst=replace(inst,min_baseline_m=1400.,rms_jy_dirty_beam=.001,
                         name="development_uv_proxy_changed_conditions_v1")
        w,pb=operator(grid,inst)
        write_npz(root/"public"/"instruments"/f"{mode}.npz",weights=w,
                  primary_beam=pb.astype("f4"),psf=point_spread_function(w).astype("f4"))
        atomic_json(root/"public"/"instruments"/f"{mode}.json",{
            "instrument":public_instrument(inst),"grid":{"npix":grid.npix,"pixel_arcsec":grid.pixel_arcsec},
            "sampling_report":coverage_diagnostics(grid,inst),
            "metadata_status":"known synthetic observing design; noise/calibration values are assumed error models",
            "uv_grid_convention":"unshifted fftfreq cycles/radian; x coordinate west, conjugate to west; y north",
            "not_available_to_predictor":["source properties","true centroid","true gains","generating seeds","truth images"]})
    shared={p.relative_to(root/"public").as_posix():digest(p)
            for p in sorted((root/"public/instruments").glob("*"))}
    atomic_json(root/"protected"/"shared_checksums.json",shared)
    initial={p.relative_to(root).as_posix():digest(p)
             for p in sorted(root.rglob("*")) if p.is_file() and p!=initialization}
    spec={"schema_version":PUBLIC_SCHEMA,"resume_spec":spec,"config":config,
          "sampling_seed_policy":"private random seed; requested seed is a documented example only",
          "source_manifest":manifest,"initial_checksums":initial,
          "created_utc":datetime.now(timezone.utc).isoformat()}
    spec["spec_sha256"]=canonical_hash(spec)
    # Provenance is the initialization commit: a partial initialization cannot be
    # mistaken for a resumable dataset.
    atomic_json(provenance,spec)
    initialization.unlink()
    return config,plan


def valid_commit(root, item):
    root=Path(root)
    path=root/"protected"/"commits"/(item["source_id"]+".json")
    if not path.exists():
        return False
    commit=json.loads(path.read_text())
    expected_ids={r["realization_id"] for r in item["realizations"]}
    if (commit["source_id"] != item["source_id"] or
        {r["realization_id"] for r in commit["records"]} != expected_ids or
        len(commit["records"]) != len(expected_ids) or
        any(r["source_id"]!=item["source_id"] or r["split"]!=item["split"]
            for r in commit["records"])):
        raise RuntimeError(f"Invalid source commit: {item['source_id']}")
    required={f"protected/labels/{item['source_id']}.json",f"protected/truth/{item['source_id']}.fits"}
    required.update("public/"+r[key] for r in commit["records"] for key in ("measurement","image"))
    if not required <= commit["checksums"].keys():
        raise RuntimeError(f"Incomplete checksum coverage: {item['source_id']}")
    verify_checksums(root,commit["checksums"])
    return True


def generate_source(root, config, item):
    source=Source(**item["source"]); sky=Sky(**item["sky"]); grid=render_grid(config,source.profile)
    inst=Instrument(**config["instrument"])
    mode="base"
    if item["challenge_kind"]=="instrument":
        mode="challenge_instrument"
        inst=replace(inst,min_baseline_m=1400.,rms_jy_dirty_beam=.001,name="development_uv_proxy_changed_conditions_v1")
    cube=render(source,grid,sky,inst.frequencies_ghz,inst.bandwidths_mhz,inst.frequency_quadrature)
    weights,_=operator(grid,inst)
    truth=Path("protected/truth")/(item["source_id"]+".fits")
    write_image(root/truth,cube,grid,sky,inst,"Jy/sr","intrinsic_sky_contrast")
    files=[truth]
    records=[]; hidden=[]
    for realization in item["realizations"]:
        rid=realization["realization_id"]
        measurement,corruption=observe(cube,grid,inst,realization["seed"])
        vispath=Path("public/observations")/(rid+".npz")
        imagepath=Path("public/observations")/(rid+".fits")
        support=weights>0
        # Indices/weights are in shared public operator, stored once.
        write_npz(root/vispath,vis_jy=measurement["vis_jy"][support].astype("complex64"),
                  sigma_real_jy=measurement["sigma_real_jy"])
        image=dirty_image(measurement["vis_jy"],weights)
        write_image(root/imagepath,image,grid,sky,inst,"Jy/beam","measurement_dirty_image")
        files += [vispath,imagepath]
        records.append({"source_id":item["source_id"],"realization_id":rid,"split":item["split"],
                        "instrument_id":mode,"measurement":vispath.relative_to("public").as_posix(),
                        "image":imagepath.relative_to("public").as_posix()})
        hidden.append({"realization_id":rid,**corruption})
    labelpath=Path("protected/labels")/(item["source_id"]+".json")
    atomic_json(root/labelpath,{"source_id":item["source_id"],"parameters":asdict(source),
                              "parameter_definitions":source_parameter_definitions(source.profile),
                              "sky_nuisance":asdict(sky),"corruptions":hidden,"truth":truth.as_posix(),
                              "render_subpixels":grid.subpixels,
                              "render_spatial_rule":grid.spatial_rule,
                              "challenge_kind":item["challenge_kind"],"units":{
                                  "em0_pc_cm6":"pc cm-6","te_k":"K","major_fwhm_arcsec":"arcsec", "pa_deg":"deg","axis_ratio":"1"}})
    files.append(labelpath)
    checksums={p.as_posix():digest(root/p) for p in files}
    commit={"source_id":item["source_id"],"records":records,"checksums":checksums}
    atomic_json(root/"protected"/"commits"/(item["source_id"]+".json"),commit)
    return sum((root/p).stat().st_size for p in files)+(root/"protected/commits"/(item["source_id"]+".json")).stat().st_size


def publish_indexes(root, complete, membership=None):
    """Publish only committed products, with opaque-ID ordering and shared hashes."""
    root=Path(root)
    records=[]
    for item in complete:
        commit=json.loads((root/"protected/commits"/(item["source_id"]+".json")).read_text())
        for r in commit["records"]:
            r=dict(r)
            r["sha256"]={k.split("public/",1)[1]:v for k,v in commit["checksums"].items()
                         if k in {"public/"+r["measurement"],"public/"+r["image"]}}
            records.append(r)
    records.sort(key=lambda x:x["realization_id"])
    shared=json.loads((root/"protected/shared_checksums.json").read_text())
    verify_checksums(root/"public",shared)
    if membership is None:
        membership=json.loads((root/"protected/subset_membership.json").read_text())
    available={x["source_id"] for x in complete}
    # Membership is stable across resumes. Partially generated levels are omitted.
    nested={split:{n:sorted(ids) for n,ids in levels.items() if set(ids)<=available}
            for split,levels in membership.items()}
    atomic_json(root/"public/subsets.json",{
        "schema_version":PUBLIC_SCHEMA,"unit":"independent physical sources",
        "selection":"fixed nested membership; opaque ID order carries no generation rank",
        "nested_source_ids":nested})
    atomic_json(root/"public/manifest.json",{"schema_version":PUBLIC_SCHEMA,
                "shared_sha256":shared,"subsets_sha256":digest(root/"public/subsets.json"),
                "records":records})
    return records


def generate(root, config_path, n_sources, limit=None, max_seconds=1800, max_gib=20,*,
             source_headroom_seconds=0.,source_headroom_mib=0.):
    """Cooperative source-boundary budgets, including initialization time.

    Explicit one-parent/finalization reservations prevent admission when
    insufficient budget remains. This is not hard preemption of a slow source;
    overruns are reported, never called a complete budget-compliant run.
    """
    for name,value in (("max_seconds",max_seconds),("max_gib",max_gib),
                       ("source_headroom_seconds",source_headroom_seconds),("source_headroom_mib",source_headroom_mib)):
        if (isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value)
                or value<0 or (name=="max_gib" and value==0)):
            raise ValueError(f"Invalid generation budget: {name}")
    start=time.perf_counter(); cpu_start=time.process_time()
    root=Path(root)
    config,plan=prepare(root,config_path,n_sources)
    initialization_seconds=time.perf_counter()-start
    lock=root/"protected"/"generation.lock"
    try:
        handle=lock.open("x")
    except FileExistsError as e:
        raise RuntimeError("Dataset is locked; check running workers before removing a stale generation.lock") from e
    generated=0; bytes_written=0; skipped=0; complete=[]
    run_id=uuid.uuid4().hex
    run_path=root/"protected/runs"/(run_id+".json")
    stats={"run_id":run_id,"started_utc":datetime.now(timezone.utc).isoformat(),
           "status":"running","code_sha256":code_digest(),"versions":versions(),
           "dataset_provenance_sha256":digest(root/"protected/provenance.json"),
           "initialization_seconds":initialization_seconds,
           "limits":{"new_sources":limit,"wall_seconds":max_seconds,"total_gib":max_gib,
                     "source_headroom_seconds":source_headroom_seconds,"source_headroom_mib":source_headroom_mib,
                     "enforcement":"cooperative; includes prepare; reservations are caller-supplied measured bounds, not hard preemption"}}
    atomic_json(run_path,stats)
    cap_bytes=int(max_gib*1024**3)
    try:
        handle.write(json.dumps({"pid":os.getpid(),"run_id":run_id})+"\n"); handle.close()
        current_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        if current_bytes>=cap_bytes and any(not valid_commit(root,item) for item in plan):
            raise RuntimeError("Dataset already reached --max-gib; increase the cap explicitly to continue")
        if shutil.disk_usage(root).free < 16*1024**2:
            raise RuntimeError("Less than 16 MiB free; refusing to begin a generation batch")
        # File creation times and directory insertion order must not reproduce
        # Sobol rank either. ID ordering is keyed independently of the design.
        for item in sorted(plan,key=lambda item:item["source_id"]):
            if valid_commit(root,item):
                complete.append(item); skipped+=1
                continue
            if ((limit is not None and generated>=limit) or
                    time.perf_counter()-start+source_headroom_seconds>=max_seconds or
                    current_bytes+bytes_written+int(source_headroom_mib*1024**2)>=cap_bytes):
                break
            avg_source_bytes=bytes_written/max(generated,1)
            if avg_source_bytes and shutil.disk_usage(root).free < max(4*avg_source_bytes, 4*1024**2):
                break
            try:
                bytes_written+=generate_source(root,config,item)
            except Exception as e:
                atomic_json(root/"protected"/"failure.json",{"source_id":item["source_id"],"error":repr(e),"traceback":traceback.format_exc()})
                raise
            complete.append(item); generated+=1
            if generated%64==0:
                print(f"Committed {generated} new physical sources ({skipped} resumed), {time.perf_counter()-start:.1f}s",flush=True)
        records=publish_indexes(root,complete)
        elapsed=time.perf_counter()-start
        # Includes initial/shared files, commit JSON and rebuilt indices. Final
        # run-summary writes remain covered by the caller's reservation.
        actual_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        budget_exceeded=elapsed>max_seconds or actual_bytes>cap_bytes
        stats.update({"new_sources":generated,"resumed_sources":skipped,"complete_sources":len(complete),
               "planned_sources":len(plan),"realizations":len(records),"wall_seconds":elapsed,
               "cpu_seconds":time.process_time()-cpu_start,"new_bytes":bytes_written,
               "seconds_per_new_source":elapsed/generated if generated else None,
               "bytes_per_new_source":bytes_written/generated if generated else None,
               "output_bytes_before_final_history_write":actual_bytes,
               "budget_exceeded":budget_exceeded,
               "status":("budget_exceeded" if budget_exceeded else
                         "complete" if len(complete)==len(plan) else "bounded_checkpoint")})
        stats["finished_utc"]=datetime.now(timezone.utc).isoformat()
        atomic_json(run_path,stats)
        atomic_json(root/"protected"/"last_run.json",stats)
        return stats
    except BaseException as error:
        stats.update({"status":"interrupted" if isinstance(error,KeyboardInterrupt) else "failed",
                      "finished_utc":datetime.now(timezone.utc).isoformat(),
                      "error":repr(error),"new_sources":generated,"resumed_sources":skipped,
                      "wall_seconds":time.perf_counter()-start,"cpu_seconds":time.process_time()-cpu_start})
        atomic_json(run_path,stats)
        atomic_json(root/"protected/last_run.json",stats)
        raise
    finally:
        handle.close()
        lock.unlink(missing_ok=True)
