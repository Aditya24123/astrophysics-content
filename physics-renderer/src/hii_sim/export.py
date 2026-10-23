"""Read-only source migration; exporting cannot undo a prior label disclosure."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import shutil

from .dataset import (PUBLIC_SCHEMA, opaque, publish_indexes, safe_relative,
                      subset_membership, valid_commit, verify_checksums)
from .io import atomic_json, canonical_hash, code_digest, digest, source_manifest, versions


def _copy_verified(source,target,expected):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    if digest(target)!=expected:
        raise RuntimeError(f"Copy integrity mismatch: {source}")


def _validated_membership(membership,plan,legacy=False):
    if set(membership)!={"train","validation","test"}:
        raise ValueError("Invalid subset split keys")
    by_id={x["source_id"]:x for x in plan}
    for split,levels in membership.items():
        previous=set()
        ordered=[x["source_id"] for x in sorted(plan,key=lambda x:x["source_index"])
                 if x["split"]==split]
        for size,ids in sorted(levels.items(),key=lambda pair:int(pair[0])):
            current=set(ids)
            if (int(size)!=len(ids) or len(ids)!=len(current) or not previous<=current or
                any(sid not in by_id or by_id[sid]["split"]!=split for sid in ids)):
                raise ValueError("Invalid subset membership, group or nesting")
            if legacy and current!=set(ordered[:int(size)]):
                raise ValueError("Legacy subset does not match the original generation-prefix definition")
            previous=current
    return membership


def export_v2(source_root, destination):
    """Copy verified products to a NEW root; rotate IDs, preserve subset membership.

    All committed products are verified. Shared v1 products lack an original
    hash: hashes are established now and that limit is recorded. No source file,
    lock, provenance or history is modified. No hardlinks are used.
    """
    source=Path(source_root).resolve(); target=Path(destination).resolve()
    if target==source or target.is_relative_to(source):
        raise ValueError("Export destination must be outside the immutable source dataset")
    if target.exists() and any(target.iterdir()):
        raise ValueError("Export destination must be new or empty")
    if (source/"protected/generation.lock").exists():
        raise RuntimeError("Source has a writer lock; export only after the writer is stopped and its stale lock reviewed")
    provenance_path=source/"protected/provenance.json"
    old_provenance=json.loads(provenance_path.read_text())
    if canonical_hash({k:v for k,v in old_provenance.items() if k!="spec_sha256"})!=old_provenance.get("spec_sha256"):
        raise RuntimeError("Source provenance checksum mismatch")
    if old_provenance.get("schema_version")==PUBLIC_SCHEMA:
        if not old_provenance.get("initial_checksums"):
            raise RuntimeError("Source v2 provenance lacks initial checksum coverage")
        verify_checksums(source,old_provenance["initial_checksums"])
    source_hash=digest(provenance_path)
    plan=json.loads((source/"protected/plan.json").read_text())
    protected_subset=source/"protected/subset_membership.json"
    if old_provenance.get("schema_version")==PUBLIC_SCHEMA:
        if "protected/subset_membership.json" not in old_provenance["initial_checksums"]:
            raise RuntimeError("Source v2 provenance does not anchor subset membership")
        membership=_validated_membership(json.loads(protected_subset.read_text()),plan)
    else:
        old_subset=source/"public/subsets.json"
        membership=(json.loads(old_subset.read_text())["nested_source_ids"]
                    if old_subset.exists() else subset_membership(plan))
        membership=_validated_membership(membership,plan,legacy=True)
    complete=[item for item in plan if valid_commit(source,item)]
    if not complete:
        raise ValueError("No complete sources to export")
    old_shared_path=source/"protected/shared_checksums.json"
    if old_shared_path.exists():
        verify_checksums(source/"public",json.loads(old_shared_path.read_text()))
    salt=secrets.token_hex(32)
    source_map={x["source_id"]:opaque(0,"source:"+x["source_id"],salt) for x in complete}
    realization_map={r["realization_id"]:opaque(0,"realization:"+r["realization_id"],salt)
                     for x in complete for r in x["realizations"]}
    target.mkdir(parents=True,exist_ok=True)
    atomic_json(target/"protected/export_state.json",{"status":"copying","source_provenance_sha256":source_hash})
    try:
        for path in sorted((source/"public/instruments").glob("*")):
            _copy_verified(path,target/"public/instruments"/path.name,digest(path))
        shared={p.relative_to(target/"public").as_posix():digest(p)
                for p in sorted((target/"public/instruments").glob("*"))}
        atomic_json(target/"protected/shared_checksums.json",shared)
        remapped=[]
        # Copy order must not leak the original rank through file timestamps or
        # directory insertion order, even though visible IDs have been rotated.
        for item in sorted(complete,key=lambda item:source_map[item["source_id"]]):
            old_sid=item["source_id"]; sid=source_map[old_sid]
            old_commit=json.loads((source/"protected/commits"/(old_sid+".json")).read_text())
            new_item=deepcopy(item); new_item["source_id"]=sid
            for r in new_item["realizations"]:
                r["realization_id"]=realization_map[r["realization_id"]]
            remapped.append(new_item)
            records=[]; files=[]
            for old_record in old_commit["records"]:
                r=deepcopy(old_record); rid=realization_map[r["realization_id"]]
                r["source_id"]=sid; r["realization_id"]=rid
                r.pop("sha256",None)
                for key in ("measurement","image"):
                    old_path=safe_relative(source/"public",r[key])
                    new_relative=Path("public/observations")/(rid+old_path.suffix)
                    new_path=target/new_relative; new_path.parent.mkdir(parents=True,exist_ok=True)
                    _copy_verified(old_path,new_path,old_commit["checksums"]["public/"+r[key]])
                    r[key]=new_relative.relative_to("public").as_posix()
                    files.append(new_relative)
                records.append(r)
            label=json.loads((source/"protected/labels"/(old_sid+".json")).read_text())
            new_truth=Path("protected/truth")/(sid+".fits")
            (target/new_truth).parent.mkdir(parents=True,exist_ok=True)
            _copy_verified(safe_relative(source,label["truth"]),target/new_truth,
                           old_commit["checksums"][label["truth"]])
            label["source_id"]=sid; label["truth"]=new_truth.as_posix()
            for corruption in label["corruptions"]:
                corruption["realization_id"]=realization_map[corruption["realization_id"]]
            label_path=Path("protected/labels")/(sid+".json")
            atomic_json(target/label_path,label)
            files += [new_truth,label_path]
            atomic_json(target/"protected/commits"/(sid+".json"),{
                "source_id":sid,"records":records,
                "checksums":{p.as_posix():digest(target/p) for p in files}})
        mapped_membership={s:{n:[source_map[x] for x in ids] for n,ids in levels.items()
                              if set(ids)<=source_map.keys()} for s,levels in membership.items()}
        atomic_json(target/"protected/subset_membership.json",mapped_membership)
        atomic_json(target/"protected/plan.json",remapped)
        atomic_json(target/"protected/export_mapping.json",{
            "source_ids":source_map,"realization_ids":realization_map,"export_id_key":salt})
        atomic_json(target/"protected/source_provenance.json",old_provenance)
        for name in ("runs","source_snapshot"):
            if (source/"protected"/name).is_dir():
                shutil.copytree(source/"protected"/name,target/"protected/source_history"/name)
        for name in ("last_run.json","failure.json","source_manifest.json"):
            path=source/"protected"/name
            if path.exists():
                _copy_verified(path,target/"protected/source_history"/name,digest(path))
        provenance={"schema_version":PUBLIC_SCHEMA,"operation":"versioned_export_no_resimulation",
                    "config":old_provenance["config"],"source_provenance_sha256":source_hash,
                    "source_location":str(source),"export_code_sha256":code_digest(),
                    "export_source_manifest":source_manifest(),"versions":versions(),
                    "created_utc":datetime.now(timezone.utc).isoformat(),
                    "shared_integrity":("verified original hashes" if old_shared_path.exists() else
                        "legacy shared products were not originally checksummed; hashes established at export"),
                    "membership_integrity":("verified protected membership hash" if old_provenance.get("schema_version")==PUBLIC_SCHEMA else
                        "legacy membership lacks historical hash; checked groups, counts, nesting and original prefix definition against protected plan"),
                    "security_limit":"Prior v1 target disclosure cannot be undone. Rotated IDs and ordering block the direct index attack, but known legacy products remain recognizable by content/fingerprints. Do not use this legacy bank as a secrecy-protected blind benchmark."}
        records=publish_indexes(target,remapped,mapped_membership)
        immutable=[target/"protected"/name for name in (
            "plan.json","subset_membership.json","shared_checksums.json",
            "export_mapping.json","source_provenance.json")]
        immutable+=list((target/"public/instruments").glob("*"))
        immutable+=[p for p in (target/"protected/source_history").rglob("*") if p.is_file()]
        provenance["initial_checksums"]={p.relative_to(target).as_posix():digest(p) for p in immutable}
        provenance["spec_sha256"]=canonical_hash(provenance)
        atomic_json(target/"protected/provenance.json",provenance)
        if digest(provenance_path)!=source_hash:
            raise RuntimeError("Source provenance changed during export")
        summary={"status":"complete","operation":"versioned_export_no_resimulation",
                 "sources":len(remapped),"realizations":len(records),"schema_version":PUBLIC_SCHEMA,
                 "source_provenance_sha256":source_hash,"source_modified":False}
        atomic_json(target/"protected/export_state.json",summary)
        atomic_json(target/"protected/runs/export.json",{**summary,**provenance})
        return summary
    except BaseException as error:
        atomic_json(target/"protected/export_state.json",{"status":"failed","error":repr(error),
                    "source_provenance_sha256":source_hash,"source_modified":False})
        raise
