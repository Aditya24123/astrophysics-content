"""Reconstruct one public observation without reading protected data or labels."""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

from hii_sim.io import atomic_json, digest, source_manifest, versions
from hii_sim.loaders import PredictorDataset
from hii_sim.reconstruction import reconstruct_measurements


def reconstruct_sample(public_root, output_dir, *, index=0, split=None, niter=3000,
                       threshold_sigma=3., allow_legacy_audit=False):
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"reconstruction.fits").exists() or (output/"reconstruction.json").exists():
        raise FileExistsError("Use a new output directory; existing reconstructions are preserved")
    dataset=PredictorDataset(public_root,split=split,verify=True,allow_legacy=allow_legacy_audit)
    item=dataset[index]; meta=item["metadata"]; weights=meta["weights"]
    vis=np.zeros(weights.shape,dtype=complex)
    stored=item["observations"]["vis_jy"]
    if stored.size!=np.count_nonzero(weights):
        raise ValueError("Measurements do not match public sampling support")
    vis[weights>0]=stored
    result=reconstruct_measurements(vis,weights,meta["grid"]["pixel_arcsec"],
        rms_jy_dirty_beam=meta["instrument"]["rms_jy_dirty_beam"],
        niter=niter,threshold_sigma=threshold_sigma)
    wcs=fits.Header.fromstring(meta["image"]["wcs"])
    hdus=[fits.PrimaryHDU()]
    hdus[0].header["ALGO"]="HOGBOM"
    hdus[0].header["PBCOR"]=False
    hdus[0].header["HISTORY"]="Reconstructed from public measured visibilities only"
    descriptions=[]
    for i,channel in enumerate(result["channels"]):
        base=wcs.copy();base["FREQHZ"]=meta["image"]["frequencies_hz"][i]
        base["BANDWID"]=meta["image"]["bandwidths_hz"][i]
        for suffix,array,unit,beam_type in (
            ("MODEL",channel["restored_jy_clean_beam"],"Jy/beam","CLEAN"),
            ("COMP",channel["components_jy"],"Jy/pixel",None),
            ("RESID",channel["residual_jy_dirty_beam"],"Jy/beam","DIRTY"),
            ("DIRTY",result["dirty_jy_dirty_beam"][i],"Jy/beam","DIRTY"),
            ("PSF",result["psf"][i],"1",None),
            ("CBEAM",channel["clean_beam"],"1",None)):
            header=base.copy();header["BUNIT"]=unit
            if beam_type:
                header["BEAMTYPE"]=beam_type
            if beam_type=="CLEAN":
                header["BMAJ"]=channel["beam"]["major_fwhm_arcsec"]/3600
                header["BMIN"]=channel["beam"]["minor_fwhm_arcsec"]/3600
                header["BPA"]=channel["beam"]["pa_deg"]
            if beam_type=="DIRTY":
                header["COMMENT"]="Unit-peak dirty PSF; do not apply Gaussian beam-area photometry"
            hdus.append(fits.ImageHDU(array,header=header,name=f"CH{i:03}_{suffix}"))
        descriptions.append({key:channel[key] for key in ("beam","iterations","stop_reason",
            "residual_peak_jy_dirty_beam","residual_rms_jy_dirty_beam")})
    fits.HDUList(hdus).writeto(output/"reconstruction.fits",checksum=True)
    record=dataset.records[index]
    report={"schema_version":"2.0","input_bookkeeping":item["bookkeeping"],
        "input_observation_sha256":record["sha256"],
        "shared_metadata_sha256":dataset.shared,
        "algorithm":result["policy"],"channels":descriptions,
        "versions":versions(),"source_manifest":source_manifest(),
        "outputs":{"reconstruction.fits":digest(output/"reconstruction.fits")},
        "legacy_input_audit_only":dataset.legacy,
        "photometry_caution":"MODEL and RESID have different beam units; no missing-spacing flux correction",
        "truth_or_labels_read":False}
    atomic_json(output/"reconstruction.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--public",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--index",type=int,default=0)
    parser.add_argument("--split")
    parser.add_argument("--niter",type=int,default=3000)
    parser.add_argument("--threshold-sigma",type=float,default=3.)
    parser.add_argument("--allow-legacy-audit",action="store_true")
    args=parser.parse_args()
    result=reconstruct_sample(args.public,args.out,index=args.index,split=args.split,
        niter=args.niter,threshold_sigma=args.threshold_sigma,allow_legacy_audit=args.allow_legacy_audit)
    print(f"Wrote measurement-only reconstruction for {len(result['channels'])} frequency channels")
