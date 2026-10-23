"""Numerical imaging controls and a frozen audit of the legacy extended case.

Analytic image fixtures are known truth only for this validation module. The
reconstructor receives measured Fourier data, weights, and declared settings.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import atomic_json, canonical_hash, source_manifest, versions
from .reconstruction import (aperture_flux_diagnostic, dirty_image, hogbom,
                             point_spread_function, reconstruct_measurements,
                             _circular_convolve)


def _fixture(n, pixel, kind, total_jy=.01):
    y,x = (np.indices((n,n))-n//2)*pixel
    if kind == "point":
        sky = np.zeros((n,n)); sky[n//2,n//2] = 1.
    elif kind == "gaussian_30x20":
        sky = np.exp(-4*np.log(2)*(((x-8)/30)**2+((y+10)/20)**2))
    elif kind == "gaussian_90":
        sky = np.exp(-4*np.log(2)*((x/90)**2+(y/90)**2))
    elif kind == "disk_50":
        sky = (np.hypot(x,y) <= 25).astype(float)
    else:
        raise ValueError(kind)
    return sky*(total_jy/sky.sum())


def _legacy_weights(n=96, pixel=2.5, include_dc=False):
    """Frozen v1 first-band operator, for the historical 2.423 mJy audit only."""
    uv=np.fft.fftfreq(n, np.deg2rad(pixel/3600))
    v,u=np.meshgrid(uv,uv,indexing="ij"); radius=np.hypot(u,v)
    sigma=np.deg2rad(25/3600)/np.sqrt(8*np.log(2))
    taper=np.exp(-2*np.pi**2*sigma*sigma*radius*radius)
    if include_dc:
        return taper
    rng=np.random.default_rng(9227); uniform=rng.uniform(size=(n,n))
    ids=(-np.arange(n))%n
    uniform=np.minimum(uniform,uniform[ids[:,None],ids[None,:]])
    mask=uniform < 1-np.sqrt(.5)
    wavelength=299792458./1.06e9
    return np.where(mask & (radius>=35/wavelength) & (radius<=3400/wavelength),taper,0.)


def _moments(image, pixel):
    """Positive-part image moments, reported explicitly as a morphology proxy."""
    a=np.maximum(image,0); total=a.sum()
    if total <= 0:
        return None
    y,x=(np.indices(a.shape)-np.array(a.shape)[:,None,None]//2)*pixel
    center=np.array([(a*x).sum(),(a*y).sum()])/total
    dx=x-center[0]; dy=y-center[1]
    covariance=np.array([[(a*dx*dx).sum(),(a*dx*dy).sum()],
                         [(a*dx*dy).sum(),(a*dy*dy).sum()]])/total
    major,minor=np.sqrt(np.maximum(np.linalg.eigvalsh(covariance)[::-1],0))*np.sqrt(8*np.log(2))
    return {"west_centroid_arcsec":float(center[0]),"north_centroid_arcsec":float(center[1]),
            "moment_equivalent_major_fwhm_arcsec":float(major),
            "moment_equivalent_minor_fwhm_arcsec":float(minor),
            "definition":"positive-part second moments; not a Gaussian fit"}


def _case(sky_jy_pixel, weights, pixel, *, name, niter=6000, threshold=1e-7):
    vis=np.fft.fft2(np.fft.ifftshift(sky_jy_pixel))
    measured=np.where(weights>0,vis,0.)
    dirty=dirty_image(measured[None],weights[None])[0]
    psf=point_spread_function(weights[None])[0]
    result=hogbom(dirty,psf,gain=.2,niter=niter,threshold=threshold,pixel_arcsec=pixel)
    model=result["components_jy"]
    predicted=np.fft.fft2(np.fft.ifftshift(model))
    supported=weights>0
    residual_vis=np.where(supported, measured-predicted,0.)
    reference=_circular_convolve(sky_jy_pixel,result["clean_beam"])
    y,x=np.indices(weights.shape)-np.array(weights.shape)[:,None,None]//2
    # Predeclared field-centred 100-arcsec aperture, independent of source labels.
    aperture=np.hypot(x,y)*pixel<=100.
    beam_scale=result["beam"]["major_fwhm_arcsec"]/pixel
    photometry=[aperture_flux_diagnostic(result,psf,aperture,beam_scale*f)
                for f in (.5,1.,1.5,2.,3.)]
    residual_image_check=dirty_image(residual_vis[None],weights[None])[0]
    full_psf_area=float(psf.sum())
    dc_present=bool(weights[0,0]>0)
    report={"name":name,"apparent_input_flux_jy":float(sky_jy_pixel.sum()),
            "clean_component_flux_jy":float(model.sum()),
            "positive_component_flux_jy":float(model[model>0].sum()),
            "negative_component_flux_jy":float(model[model<0].sum()),
            "outside_photometry_aperture_component_flux_jy":float(model[~aperture].sum()),
            "restored_model_integrated_flux_jy":float(result["restored_jy_clean_beam"].sum()/result["beam"]["area_pixels"]),
            "residual_peak_jy_dirty_beam":result["residual_peak_jy_dirty_beam"],
            "residual_rms_jy_dirty_beam":result["residual_rms_jy_dirty_beam"],
            "measurement_residual_relative_l2":float(np.linalg.norm(residual_vis[supported])/np.linalg.norm(measured[supported])),
            "weighted_measurement_residual_relative_l2":float(np.linalg.norm(residual_vis*weights)/np.linalg.norm(measured*weights)),
            "measurement_image_residual_closure_max_abs":float(np.max(np.abs(residual_image_check-result["residual_jy_dirty_beam"]))),
            "restored_model_vs_beam_convolved_truth_relative_l2":float(np.linalg.norm(result["restored_jy_clean_beam"]-reference)/np.linalg.norm(reference)),
            "restored_model_morphology":_moments(result["restored_jy_clean_beam"],pixel),
            "beam_convolved_truth_morphology":_moments(reference,pixel),
            "beam":result["beam"],"iterations":result["iterations"],
            "stop_reason":result["stop_reason"],"threshold_jy_dirty_beam":threshold,
            "zero_spacing_present":dc_present,"full_field_psf_area_pixels":full_psf_area,
            "aperture_radius_arcsec":100.,"aperture_flux_by_dirty_beam_radius":photometry}
    if dc_present:
        residual_flux=float(result["residual_jy_dirty_beam"].sum()/full_psf_area)
        report["full_field_residual_flux_jy"]=residual_flux
        report["component_plus_full_field_residual_flux_jy"]=float(model.sum()+residual_flux)
        report["full_field_flux_relative_error"]=float((model.sum()+residual_flux)/sky_jy_pixel.sum()-1)
    else:
        report["full_field_residual_flux_jy"]=None
        report["component_plus_full_field_residual_flux_jy"]=None
        report["total_flux_identifiable_without_extra_constraint"]=False
    return report, (dirty,result["restored_jy_clean_beam"],result["residual_jy_dirty_beam"],reference)


def validate_reconstruction(output_dir, *, grid=None, inst=None, widefield=False):
    """Write measured imaging diagnostics; optionally test the current operator."""
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    initial_manifest=source_manifest()
    reports=[]; panels=[]
    control_weights=_legacy_weights(include_dc=True)
    legacy_weights=_legacy_weights()
    for kind in ("point","gaussian_30x20","gaussian_90","disk_50"):
        truth=_fixture(96,2.5,kind)
        for label_name,w in (("complete_uv_with_dc_control",control_weights),
                             ("frozen_legacy_v1",legacy_weights)):
            report,images=_case(truth,w,2.5,name=f"{label_name}/{kind}")
            reports.append(report)
            if kind=="gaussian_90":
                panels.append((report["name"],images))
    # Exactly reproduce the historical case: first band, Gaussian PB, 3000
    # iterations, 1 microJy threshold. Separate intrinsic and apparent flux.
    truth=_fixture(96,2.5,"gaussian_90")
    y,x=(np.indices((96,96))-48)*2.5
    pb=np.exp(-4*np.log(2)*(x*x+y*y)/(42*60/1.06)**2)
    audit,images=_case(truth*pb,legacy_weights,2.5,name="historical_2p423mJy_case",
                       niter=3000,threshold=1e-6)
    audit["intrinsic_input_flux_jy"]=.01
    audit["primary_beam_attenuated_flux_jy"]=float((truth*pb).sum())
    # All measured baselines miss DC: adding any constant image is in the null
    # space. This is direct evidence that total flux cannot be learned from
    # these data without additional measurement or a prior/support constraint.
    offset=np.full(truth.shape,.005/truth.size)
    delta=np.fft.fft2(np.fft.ifftshift(offset))
    audit["constant_5mJy_addition_measured_visibility_max_abs_jy"]=float(np.max(np.abs(delta[legacy_weights>0])))
    reports.append(audit);panels.append((audit["name"],images))
    if grid is not None and inst is not None:
        from .instrument import operator
        weights,beams=operator(grid,inst)
        for kind in ("point","gaussian_30x20","gaussian_90","disk_50"):
            truth=_fixture(grid.npix,grid.pixel_arcsec,kind)*beams[0]
            report,images=_case(truth,weights[0],grid.pixel_arcsec,name=f"current_operator/{kind}")
            report["configured_thermal_rms_jy_dirty_beam"]=float(inst.rms_jy_dirty_beam)
            report["noiseless_dirty_peak_over_configured_thermal_rms"]=(float(np.max(images[0])/inst.rms_jy_dirty_beam)
                if inst.rms_jy_dirty_beam>0 else None)
            report["diagnostic_threshold_is_not_production_noise_threshold"]=True
            reports.append(report)
        configuration={"grid":asdict(grid),"instrument":asdict(inst)}
    else:
        configuration={"current_operator":"not requested"}
    if widefield:
        from dataclasses import replace
        from .schema import Grid, Instrument
        from .instrument import operator
        wide_grid=Grid(1024,2.5,1)
        wide_inst=replace(inst if inst is not None else Instrument(),
            frequencies_ghz=(1.06,),bandwidths_mhz=(128.,),min_baseline_m=35.)
        weights,beams=operator(wide_grid,wide_inst)
        truth=_fixture(wide_grid.npix,wide_grid.pixel_arcsec,"gaussian_90")*beams[0]
        case,_images=_case(truth,weights[0],wide_grid.pixel_arcsec,
                          name="widefield_1024_short_baselines/gaussian_90",niter=2000,threshold=1e-6)
        uv=np.fft.fftfreq(wide_grid.npix,np.deg2rad(wide_grid.pixel_arcsec/3600))
        vv,uu=np.meshgrid(uv,uv,indexing="ij")
        baselines=np.hypot(uu,vv)[weights[0]>0]*299792458./1.06e9
        case["actual_min_sampled_baseline_m"]=float(baselines.min())
        case["actual_max_sampled_baseline_m"]=float(baselines.max())
        case["scope"]="larger field samples more short spacings; still synthetic Fourier cells and no DC"
        reports.append(case)
        configuration["widefield_audit"]={"grid":asdict(wide_grid),"instrument":asdict(wide_inst)}
    configuration["frozen_legacy_audit"]={"grid_npix":96,"pixel_arcsec":2.5,
        "frequency_ghz":1.06,"min_baseline_m":35.,"max_baseline_m":3400.,
        "sampling_fraction":.5,"sampling_seed":9227,"taper_fwhm_arcsec":25.,
        "legacy_only":True}
    report={"schema_version":"2.0","scope":"deterministic measurement-only image reconstruction checks; no CASA run",
            "versions":versions(),"source_manifest":initial_manifest,
            "source_files_changed_during_validation":initial_manifest!=source_manifest(),"configuration":configuration,
            "configuration_sha256":canonical_hash(configuration),"cases":reports}
    atomic_json(out/"reconstruction_validation.json",report)
    fig,axes=plt.subplots(len(panels),4,figsize=(13,3.2*len(panels)),constrained_layout=True)
    titles=("Dirty\n(mJy / dirty beam)","CLEAN model\n(mJy / clean beam)",
            "Residual\n(µJy / dirty beam)","Known sky + clean beam\n(mJy / clean beam)")
    # Physical x increases west on this array. Label that explicitly.
    edges=(-121.25,118.75,-121.25,118.75)
    for row,(name,images) in enumerate(panels):
        low=min(float(images[k].min())*1e3 for k in (0,1,3))
        high=max(float(images[k].max())*1e3 for k in (0,1,3))
        for col,im in enumerate(images):
            scale=1e6 if col==2 else 1e3
            limits={} if col==2 else {"vmin":low,"vmax":high}
            art=axes[row,col].imshow(im*scale,origin="lower",extent=edges,cmap="RdBu_r",**limits)
            axes[row,col].set(xlabel="West offset (arcsec)",ylabel="North offset (arcsec)",
                               title=titles[col] if row==0 else None)
            axes[row,col].title.set_fontsize(10)
            fig.colorbar(art,ax=axes[row,col],shrink=.8)
        label_name={"complete_uv_with_dc_control/gaussian_90":"Complete coverage + DC",
                    "frozen_legacy_v1/gaussian_90":"Legacy sparse coverage",
                    "historical_2p423mJy_case":"Original 2.423 mJy case"}[name]
        axes[row,0].text(.02,.97,label_name,transform=axes[row,0].transAxes,va="top",fontsize=7,
                         bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
    fig.savefig(out/"reconstruction_validation.png",dpi=150);plt.close(fig)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    parser.add_argument("--config")
    parser.add_argument("--widefield",action="store_true",help="Include bounded 1024-pixel short-spacing comparison")
    args=parser.parse_args()
    if args.config:
        from .schema import Grid,Instrument,load_config
        config=load_config(args.config)
        result=validate_reconstruction(args.out,grid=Grid(**config["grid"]),inst=Instrument(**config["instrument"]),widefield=args.widefield)
    else:
        result=validate_reconstruction(args.out,widefield=args.widefield)
    print(json.dumps({"cases":len(result["cases"]),"output":args.out}))
