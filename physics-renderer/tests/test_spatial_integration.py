"""Bounded integration regressions; no source fitting or dataset qualification."""
from dataclasses import replace, asdict
import json
from pathlib import Path
import numpy as np
import pytest
from astropy import units as u
from scipy.integrate import quad, dblquad

from hii_sim.physics import render, intensity, em_profile, RJ_JY_SR
from hii_sim.schema import Source, Sky, Grid, Instrument, load_config, render_grid
from hii_sim.instrument import operator, expected_measurement, observe, noise_sigma_real, independent_fourier_cells
from hii_sim.dataset import make_plan
from hii_sim.dataset import generate
from hii_sim.loaders import NPEInputDataset, PairedNPEDataset
from hii_sim.io import digest

V3="configs/compact_proxy_gl4_v3.json"


def joint_thermal_norm(delta,grid,inst):
    """Rank-one covariance inverse, independent of production whitening code."""
    weights,_=operator(grid,inst)
    unique,self_mode=independent_fourier_cells(grid.npix)
    z=delta/noise_sigma_real(grid,inst)[:,None,None]
    rho=inst.noise_frequency_correlation
    total=0.
    for y,x in zip(*np.where(unique)):
        selected=weights[:,y,x]>0
        count=int(selected.sum())
        if not count:continue
        values=z[selected,y,x]/(np.sqrt(2) if self_mode[y,x] else 1)
        components=(values.real,) if self_mode[y,x] else (values.real,values.imag)
        for component in components:
            total+=(np.sum(component**2)-rho*np.sum(component)**2/(1-rho+count*rho))/(1-rho)
    return float(np.sqrt(max(0,total)))


CASES=[
    ("base_compact",Source(5e5,5000,12,.5,37),Sky(east_arcsec=.6,north_arcsec=-.7)),
    ("base_extended",Source(5e5,5000,60,1,0),Sky(east_arcsec=10,north_arcsec=-10)),
    ("turnover_compact",Source(1e7,5000,12,.5,37),Sky(east_arcsec=.6,north_arcsec=-.7)),
    ("turnover_extended",Source(1e7,12000,60,.5,57),Sky(east_arcsec=10,north_arcsec=-10)),
]
# Cover pixel-center, edge, corner and off-grid source centers; two orientations.
CASES += [(f"phase_{phase}_{em:g}",Source(em,5000,12,.5,pa),Sky(east_arcsec=e,north_arcsec=n))
          for phase,e,n,pa in (("center",0,0,0),("edge",1.25,0,90),("corner",1.25,-1.25,113))
          for em in (5e5,1e7)]


@pytest.mark.parametrize("name,source,sky",CASES,ids=[c[0] for c in CASES])
def test_gaussian_gl4_convergence_in_joint_noise_units(name,source,sky,record_property):
    config=load_config(V3);grid=render_grid(config,source.profile);inst=Instrument(**config["instrument"])
    def cube(g):return render(source,g,sky,inst.frequencies_ghz,inst.bandwidths_mhz,inst.frequency_quadrature)
    gl4=cube(grid);gl8=cube(replace(grid,subpixels=8));gl12=cube(replace(grid,subpixels=12))
    mid4=cube(replace(grid,spatial_rule="midpoint"))
    ref=expected_measurement(gl12,grid,inst)
    metrics={"case":name,"source":asdict(source),"sky":asdict(sky),
             "GL4_joint_error_vs_GL12":joint_thermal_norm(expected_measurement(gl4,grid,inst)-ref,grid,inst),
             "GL8_joint_error_vs_GL12":joint_thermal_norm(expected_measurement(gl8,grid,inst)-ref,grid,inst),
             "midpoint4_joint_error_vs_GL12":joint_thermal_norm(expected_measurement(mid4,grid,inst)-ref,grid,inst),
             "GL4_pixel_relative_L2":float(np.linalg.norm(gl4-gl12)/np.linalg.norm(gl12)),
             "GL4_max_finite_field_flux_relative_error":float(np.max(abs(gl4.sum(axis=(1,2))/gl12.sum(axis=(1,2))-1)))}
    record_property("measured_integration_errors",json.dumps(metrics))
    assert metrics["GL4_joint_error_vs_GL12"]<.1  # Existing declared joint-noise budget, unchanged.
    assert metrics["GL8_joint_error_vs_GL12"]<1e-8  # Numerical reference refinement, not an accuracy claim for the physics law.
    assert metrics["GL4_joint_error_vs_GL12"]<metrics["midpoint4_joint_error_vs_GL12"]
    if name in ("base_compact","turnover_compact","turnover_extended"):
        assert metrics["midpoint4_joint_error_vs_GL12"]>.1  # Reproduce the diagnosed defect.
    assert np.isfinite(gl4).all() and gl4.min()>=0


@pytest.mark.parametrize("profile",["gaussian_em","ellipsoid_em"])
def test_midpoint_default_preserves_historical_values_bitwise(profile):
    source=Source(5e5,8000,12,.5,37,profile);sky=Sky(east_arcsec=.6,north_arcsec=-.7)
    grid=Grid(16,2.5,4);frequency=[1.06,1.95];width=[8,8]
    # Historical midpoint algorithm retained independently as a compatibility oracle.
    east,north=grid.coordinates()
    offsets=((np.arange(4)+.5)/4-.5)*grid.pixel_arcsec
    field=np.stack([em_profile(source,east+dx-sky.east_arcsec,north+dy-sky.north_arcsec)
                    for dy in offsets for dx in offsets])
    nodes,weights=np.polynomial.legendre.leggauss(3);legacy=[]
    for f,b in zip(frequency,width):
        plane=np.zeros((16,16))
        for node,weight in zip(nodes,weights):
            plane+=weight/2*intensity(field*u.pc/u.cm**6,8000*u.K,(f+node*b/2000)*u.GHz).value.mean(axis=0)
        legacy.append(plane)
    np.testing.assert_array_equal(render(source,grid,sky,frequency,width),legacy)
    np.testing.assert_array_equal(render(source,replace(grid,spatial_rule="midpoint"),sky,frequency,width),legacy)


def test_gaussian_pixel_integrates_nonlinear_transfer_before_averaging(record_property):
    source=Source(1e7,5000,12,.5,37);sky=Sky(east_arcsec=.6,north_arcsec=-.7,background_rj_k=100)
    grid=Grid(16,2.5,8,"gauss_legendre");f=1.06
    angle=np.deg2rad(source.pa_deg)
    gaunt=np.logaddexp(5.960-np.sqrt(3)/np.pi*np.log(f*(source.te_k/1e4)**-1.5),1.)
    tau0=.05468*source.te_k**-1.5*f**-2*source.em0_pc_cm6*gaunt
    def tau(e,n):
        e-=sky.east_arcsec;n-=sky.north_arcsec
        major=n*np.cos(angle)+e*np.sin(angle);minor=e*np.cos(angle)-n*np.sin(angle)
        return tau0*np.exp(-4*np.log(2)*((major/12)**2+(minor/6)**2))
    lo,hi=-grid.pixel_arcsec/2,grid.pixel_arcsec/2
    value,error=dblquad(lambda n,e:-np.expm1(-tau(e,n)),lo,hi,lambda e:lo,lambda e:hi,epsabs=1e-12,epsrel=1e-12)
    factor=RJ_JY_SR*(f*1e9)**2*(source.te_k-sky.background_rj_k)
    reference=value/grid.pixel_arcsec**2*factor
    actual=render(source,grid,sky,[f],[0],1)[0,8,8]
    relative_error=float(abs(actual/reference-1))
    average_tau=dblquad(lambda n,e:tau(e,n),lo,hi,lambda e:lo,lambda e:hi)[0]/grid.pixel_arcsec**2
    wrong=(-np.expm1(-average_tau))*factor
    record_property("adaptive_pixel_reference",json.dumps({"relative_error":relative_error,
        "reference_integration_error_estimate":error,"transfer_of_average_tau_relative_bias":float(wrong/reference-1)}))
    assert relative_error<1e-11
    assert abs(wrong/reference-1)>1e-4


@pytest.mark.parametrize("em,temperature",[(1e4,12000),(5e5,5000),(1e7,5000)])
def test_integrated_gaussian_flux_matches_independent_radial_quadrature(em,temperature,record_property):
    source=Source(em,temperature,12,.5,113);grid=Grid(64,2.5,4,"gauss_legendre")
    f=1.06;sky=Sky(east_arcsec=1.25,north_arcsec=-1.25)
    plane=render(source,grid,sky,[f],[0],1)[0]
    reference_plane=render(source,replace(grid,subpixels=8),sky,[f],[0],1)[0]
    gaunt=np.logaddexp(5.960-np.sqrt(3)/np.pi*np.log(f*(temperature/1e4)**-1.5),1.)
    tau0=.05468*temperature**-1.5*f**-2*em*gaunt
    integral,err=quad(lambda x:-np.expm1(-tau0*np.exp(-x)),0,80,epsabs=1e-13,epsrel=1e-12)
    reference=integral*RJ_JY_SR*(f*1e9)**2*temperature*np.pi*12*6/(4*np.log(2))*u.arcsec.to(u.rad)**2
    error=float(abs(plane.sum()*grid.omega_sr/reference-1))
    refined_error=float(abs(reference_plane.sum()*grid.omega_sr/reference-1))
    record_property("GL4_independent_flux_relative_error",error)
    record_property("GL8_independent_flux_relative_error",refined_error)
    record_property("radial_reference_error_estimate",err)
    # GL8 anchors the independent high-precision normalization reference.
    # GL4 accuracy is judged in measured-space by the unchanged .1 noise gate.
    assert refined_error<1e-8


def test_new_config_preserves_prior_plan_and_measurement_noise_operator():
    old=load_config("configs/compact_proxy_v2.json");new=load_config(V3)
    without_rules=json.loads(json.dumps(new));without_rules["grid"].pop("spatial_rule");without_rules.pop("profile_spatial_rules")
    assert without_rules==old
    assert make_plan(new,16)==make_plan(old,16)
    old_grid,new_grid=Grid(**old["grid"]),Grid(**new["grid"]);inst=Instrument(**old["instrument"])
    for a,b in zip(operator(old_grid,inst),operator(new_grid,inst)):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(noise_sigma_real(old_grid,inst),noise_sigma_real(new_grid,inst))
    signal=np.ones((6,96,96))
    old_observation,old_hidden=observe(signal,old_grid,inst,51)
    new_observation,new_hidden=observe(signal,new_grid,inst,51)
    for key in old_observation:np.testing.assert_array_equal(old_observation[key],new_observation[key])
    assert old_hidden==new_hidden
    assert render_grid(new,"gaussian_em").spatial_rule=="gauss_legendre"
    ellipsoid=render_grid(new,"ellipsoid_em")
    assert ellipsoid.spatial_rule=="midpoint" and ellipsoid.subpixels==16


def test_invalid_rule_order_and_hard_boundary_usage_rejected(tmp_path):
    for kwargs in ({"subpixels":True},{"subpixels":4.5},{"spatial_rule":"unknown"}):
        with pytest.raises(ValueError):Grid(**kwargs)
    with pytest.raises(ValueError,match="smooth Gaussian"):
        render(Source(5e5,8000,12,.5,37,"ellipsoid_em"),Grid(16,2.5,4,"gauss_legendre"),Sky(),[1.44])
    config=json.loads(Path(V3).read_text());config.pop("profile_spatial_rules")
    path=tmp_path/"invalid.json";path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match="Hard-boundary"):load_config(path)


def test_integration_provenance_is_protected_and_resume_stays_strict(tmp_path):
    config=load_config(V3)
    config["grid"].update(npix=16,pixel_arcsec=10.)
    config["instrument"].update(frequencies_ghz=[1.31,1.82],bandwidths_mhz=[8.,8.],
                               min_baseline_m=300.,max_short_spacing_ratio=None)
    config["challenge_sources"]=1
    path=tmp_path/"config.json";path.write_text(json.dumps(config))
    root=tmp_path/"bank";generate(root,path,1,max_seconds=120,max_gib=1)
    paired=PairedNPEDataset(root,challenge_evaluation=True)
    for i in range(len(paired)):
        inputs,label=paired[i]
        sid=inputs["bookkeeping"]["source_id"]
        hidden=json.loads((root/"protected/labels"/(sid+".json")).read_text())
        is_gaussian=label["parameter_definitions"]["profile"]=="gaussian_em"
        assert hidden["render_spatial_rule"]==("gauss_legendre" if is_gaussian else "midpoint")
        assert hidden["render_subpixels"]==(4 if is_gaussian else 16)
        assert label["parameters"]=={key:hidden["parameters"][key] for key in label["target_order"]}
    public=NPEInputDataset(root/"public")
    assert len(public)==2
    assert "spatial_rule" not in asdict(public[0]["features"]["observing"])
    for file in (root/"public").rglob("*.json"):
        text=file.read_text()
        assert "spatial_rule" not in text and "subpixels" not in text and "gauss_legendre" not in text
    provenance=root/"protected/provenance.json";before=digest(provenance)
    config["grid"]["spatial_rule"]="midpoint"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match="Resume refused"):generate(root,path,1,max_seconds=120,max_gib=1)
    assert digest(provenance)==before
