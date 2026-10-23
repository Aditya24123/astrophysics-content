"""Independent numerical checks for the phase-one forward model."""
import json
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp
from astropy import units as u
from astropy.constants import c, k_B

from .instrument import expected_measurement, observe, operator
from .physics import em_profile, intensity, optical_depth, render
from .reconstruction import dirty_image, hogbom, point_spread_function
from .schema import Grid, Instrument, Sky, Source, load_config, render_grid
from .io import code_digest, versions, digest
from .sensitivity import representative_sensitivity


def _independent_flux(source, frequency_ghz):
    """Radial quadrature of the analytic Gaussian-EM profile (Jy)."""
    freq = frequency_ghz*u.GHz
    area_scale = (np.pi*source.major_fwhm_arcsec*source.axis_ratio*source.major_fwhm_arcsec
                  /(4*np.log(2)))*(u.arcsec.to(u.rad)**2)
    def integrand(x):
        em = source.em0_pc_cm6*np.exp(-x)
        return intensity(em*u.pc/u.cm**6, source.te_k*u.K, freq).to_value(u.Jy/u.sr)
    value, error = quad(integrand, 0, 40, epsabs=0, epsrel=1e-10)
    return value*area_scale, error*area_scale


def validate(config_path, output_dir):
    config = load_config(config_path)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    source = Source(1.2e5, 8000, 32, .72, 37)
    grid = render_grid(config,source.profile)
    inst = Instrument(**config["instrument"])
    sky = Sky(ra_deg=180, dec_deg=30)
    checks = {}

    # Independent ODE solution of dT_b/dtau = Te - T_b. Store measured errors.
    ode_errors = []
    for tau in (1e-8, 0.1, 1, 20):
        sol = solve_ivp(lambda _x, y: [source.te_k-y[0]], (0, tau), [0.0],
                        rtol=1e-12, atol=1e-12)
        closed = source.te_k*(-np.expm1(-tau))
        ode_errors.append({"tau": tau, "absolute_error_k": float(abs(sol.y[0,-1]-closed)),
                           "relative_error": float(abs(sol.y[0,-1]-closed)/closed)})
        assert abs(sol.y[0, -1]-closed) < max(1e-8, closed*2e-10)
    checks["transfer_ode_errors"] = ode_errors
    checks["transfer_ode_max_relative_error"] = max(x["relative_error"] for x in ode_errors)
    checks["transfer_ode_acceptance"] = "absolute error < max(1e-8 K, closed_form*2e-10)"

    nu = 1.44*u.GHz
    tau = optical_depth(np.array([1e-6, 1e9])*u.pc/u.cm**6, source.te_k*u.K, nu)
    assert tau[1] > tau[0] > 0
    tbthin = intensity(1e-8*u.pc/u.cm**6, source.te_k*u.K, nu).to_value(u.Jy/u.sr)
    tbthick = intensity(1e12*u.pc/u.cm**6, source.te_k*u.K, nu).to_value(u.Jy/u.sr)
    rayleigh_jeans_factor = (2*k_B*nu.to(u.Hz)**2/c**2/u.sr).to_value(u.Jy/u.sr/u.K)
    assert abs(tbthick/rayleigh_jeans_factor/source.te_k-1) < 1e-9
    checks["optically_thick_tb_over_te"] = tbthick/rayleigh_jeans_factor/source.te_k
    checks["optically_thin_tb_positive"] = bool(tbthin > 0)

    # Independent Mezger-Henderson opacity approximation (NRAO Essential Radio
    # Astronomy Eq. 4.60), distinct from the Draine Gaunt-factor expression.
    nrao_errors = []
    for temperature in (5000., 8000., 12000.):
        frequencies = np.array([1.0, 1.44, 1.95, 5.0, 8.0])
        exact = optical_depth(1e5*u.pc/u.cm**6, temperature*u.K, frequencies*u.GHz)
        mh = 3.28e-7*(temperature/1e4)**-1.35*frequencies**-2.1*1e5
        nrao_errors.extend((exact/mh-1).tolist())
    assert max(abs(x) for x in nrao_errors) < .08
    checks["independent_nrao_mezger_henderson_opacity_relative_error_range"] = [
        float(min(nrao_errors)),float(max(nrao_errors))]

    cube = render(source, grid, sky, inst.frequencies_ghz, inst.bandwidths_mhz,
                  inst.frequency_quadrature)
    assert np.isfinite(cube).all() and np.all(cube >= 0)
    checks["emission_finite_nonnegative"] = True
    tbrender = intensity(source.em0_pc_cm6*u.pc/u.cm**6, source.te_k*u.K, nu)
    astropy_tb = tbrender.to_value(u.K, equivalencies=u.brightness_temperature(nu))
    transfer_tb = source.te_k*(-np.expm1(-optical_depth(source.em0_pc_cm6*u.pc/u.cm**6,
                                                        source.te_k*u.K,nu)))
    checks["astropy_rj_temperature_relative_error"] = float(abs(astropy_tb-transfer_tb)/transfer_tb)
    omega = grid.omega_sr
    numerical_flux = cube.sum(axis=(-2,-1))*omega
    nodes, node_weights = np.polynomial.legendre.leggauss(7)
    ref = np.array([sum(weight/2*_independent_flux(source, f+node*b/2000)[0]
                        for node, weight in zip(nodes, node_weights))
                    for f, b in zip(inst.frequencies_ghz, inst.bandwidths_mhz)])
    # Finite image is many FWHM across, so all omitted Gaussian tails are negligible.
    errors = np.abs(numerical_flux-ref)/np.maximum(ref, 1e-30)
    assert np.all(errors < 0.025), f"Pixel-integrated flux differs from independent quadrature: {errors}"
    checks["flux_quadrature_relative_error_by_channel"] = errors.tolist()
    checks["integrated_flux_jy_by_channel"] = numerical_flux.tolist()
    spatial_errors = {}
    for n, pixel in ((64,2.5),(128,1.25),(256,.625)):
        test_grid=Grid(n,pixel,4)
        plane=render(source,test_grid,sky,[1.44],[0],5)[0]
        flux=plane.sum()*test_grid.omega_sr
        exact=_independent_flux(source,1.44)[0]
        spatial_errors[str(pixel)]=float(abs(flux-exact)/exact)
    assert max(spatial_errors.values()) < 1e-7
    checks["spatial_grid_flux_relative_error_by_pixel_arcsec"] = spatial_errors

    # Subpixel convergence of nonlinear transfer, assessed against 8x rendering.
    source_grid = Grid(npix=96, pixel_arcsec=2.5, subpixels=8)
    reference = render(source, source_grid, sky, [1.44], [128], 5)[0]
    conv = {}
    for nsub in (1, 2, 4):
        test = render(source, Grid(96, 2.5, nsub), sky, [1.44], [128], 5)[0]
        conv[str(nsub)] = float(np.linalg.norm(test-reference)/np.linalg.norm(reference))
    assert conv["4"] < conv["2"] < conv["1"]
    checks["subpixel_relative_l2_error_vs_8"] = conv
    q3 = render(source, Grid(96,2.5,4), sky, [1.44], [128], 3)[0]
    q5 = render(source, Grid(96,2.5,4), sky, [1.44], [128], 5)[0]
    checks["spectral_quadrature_3_vs_5_relative_l2"] = float(np.linalg.norm(q3-q5)/np.linalg.norm(q5))

    weights, pb = operator(grid, inst)
    rng1 = observe(cube, grid, inst, 145)[0]["vis_jy"]
    rng2 = observe(cube, grid, inst, 145)[0]["vis_jy"]
    assert np.array_equal(rng1, rng2)
    psf = point_spread_function(weights)
    assert np.allclose(psf.max(axis=(-2,-1)), 1, atol=1e-12)
    checks["repeatability_exact"] = True
    checks["psf_peak"] = psf.max(axis=(-2,-1)).tolist()
    checks["uv_sample_fraction"] = float((weights > 0).mean())
    checks["primary_beam_center"] = float(pb[:, grid.npix//2, grid.npix//2].min())

    # Point-source flux and missing-short-spacing response use the measurement operator.
    point = np.zeros((len(inst.frequencies_ghz), grid.npix, grid.npix))
    point[:, grid.npix//2, grid.npix//2] = 0.01/omega
    point_vis = expected_measurement(point, grid, inst)
    point_dirty = dirty_image(point_vis, weights)
    point_peak = point_dirty[:, grid.npix//2, grid.npix//2]
    assert np.allclose(point_peak, .01, rtol=1e-10, atol=1e-10)
    checks["point_source_peak_jy_per_dirty_beam"] = point_peak.tolist()
    line=psf[0,grid.npix//2,grid.npix//2:]
    half=np.flatnonzero(line[1:]<=.5)
    psf_fwhm_arcsec=float(2*(half[0]+1)*grid.pixel_arcsec) if len(half) else None
    checks["point_psf_x_slice_fwhm_pixel_estimate_arcsec"] = psf_fwhm_arcsec
    clean=hogbom(point_dirty[0],psf[0],gain=.2,niter=200,threshold=1e-8)
    checks["point_clean_component_flux_jy"] = float(clean["components_jy"].sum())
    checks["point_clean_residual_max_jy_per_dirty_beam"] = float(
        np.max(np.abs(clean["residual_jy_dirty_beam"])))
    assert abs(checks["point_clean_component_flux_jy"]-.01)<2e-7
    assert checks["point_clean_residual_max_jy_per_dirty_beam"]<2e-8
    extended_grid = Grid(npix=96, pixel_arcsec=2.5, subpixels=4)
    extended_omega = extended_grid.omega_sr
    extended_east, extended_north = extended_grid.coordinates()
    extended = np.exp(-4*np.log(2)*((extended_east/90)**2+(extended_north/90)**2))
    extended *= .01/(extended.sum()*extended_omega)
    extended_cube = np.broadcast_to(
        extended, (len(inst.frequencies_ghz), extended_grid.npix, extended_grid.npix)
    ).copy()
    # Deliberate legacy cutout comparison, not the guarded revised scope.
    low = replace_instrument(inst, min_baseline_m=35, max_short_spacing_ratio=None)
    high = replace_instrument(inst, min_baseline_m=1400, max_short_spacing_ratio=None)
    checks["legacy_extended_fixture_scope"] = {
        "grid": asdict(extended_grid), "short_spacing_guard": "disabled explicitly for historical failure reproduction",
        "instruments": [asdict(low), asdict(high)]}
    recovered = []
    extended_dirty = None
    extended_psf = None
    for candidate in (low, high):
        w, _ = operator(extended_grid, candidate)
        d = dirty_image(expected_measurement(extended_cube, extended_grid, candidate), w)
        recovered.append(float(d[:, extended_grid.npix//2, extended_grid.npix//2].mean()))
        if candidate.min_baseline_m == low.min_baseline_m:
            extended_dirty = d
            extended_psf = point_spread_function(w)
    checks["extended_source_peak_low_vs_high_min_baseline"] = recovered
    checks["extended_source_short_spacing_response_abs_ratio"] = (
        abs(recovered[1])/abs(recovered[0]))
    assert checks["extended_source_short_spacing_response_abs_ratio"] < .8
    extended_clean = hogbom(extended_dirty[0], extended_psf[0], gain=.2,
                            niter=3000, threshold=1e-6)
    checks["extended_source_input_flux_jy"] = .01
    checks["extended_source_hogbom_component_flux_jy"] = float(
        extended_clean["components_jy"].sum())
    checks["extended_source_hogbom_residual_max_jy_per_dirty_beam"] = float(
        np.max(np.abs(extended_clean["residual_jy_dirty_beam"])))
    assert 0 < checks["extended_source_hogbom_component_flux_jy"] < .01
    assert np.isfinite(checks["extended_source_hogbom_residual_max_jy_per_dirty_beam"])

    checks["sensitivity"] = representative_sensitivity(Grid(**config["grid"]), sky, inst,
        config.get("profile_subpixels"),config.get("profile_spatial_rules"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    axes[0].imshow(cube[2], origin="lower"); axes[0].set_title("Intrinsic Jy/sr")
    axes[1].imshow(dirty_image(observe(cube,grid,inst,145)[0]["vis_jy"],weights)[2],origin="lower")
    axes[1].set_title("Fast dirty image Jy/beam")
    axes[2].plot(inst.frequencies_ghz, numerical_flux, "o-", label="rendered pixel sum")
    axes[2].plot(inst.frequencies_ghz, ref, "x--", label="independent integral")
    axes[2].set(xlabel="Frequency (GHz)",ylabel="Integrated flux (Jy)"); axes[2].legend(fontsize=7)
    fig.savefig(out/"pilot_validation.png", dpi=160); plt.close(fig)
    report = {"schema_version":"2.0","scope":"development approximation validation; no CASA end-to-end run",
              "provenance": {"created_utc": datetime.now(timezone.utc).isoformat(),
                             "code_sha256": code_digest(), "versions": versions(),
                             "configuration_sha256": digest(config_path), "configuration": config,
                             "physics_validation_grid": asdict(grid)},
              "source":{"em0_pc_cm6":source.em0_pc_cm6,"te_k":source.te_k,
                        "major_fwhm_arcsec":source.major_fwhm_arcsec,"axis_ratio":source.axis_ratio,
                        "pa_deg":source.pa_deg},"checks":checks}
    (out/"validation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report


def replace_instrument(inst, **kwargs):
    from dataclasses import replace
    return replace(inst, **kwargs)
