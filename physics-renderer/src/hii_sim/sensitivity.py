"""Local design diagnostics, not an estimator or a recoverability guarantee.

For the specific real-image FFT noise generator, conjugates are identical
information. Whiten one representative per pair, with channel covariance and
real-only self modes. Marginalize linearized shared amplitude/slope errors.
"""
from dataclasses import replace
import numpy as np
from scipy.linalg import solve_triangular
from .instrument import expected_measurement, independent_fourier_cells, noise_sigma_real, operator
from .physics import render
from .schema import Source

PARAMETERS = ("ln_EM0", "ln_Te", "ln_major_size", "axis_ratio", "PA_radians")


def whiten_columns(columns, weights, sigma_real, rho):
    """[frequency,y,x,column] complex values -> independent whitened real rows.

    Missing channels are marginalized by selecting their covariance submatrix.
    Cov(Re V_f,Re V_g)=sigma_f sigma_g [rho+(1-rho) delta_fg].
    Same for imaginary parts; self-conjugate modes are real with 2x variance.
    This is specific to observe(), not a generic interferometer noise model.
    """
    columns = np.asarray(columns)
    if columns.ndim == 3:
        columns = columns[..., None]
    if columns.shape[:3] != weights.shape or np.any(np.asarray(sigma_real) <= 0):
        raise ValueError("Shape mismatch or nonpositive thermal sigma")
    nf, n, _, nc = columns.shape
    if nf > 62:
        raise ValueError("Whitening grouping supports at most 62 frequency planes")
    unique, self_cells = independent_fourier_cells(n)
    flags = weights > 0
    codes = np.sum(flags.astype(np.int64)*(1 << np.arange(nf))[:, None, None], axis=0)
    rows = []
    for code in np.unique(codes[unique]):
        if code == 0:
            continue
        channels = np.flatnonzero(code & (1 << np.arange(nf)))
        sig = np.asarray(sigma_real)[channels]
        covariance = np.outer(sig, sig)*(rho+(1-rho)*np.eye(len(channels)))
        lower = np.linalg.cholesky(covariance)
        for self_only in (False, True):
            select = unique & (codes == code) & (self_cells == self_only)
            if not select.any():
                continue
            values = columns[channels][:, select, :]
            real = solve_triangular(lower, values.real.reshape(len(channels), -1), lower=True)
            rows.append((real/np.sqrt(2) if self_only else real).reshape(-1, nc))
            if not self_only:
                imag = solve_triangular(lower, values.imag.reshape(len(channels), -1), lower=True)
                rows.append(imag.reshape(-1, nc))
    return np.concatenate(rows) if rows else np.empty((0, nc))


def scaled_measurement_jacobian(source, grid, sky, inst, step=1e-3):
    """Derivative w.r.t. dimensionless coordinates declared in PARAMETERS."""
    def measurement(s):
        return expected_measurement(render(s, grid, sky, inst.frequencies_ghz,
            inst.bandwidths_mhz, inst.frequency_quadrature), grid, inst)
    base = measurement(source)
    derivatives = []
    stencils = []
    for name in ("em0_pc_cm6", "te_k", "major_fwhm_arcsec", "axis_ratio", "pa_deg"):
        value = getattr(source, name)
        if name in ("em0_pc_cm6", "te_k", "major_fwhm_arcsec"):
            plus, minus = value*np.exp(step), value*np.exp(-step)
        else:
            delta = np.rad2deg(step) if name == "pa_deg" else step
            plus, minus = value+delta, value-delta
        if name == "axis_ratio" and plus > 1:
            deriv = (base-measurement(replace(source, **{name: minus})))/step
            stencils.append("backward_at_q_boundary")
        elif name == "te_k" and (plus > 12000 or minus < 5000):
            raise ValueError("Temperature sensitivity requires an interior point")
        else:
            deriv = (measurement(replace(source, **{name: plus}))-
                     measurement(replace(source, **{name: minus})))/(2*step)
            stencils.append("central")
        derivatives.append(deriv)
    return base, np.stack(derivatives, axis=-1), stencils


def local_diagnostic(source, grid, sky, inst, step=1e-3):
    base, derivatives, stencils = scaled_measurement_jacobian(source, grid, sky, inst, step)
    weights, _ = operator(grid, inst)
    logfreq = np.log(inst.frequencies_ghz); logfreq -= logfreq.mean()
    # First-order Gaussian nuisance approximation. sigma_amp matches the exact
    # lognormal fractional standard deviation; slope linearized about zero.
    nuisance = np.stack((base*inst.calibration_fraction,
                         base*logfreq[:, None, None]*inst.calibration_slope_std), axis=-1)
    white = whiten_columns(np.concatenate((derivatives, nuisance), axis=-1), weights,
                           noise_sigma_real(grid, inst), inst.noise_frequency_correlation)
    jac, cal = white[:, :5], white[:, 5:]
    thermal = jac.T@jac
    cross = jac.T@cal
    marginal = thermal-cross@np.linalg.solve(np.eye(2)+cal.T@cal, cross.T)
    marginal = (marginal+marginal.T)/2
    diag = np.sqrt(np.maximum(np.diag(marginal), 0))
    denominator = np.outer(diag, diag)
    correlation = np.divide(marginal, denominator, out=np.zeros_like(marginal), where=denominator > 0)
    eigenvalues, eigenvectors = np.linalg.eigh(marginal)
    return {"dimensionless_coordinates": list(PARAMETERS), "finite_difference_step": step,
            "stencils": stencils, "independent_real_data_count": len(white),
            "thermal_information": thermal.tolist(), "calibration_marginal_information": marginal.tolist(),
            "scaled_response_snr_for_delta_0p01": (0.01*diag).tolist(),
            "scaled_information_column_correlations": correlation.tolist(),
            "scaled_information_eigenvalues": eigenvalues.tolist(),
            "weakest_scaled_direction": eigenvectors[:, 0].tolist(),
            "calibration_approximation": "linearized shared amplitude/spectral-slope Gaussian nuisances; no source prior",
            "interpretation": "local design response only, not posterior errors, global identifiability or recoverability"}


def representative_sensitivity(grid, sky, inst, profile_subpixels=None, profile_spatial_rules=None):
    cases = {"thin_faint_compact": Source(1e4, 8000, 12, .7, 37),
             "thin_resolved": Source(1.2e5, 8000, 32, .72, 37),
             "bright_resolved": Source(5e5, 6500, 60, .5, 73),
             "turnover_compact": Source(1e7, 8000, 16, .7, 20),
             "circular": Source(1.2e5, 8000, 32, 1., 37),
             "ellipsoid_challenge": Source(5e5, 10000, 24, .5, 115, "ellipsoid_em")}
    from dataclasses import asdict
    reports = {}
    for name, source in cases.items():
        case_grid = replace(grid, subpixels=(profile_subpixels or {}).get(source.profile, grid.subpixels),
                            spatial_rule=(profile_spatial_rules or {}).get(source.profile,grid.spatial_rule))
        report = local_diagnostic(source, case_grid, sky, inst)
        finer = local_diagnostic(source, case_grid, sky, inst, step=5e-4)
        a = np.asarray(report["calibration_marginal_information"])
        b = np.asarray(finer["calibration_marginal_information"])
        report["information_relative_l2_step_halving"] = float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300))
        report["validation_truth"] = asdict(source)
        report["render_grid"] = asdict(case_grid)
        reports[name] = report
    return reports
