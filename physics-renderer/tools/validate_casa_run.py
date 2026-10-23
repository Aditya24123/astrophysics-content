"""Quantify genuine CASA fixture products; never generates or fabricates an MS.

Expected layout: ``--fixtures`` is output/audit_casa_inputs_v2 and
``--runs`` contains one subdirectory per case, each produced by
``hii_sim.casa_simulate``.  This is a post-run validator, not a CASA substitute.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from scipy.signal import fftconvolve

from hii_sim.instrument import expected_measurement, operator
from hii_sim.casa_simulate import clean_stop_diagnostics
from hii_sim.io import code_digest, source_manifest, versions
from hii_sim.reconstruction import dirty_image, fit_clean_beam, point_spread_function
from hii_sim.schema import Grid, Instrument, load_config


TOLERANCES = {
    "point_peak_flux_relative_error_max": 0.10,
    "point_centroid_error_pixels_max": 0.5,
    "psf_vs_image_header_beam_relative_error_max": 0.10,
    "niter_zero_image_residual_relative_l2_max": 1e-3,
    "product_grid_alignment_max_pixels": 0.05,
    "psf_peak_absolute_error_max": 1e-3,
}

# This is the narrow prerequisite suite for declaring CASA fixture gates passed.
# These cases are produced by tools/prepare_casa_validation.py.
REQUIRED_CASES = {
    "point_noiseless": {"fwhm_arcsec": 0.0, "thermalnoise": ""},
    "point_noisy": {"fwhm_arcsec": 0.0, "thermalnoise": "tsys-atm"},
    "extended45_noiseless": {"fwhm_arcsec": 45.0, "thermalnoise": ""},
    "extended45_noisy": {"fwhm_arcsec": 45.0, "thermalnoise": "tsys-atm"},
    "extended90_noiseless": {"fwhm_arcsec": 90.0, "thermalnoise": ""},
    "extended90_noisy": {"fwhm_arcsec": 90.0, "thermalnoise": "tsys-atm"},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_primary(path):
    with fits.open(path, checksum=True, memmap=False) as hdus:
        data = np.asarray(hdus[0].data, dtype=float).squeeze()
        header = hdus[0].header.copy()
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError(f"Expected a finite 2-D exported CASA image: {path}")
    return data, header


def relative_l2(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) /
                 max(np.linalg.norm(b), 1e-30))


def restoring_beam_kernel(shape, header):
    """Unit-peak restoring Gaussian using CASA BMAJ/BMIN/BPA and local WCS."""
    try:
        major_deg = float(header["BMAJ"])
        minor_deg = float(header["BMIN"])
        pa_deg = float(header["BPA"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("CLEAN restored image requires finite BMAJ, BMIN and BPA") from exc
    if not np.isfinite([major_deg, minor_deg, pa_deg]).all() or major_deg <= 0 or minor_deg <= 0 or major_deg < minor_deg:
        raise ValueError("CLEAN restoring beam requires finite positive ordered BMAJ >= BMIN and finite BPA")
    wcs = WCS(header).celestial
    lon, lat = int(wcs.wcs.lng), int(wcs.wcs.lat)
    matrix = np.asarray(wcs.pixel_scale_matrix, dtype=float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("CLEAN restoring beam has invalid celestial pixel geometry")
    # For celestial TAN/SIN headers this matrix maps pixels to the local
    # tangent-plane longitude/latitude axes in angular units. Applying an
    # extra cos(dec) would shrink the east axis a second time.
    east_dxdy = matrix[lon] * 3600.
    north_dxdy = matrix[lat] * 3600.
    cy, cx = np.array(shape, dtype=float)//2
    yy, xx = np.indices(shape, dtype=float)
    dx, dy = xx-cx, yy-cy
    east = east_dxdy[0]*dx + east_dxdy[1]*dy
    north = north_dxdy[0]*dx + north_dxdy[1]*dy
    pa = np.deg2rad(pa_deg)
    major_coord = east*np.sin(pa) + north*np.cos(pa)
    minor_coord = east*np.cos(pa) - north*np.sin(pa)
    major_arcsec, minor_arcsec = major_deg*3600., minor_deg*3600.
    beam = np.exp(-4*np.log(2)*((major_coord/major_arcsec)**2 +
                               (minor_coord/minor_arcsec)**2))
    return beam, {"major_fwhm_arcsec": major_arcsec,
                  "minor_fwhm_arcsec": minor_arcsec, "pa_deg_north_through_east": pa_deg,
                  "area_pixels": float(beam.sum()),
                  "geometry": "unit-peak elliptical Gaussian from header beam and local celestial WCS Jacobian"}


def centered_convolution(image, kernel):
    full = fftconvolve(image, kernel, mode="full")
    cy, cx = np.array(kernel.shape)//2
    return full[cy:cy+image.shape[0], cx:cx+image.shape[1]]


def beam_fit_difference(declared, fitted):
    """Descriptive comparison of CASA restoration header to PSF core fit."""
    dpa = float(declared["pa_deg_north_through_east"])
    fpa = float(fitted["pa_deg"])
    angle = abs((dpa-fpa+90.) % 180. - 90.)
    return {"major_relative_difference": float(declared["major_fwhm_arcsec"] /
                                                 fitted["major_fwhm_arcsec"]-1.),
            "minor_relative_difference": float(declared["minor_fwhm_arcsec"] /
                                                 fitted["minor_fwhm_arcsec"]-1.),
            "position_angle_difference_deg_modulo_180": angle,
            "interpretation": "descriptive; CASA restoring-beam metadata controls restoration reference"}


def json_safe(value):
    """Convert NumPy scalars and reject non-finite values before report writes."""
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Non-finite value in CASA validation report")
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(f"Unsupported CASA report value: {type(value).__name__}")


def strict_json_dumps(value):
    return json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n"


def _normalized_bunit(value):
    return str(value or "").strip().lower().replace(" ", "")


def effective_product_unit(role, raw_bunit, *, stage, imaging, casa_version,
                           producer, actual_casa, dirty_residual_matches_image=False):
    """Interpret CASA output roles without changing or relabeling native FITS."""
    raw = str(raw_bunit or "").strip()
    normalized = _normalized_bunit(raw)
    if stage not in {"dirty", "clean"} or not isinstance(imaging, dict):
        raise ValueError("Unsupported CASA image producer stage or missing tclean settings")
    niter = imaging.get("niter")
    if stage == "dirty" and niter != 0:
        raise ValueError("Dirty products must come from the declared niter=0 tclean producer")
    if stage == "clean" and (not isinstance(niter, int) or niter <= 0):
        raise ValueError("CLEAN products must come from a positive-iteration tclean producer")
    if producer != "casatasks.tclean" and not str(producer).startswith("test-double:"):
        raise ValueError("Unsupported CASA image producer for role-based unit interpretation")

    def require_real_blank_evidence():
        if (not actual_casa or producer != "casatasks.tclean" or
                str(casa_version) != "6.7.4.8"):
            raise ValueError("Blank FITS BUNIT requires verified CASA 6.7.4.8 tclean role evidence")

    if role in {"image"}:
        if normalized not in {"jy/beam", "jybeam-1", "jybeam^-1"}:
            raise ValueError(f"CASA {stage} restored/dirty image requires explicit Jy/beam BUNIT; found {raw!r}")
        effective = ("Jy/dirty beam" if stage == "dirty" else
                     "native composite: model in Jy/clean beam plus residual in Jy/dirty beam")
        basis = "explicit native Jy/beam plus tclean image role"
    elif role == "model" and stage == "clean":
        if normalized not in {"jy/pixel", "jy/pix", "jy/pixel-1"}:
            raise ValueError(f"CLEAN component model requires explicit Jy/pixel BUNIT; found {raw!r}")
        effective, basis = "Jy/pixel", "explicit native BUNIT and tclean component-model role"
    elif role == "residual":
        if normalized and normalized not in {"jy/beam", "jybeam-1", "jybeam^-1"}:
            raise ValueError(f"CASA {stage} residual BUNIT conflicts with the tclean residual role: {raw!r}")
        effective = "Jy/dirty beam"
        if normalized:
            basis = "native Jy/beam plus tclean residual role; no beam conversion"
        else:
            require_real_blank_evidence()
            if stage == "dirty" and not dirty_residual_matches_image:
                raise ValueError("Blank dirty residual requires exact pixel equality to its explicit-unit niter=0 dirty image")
            basis = ("verified CASA 6.7.4.8 tclean niter=0 residual role and exact equality to the explicit Jy/beam dirty image"
                     if stage == "dirty" else
                     "verified CASA 6.7.4.8 tclean CLEAN residual role; official CASA semantics specify dirty-beam residual units")
    elif role in {"psf", "pb"}:
        if normalized and normalized not in {"1", "dimensionless"}:
            raise ValueError(f"CASA {stage} {role.upper()} must be dimensionless; found conflicting BUNIT {raw!r}")
        if not normalized:
            require_real_blank_evidence()
            basis = ("verified CASA tclean PSF role; unit-peak response validated numerically"
                     if role == "psf" else
                     "verified CASA tclean normalized primary-beam role; dimensionless response validated numerically")
        else:
            basis = "explicit dimensionless BUNIT and tclean product role"
        effective = "dimensionless normalized PSF response" if role == "psf" else "dimensionless primary-beam response"
    else:
        raise ValueError(f"Unsupported CASA product role/stage: {stage}.{role}")
    return {"raw_fits_bunit": raw, "effective_unit": effective,
            "interpretation_basis": basis, "producer": producer,
            "producer_stage": stage, "casa_version": str(casa_version)}


def _path_text(value):
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError("Measurement Set path must be a nonempty plain filesystem path")
    return value


def _absolute_ms_identity(value):
    """Lexical absolute-path identity across hosts; never resolves remote MS."""
    value = _path_text(value)
    if value.startswith(("\\\\?\\", "\\\\.\\")):
        raise ValueError("Unsupported device path for Measurement Set")
    windows = PureWindowsPath(value)
    if windows.drive:
        if not windows.is_absolute():
            raise ValueError("Measurement Set path must be absolute, not drive-relative")
        path, style = windows, "windows"
    else:
        if "\\" in value:
            raise ValueError("Malformed mixed or rooted-relative Measurement Set path")
        path, style = PurePosixPath(value), "posix"
        if not path.is_absolute():
            raise ValueError("Measurement Set path must be absolute")
    if len(path.parts) < 2 or ".." in path.parts:
        raise ValueError("Measurement Set path must name a directory without parent traversal")
    # Pure paths normalize separators, redundant '.' and trailing separators.
    # Keep case, drive/share and host identity exact; do not map symlink aliases.
    return style, path.as_posix()


def _relative_ms_path(value):
    value = _path_text(value).replace("\\", "/")
    windows, posix = PureWindowsPath(value), PurePosixPath(value)
    if windows.drive or posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ValueError("Legacy Measurement Set product path must be relative without traversal")
    return value


def verified_runtime_context(runtime_provenance, products, products_path, case_name, channel):
    """Bind blank-unit interpretations to independently hashed CASA run evidence."""
    saved_ms_origin = "measurement_only_reconstruction_of_existing_ms"
    origin = products.get("product_origin")
    if "product_origin" in products and origin != saved_ms_origin:
        raise ValueError("Unsupported CASA product_origin")
    package_info = products.get("casa_packages", {})
    version = (package_info.get("casatasks") or products.get("casa_version") or "unknown")
    if version.startswith("unknown"):
        version = "unknown"
    if runtime_provenance is None:
        if origin == saved_ms_origin:
            raise ValueError("Saved-MS reconstruction requires independently checksum-bound runtime provenance")
        return {"actual_casa": False, "casa_version": version,
                "producer": products.get("imaging_producer", ""),
                "evidence": None}
    if runtime_provenance.get("actual_CASA") is not True or runtime_provenance.get("test_double") is not False:
        raise ValueError("Runtime provenance does not identify a genuine CASA run")
    versions = runtime_provenance.get("versions", {})
    if versions.get("casatasks") != "6.7.4.8" or versions.get("casatools") != "6.7.4.8":
        raise ValueError("Blank-unit compatibility is only audited for CASA 6.7.4.8 task/tool packages")
    if products.get("runner_sha256") != runtime_provenance.get("runner_sha256"):
        raise ValueError("Runtime provenance runner hash does not match the CASA product report")
    audited_historical_runner = "00533bd4ddeef2119fde3357d4f10658715546e838ec6783c0f5f48f97657e06"
    declared_producer = products.get("imaging_producer")
    if declared_producer is not None and declared_producer != "casatasks.tclean":
        raise ValueError("Native CASA report declares an unsupported image producer")
    if (declared_producer is None and products.get("runner_sha256") != audited_historical_runner):
        raise ValueError("No supported tclean producer evidence for this historical runner")
    if version not in {"unknown", versions["casatasks"]}:
        raise ValueError("Native and external CASA versions conflict")
    for package in ("casatasks", "casatools"):
        declared = package_info.get(package)
        if declared not in {None, "unknown", versions[package]}:
            raise ValueError("Native and external CASA package versions conflict")
    case_record = runtime_provenance.get("cases", {}).get(case_name)
    if not isinstance(case_record, dict) or case_record.get("status") != "completed_unvalidated":
        raise ValueError("Runtime provenance lacks the completed case record")
    if case_record.get("report_sha256") != sha256(products_path):
        raise ValueError("Runtime provenance does not checksum-anchor this CASA product report")
    selected = case_record.get("selected_ms", "")
    if origin == saved_ms_origin:
        if declared_producer != "casatasks.tclean":
            raise ValueError("Saved-MS reconstruction requires explicit casatasks.tclean producer evidence")
        identities = [_absolute_ms_identity(value) for value in (
            products.get("input_measurement_set"), channel.get("measurement_set"), selected)]
        if len(set(identities)) != 1:
            raise ValueError("Saved-MS absolute Measurement Set paths disagree across report, channel and runtime evidence")
        path_contract = "exact normalized absolute agreement across input_measurement_set, channel.measurement_set and selected_ms"
    else:
        _absolute_ms_identity(selected)
        relative_ms = _relative_ms_path(channel.get("measurement_set"))
        if not selected.replace("\\", "/").endswith("/" + case_name + "/" + relative_ms):
            raise ValueError("Runtime provenance selected Measurement Set does not match the case product report")
        path_contract = "legacy relative product path with case-directory suffix guard"
    return {"actual_casa": True, "casa_version": "6.7.4.8",
            "producer": "casatasks.tclean",
            "evidence": {"runtime_provenance_sha256": sha256(runtime_provenance["_path"]),
                         "case_report_sha256": case_record["report_sha256"],
                         "runner_sha256": runtime_provenance["runner_sha256"],
                         "measurement_set_path_contract": path_contract,
                         "package_versions": {"casatasks": versions["casatasks"],
                                              "casatools": versions["casatools"]}}}


def validate_case_manifest(case_data):
    """Require the full CASA suite unless a nonempty suite is marked test-only."""
    if not isinstance(case_data, dict) or not isinstance(case_data.get("cases"), list):
        raise ValueError("cases.json must contain a 'cases' list")
    cases = case_data["cases"]
    if not cases:
        raise ValueError("CASA validation suite is empty; required cases were not run")
    test_double_only = case_data.get("test_double_only") is True
    names = []
    for row in cases:
        if not isinstance(row, dict) or not isinstance(row.get("bundle"), str):
            raise ValueError("Every CASA validation case needs a bundle name")
        name = row["bundle"]
        if name not in REQUIRED_CASES:
            raise ValueError(f"Unknown CASA validation case: {name}")
        if name in names:
            raise ValueError(f"Duplicate CASA validation case: {name}")
        names.append(name)
        expected = REQUIRED_CASES[name]
        if (row.get("fwhm_arcsec") != expected["fwhm_arcsec"] or
                row.get("thermalnoise") != expected["thermalnoise"] or
                not np.isclose(float(row.get("input_flux_jy", np.nan)), 0.010, rtol=1e-6, atol=0.0)):
            raise ValueError(f"Case {name} does not match the declared 10-mJy fixture definition")
        if test_double_only and row.get("test_double_only") is not True:
            raise ValueError("A narrowed test-double suite must mark every case test_double_only")
        if not test_double_only and row.get("test_double_only"):
            raise ValueError("Test-double cases cannot be presented as the required CASA suite")
    missing = sorted(set(REQUIRED_CASES) - set(names))
    if missing and not test_double_only:
        raise ValueError("Missing required CASA validation cases: " + ", ".join(missing))
    return {"required_cases": list(REQUIRED_CASES), "provided_cases": names,
            "missing_required_cases": missing,
            "complete_required_suite": not missing,
            "test_double_only": test_double_only}


def grid_alignment_error_pixels(reference_header, candidate_header, shape):
    """Maximum sampled celestial WCS displacement on a shared pixel grid."""
    reference = WCS(reference_header).celestial
    candidate = WCS(candidate_header).celestial
    if reference.pixel_n_dim != 2 or candidate.pixel_n_dim != 2:
        raise ValueError("A valid two-dimensional celestial WCS is required")
    ny, nx = shape
    x = np.linspace(0., nx-1., 5)
    y = np.linspace(0., ny-1., 5)
    xx, yy = np.meshgrid(x, y)
    probes = np.column_stack((xx.ravel(), yy.ravel()))
    ref_world = reference.pixel_to_world(probes[:, 0], probes[:, 1])
    candidate_world = candidate.pixel_to_world(probes[:, 0], probes[:, 1])
    separation_arcsec = np.asarray(ref_world.separation(candidate_world).arcsec, dtype=float)
    pixel_scale_arcsec = float(np.sqrt(abs(np.linalg.det(reference.pixel_scale_matrix))) * 3600)
    if (not np.isfinite(separation_arcsec).all() or not np.isfinite(pixel_scale_arcsec)
            or pixel_scale_arcsec <= 0):
        raise ValueError("Invalid celestial WCS coordinates or pixel scale")
    return float(np.max(separation_arcsec) / pixel_scale_arcsec)


def pixel_position_error(expected_header, expected_xy, measured_header, measured_xy):
    """Frame-aware angular separation expressed in expected-grid pixels."""
    expected_wcs = WCS(expected_header).celestial
    measured_wcs = WCS(measured_header).celestial
    expected = expected_wcs.pixel_to_world(float(expected_xy[0]), float(expected_xy[1]))
    measured = measured_wcs.pixel_to_world(float(measured_xy[0]), float(measured_xy[1]))
    scale = float(np.sqrt(abs(np.linalg.det(expected_wcs.pixel_scale_matrix))) * 3600)
    error = float(expected.separation(measured).arcsec / scale)
    if not np.isfinite(error) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid source-position WCS comparison")
    return error


def positive_core_moments(image, source_x, source_y, radius_pixels):
    """Windowed positive dirty-image moments; diagnostic, not a flux estimator."""
    image = np.asarray(image, dtype=float)
    yy, xx = np.indices(image.shape)
    radius = np.hypot(xx-source_x, yy-source_y)
    window = radius <= radius_pixels
    outside = ~window
    background = float(np.median(image[outside])) if outside.any() else 0.
    signal = np.where(window, np.maximum(image-background, 0.), 0.)
    total = float(signal.sum())
    if total <= 0:
        raise ValueError("No positive dirty-image core for morphology diagnostic")
    xbar = float((signal*xx).sum()/total)
    ybar = float((signal*yy).sum()/total)
    dx, dy = xx-xbar, yy-ybar
    covariance = np.array([[(signal*dx*dx).sum(), (signal*dx*dy).sum()],
                           [(signal*dx*dy).sum(), (signal*dy*dy).sum()]])/total
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.min(eigenvalues) <= 0:
        raise ValueError("Dirty-image moment covariance is not positive definite")
    fwhm = np.sqrt(8*np.log(2)*eigenvalues[::-1])
    return {"major_fwhm_pixels": float(fwhm[0]), "minor_fwhm_pixels": float(fwhm[1]),
            "centroid_offset_pixels": float(np.hypot(xbar-source_x, ybar-source_y)),
            "window_radius_pixels": float(radius_pixels),
            "background_median_jy_beam": background,
            "definition": "positive, median-background-subtracted second moments in declared circular window; dirty-beam sidelobes and missing spacings remain"}


def validate_case(case, fixture_root, run_dir, fast_config, runtime_provenance=None):
    run_dir = Path(run_dir)
    products_path = run_dir / "casa_products.json"
    products = json.loads(products_path.read_text())
    if products.get("status") != "completed_unvalidated":
        raise RuntimeError(f"CASA run is not task-complete: {products.get('status')}")
    if bool(case.get("test_double_only", False)) != bool(products.get("test_double", False)):
        raise ValueError("Case and product test-double markers must agree")
    if len(products.get("products", [])) != 1:
        raise RuntimeError("Fixture requires exactly one reported frequency plane")
    channel = products["products"][0]
    if products.get("input_manifest_sha256") and products["input_manifest_sha256"] != sha256(
            fixture_root / case["bundle"] / "casa_inputs.json"):
        raise ValueError("CASA product report does not match this fixture input manifest")
    dirty_products = channel.get("dirty_products", channel.get("products", {}))
    clean_products = channel.get("clean_products")
    if not isinstance(clean_products, dict) or set(clean_products) != {"image", "model", "residual", "psf", "pb"}:
        raise RuntimeError("CASA run lacks complete separate measurement-only CLEAN products")
    files = {}
    for role in ("image", "psf", "pb", "residual"):
        entry = dirty_products[role]
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("CASA product path escapes its run directory")
        path = run_dir / relative
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Missing or changed CASA product: {path}")
        files[role] = path
    arrays = {role: read_primary(path) for role, path in files.items()}
    clean_files = {}
    for role in ("image", "model", "residual", "psf", "pb"):
        entry = clean_products[role]
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("CASA CLEAN product path escapes its run directory")
        path = run_dir / relative
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Missing or changed CASA CLEAN product: {path}")
        clean_files[role] = path
    clean_arrays = {role: read_primary(path) for role, path in clean_files.items()}
    image, image_header = arrays["image"]
    residual, residual_header = arrays["residual"]
    psf, psf_header = arrays["psf"]
    pb, pb_header = arrays["pb"]
    if any(data.shape != image.shape for data, _ in arrays.values()):
        raise ValueError("CASA image, residual, PSF and PB shapes differ")
    if any(data.shape != image.shape for data, _ in clean_arrays.values()):
        raise ValueError("CASA CLEAN image, model, residual, PSF and PB shapes differ")
    clean_image, clean_image_header = clean_arrays["image"]
    clean_model, clean_model_header = clean_arrays["model"]
    clean_residual, clean_residual_header = clean_arrays["residual"]
    clean_psf, clean_psf_header = clean_arrays["psf"]
    clean_pb, clean_pb_header = clean_arrays["pb"]
    runtime = verified_runtime_context(runtime_provenance, products, products_path,
                                       case["bundle"], channel)
    if products.get("test_double"):
        runtime.update(actual_casa=False, producer="test-double:synthetic FITS")
    product_units = {}
    for stage, group in (("dirty", arrays), ("clean", clean_arrays)):
        settings = channel.get(stage+"_imaging")
        if settings is None and stage == "dirty" and products.get("test_double"):
            settings = {"niter": 0}
        for role, (_, header) in group.items():
            product_units[stage+"."+role] = effective_product_unit(
                role, header.get("BUNIT", ""), stage=stage, imaging=settings,
                casa_version=runtime["casa_version"], producer=runtime["producer"],
                actual_casa=runtime["actual_casa"],
                dirty_residual_matches_image=bool(np.array_equal(image, residual)))
    wcs = WCS(image_header).celestial
    pixel_arcsec = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600)
    if not np.isfinite(pixel_arcsec) or pixel_arcsec <= 0:
        raise ValueError("Image has an invalid celestial pixel scale")
    # Every measurement product is compared to the same dirty-image WCS.
    # Using clean_image_header as its group's reference would let the complete
    # CLEAN set drift together while still passing registration checks.
    alignment_errors = {"dirty_"+role: grid_alignment_error_pixels(image_header, header, image.shape)
                        for role, (_, header) in arrays.items()}
    alignment_errors.update({"clean_"+role: grid_alignment_error_pixels(image_header, header, image.shape)
                             for role, (_, header) in clean_arrays.items()})
    sky_path = fixture_root / case["bundle"] / "ch00.model.fits"
    sky, sky_header = read_primary(sky_path)
    if sky.shape != image.shape:
        raise ValueError("Input sky and CASA image shapes differ")
    if str(sky_header.get("BUNIT", "")).strip().lower().replace(" ", "") not in {"jy/pixel", "jy/pix"}:
        raise ValueError("Prepared apparent sky reference must be in Jy/pixel")
    if str(clean_pb_header.get("BUNIT", "")).strip().lower() not in {"1", "", "dimensionless"}:
        raise ValueError("CLEAN primary-beam image must be dimensionless")
    alignment_errors["input_sky"] = grid_alignment_error_pixels(image_header, sky_header, image.shape)
    alignment_tolerance = TOLERANCES["product_grid_alignment_max_pixels"]
    if any(error > alignment_tolerance for error in alignment_errors.values()):
        raise ValueError(f"Celestial WCS grids are misaligned by up to {max(alignment_errors.values()):.6g} pixels")

    center = tuple(np.array(psf.shape) // 2)
    psf_peak = float(psf[center])
    if abs(psf_peak-1.) > TOLERANCES["psf_peak_absolute_error_max"]:
        raise ValueError(f"Invalid PSF normalization: center={psf_peak!r}, expected unit peak")
    _, psf_fit = fit_clean_beam(psf/psf_peak, pixel_arcsec)
    clean_psf_center = tuple(np.array(clean_psf.shape)//2)
    clean_psf_peak = float(clean_psf[clean_psf_center])
    if abs(clean_psf_peak-1.) > TOLERANCES["psf_peak_absolute_error_max"]:
        raise ValueError("CLEAN PSF does not have a unit central peak")
    for stage, beam in (("dirty", pb), ("clean", clean_pb)):
        if np.min(beam) < 0 or np.max(beam) > 1.001 or not 0 < beam[center] <= 1.001:
            raise ValueError(f"{stage} primary-beam response violates normalized dimensionless response bounds")
    clean_psf_fit, clean_psf_fit_metadata = fit_clean_beam(clean_psf/clean_psf_peak, pixel_arcsec)
    clean_beam, clean_beam_metadata = restoring_beam_kernel(clean_image.shape, clean_image_header)
    clean_model_restored = centered_convolution(clean_model, clean_beam)
    apparent_sky_restored = centered_convolution(sky*clean_pb, clean_beam)
    matched_beam_model_error = clean_model_restored-apparent_sky_restored
    clean_closure = relative_l2(clean_image, clean_model_restored+clean_residual)
    clean_pixel_tolerance = 0.02
    clean_beam_psf_difference = beam_fit_difference(clean_beam_metadata, clean_psf_fit_metadata)
    beam_values = (image_header.get("BMAJ"), image_header.get("BMIN"))
    try:
        bmaj_deg, bmin_deg = (float(x) for x in beam_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("CASA image header must contain finite BMAJ and BMIN") from exc
    if (not np.isfinite([bmaj_deg, bmin_deg]).all() or bmaj_deg <= 0 or bmin_deg <= 0
            or bmaj_deg < bmin_deg):
        raise ValueError("CASA image BMAJ/BMIN must be finite, positive, and ordered major >= minor")
    beam_errors = {
        "major": abs(psf_fit["major_fwhm_arcsec"]/(bmaj_deg*3600)-1),
        "minor": abs(psf_fit["minor_fwhm_arcsec"]/(bmin_deg*3600)-1),
    }
    beam_error = max(beam_errors.values())

    metrics = {
        "case": case["bundle"], "casa_version": products.get("casa_version"),
        "verified_casa_runtime": runtime,
        "product_units": product_units,
        "test_double_only": bool(products.get("test_double", False)),
        "measurement_set": channel["measurement_set"],
        "discovered_measurement_sets": channel["discovered_measurement_sets"],
        "image_bunit": image_header.get("BUNIT"), "pixel_arcsec": pixel_arcsec,
        "product_grid_alignment_error_pixels": alignment_errors,
        "image_peak_jy_beam": float(np.max(image)),
        "image_min_jy_beam": float(np.min(image)),
        "residual_peak_abs_jy_beam": float(np.max(np.abs(residual))),
        "residual_rms_jy_beam": float(np.sqrt(np.mean(residual**2))),
        "niter_zero_image_residual_relative_l2": relative_l2(image, residual),
        "psf_center_value": psf_peak,
        "psf_fit": psf_fit,
        "image_header_beam_arcsec": {"major": bmaj_deg*3600, "minor": bmin_deg*3600},
        "psf_vs_header_beam_relative_error": beam_errors,
        "pb_center": float(pb[tuple(np.array(pb.shape)//2)]),
        "point_peak_flux_relative_error": None,
        "point_centroid_error_pixels": None,
        "extended_positive_core_moments": None,
        "header_beam_vs_psf_fit_relative_error": None,
        "fast_proxy_dirty_image_relative_l2": None,
        "fast_proxy_psf_relative_l2": None,
        "clean_restored_peak_jy_beam": float(np.max(clean_image)),
        "clean_model_component_flux_jy": float(np.sum(clean_model)),
        "clean_residual_peak_abs_jy_beam": float(np.max(np.abs(clean_residual))),
        "clean_residual_rms_jy_beam": float(np.sqrt(np.mean(clean_residual**2))),
        "clean_threshold_jy": channel.get("clean_imaging", {}).get("threshold"),
        "clean_threshold_met_by_final_residual": channel.get("clean_threshold_met_by_final_residual"),
        "clean_restoration_closure_relative_l2": clean_closure,
        "clean_restoration_closure_interpretation": "native numerical tclean pixel closure only: model term Jy/clean beam plus residual term Jy/dirty beam; no residual scaling or same-beam flux interpretation",
        "unit_normalization_evidence": {"dirty_image_equals_residual_exactly": bool(np.array_equal(image, residual)),
                                        "dirty_psf_center": psf_peak, "clean_psf_center": clean_psf_peak,
                                        "clean_restoration_closure_relative_l2": clean_closure},
        "clean_restoring_beam": clean_beam_metadata,
        "clean_psf_core_fit": clean_psf_fit_metadata,
        "clean_restoring_beam_vs_psf_fit": clean_beam_psf_difference,
        "matched_beam_apparent_sky_vs_reconstructed_model_relative_l2": relative_l2(
            clean_model_restored, apparent_sky_restored),
        "matched_beam_apparent_sky_vs_reconstructed_model_rmse_jy_beam": float(
            np.sqrt(np.mean(matched_beam_model_error**2))),
        "matched_beam_apparent_sky_peak_jy_beam": float(np.max(apparent_sky_restored)),
        "matched_beam_reconstructed_model_peak_jy_beam": float(np.max(clean_model_restored)),
        "matched_beam_comparison_usage": "validation only: (input sky Jy/pixel x measured CLEAN PB) and CLEAN model convolved by the declared CASA restoring beam; never used for imaging controls",
        "measurement_set_audit": channel.get("measurement_set_audit"),
        "extended_source_flux": "model component sum and residual are separate unit-bearing measurements; no missing-spacing correction is implied",
    }
    source_y, source_x = np.unravel_index(np.argmax(sky), sky.shape)
    clean_restored_moments = positive_core_moments(
        clean_image, source_x, source_y,
        min(max(4*float(case["fwhm_arcsec"])/pixel_arcsec,
                4*clean_beam_metadata["major_fwhm_arcsec"]/pixel_arcsec),
            min(image.shape)/2-1))
    clean_restored_moments["major_fwhm_arcsec"] = clean_restored_moments["major_fwhm_pixels"]*pixel_arcsec
    clean_restored_moments["minor_fwhm_arcsec"] = clean_restored_moments["minor_fwhm_pixels"]*pixel_arcsec
    metrics["clean_restored_positive_core_moments"] = clean_restored_moments
    if float(case["fwhm_arcsec"]) > 0:
        yy, xx = np.indices(image.shape)
        aperture_radius = min(4*float(case["fwhm_arcsec"])/pixel_arcsec, min(image.shape)/2-1)
        aperture = np.hypot(xx-source_x, yy-source_y) <= aperture_radius
        psf_radius = max(1., 2*psf_fit["major_fwhm_arcsec"]/pixel_arcsec)
        psf_area = float(psf[np.hypot(xx-psf.shape[1]//2, yy-psf.shape[0]//2) <= psf_radius].sum())
        residual_sum = float(clean_residual[aperture].sum())
        residual_flux = residual_sum/psf_area if psf_area > 1e-10 else None
        component_aperture = float(clean_model[aperture].sum())
        metrics["extended_clean_flux_bookkeeping"] = {
            "usage": "validation_only_truth_aperture_and_finite_PSF_convention_diagnostic",
            "uses_true_case_fwhm_for_aperture": True,
            "uses_truth_sky_peak_for_aperture_center": True,
            "operational_total_flux_estimator": False,
            "calibrated_photometric_uncertainty": False,
            "missing_flux_correction": False,
            "residual_interpretation": "Jy/dirty beam residual sum divided by a selected finite dirty-PSF core area; convention dependent, not calibrated total flux",
            "blind_rescaling_warning": "clean-model and dirty-residual beam roles differ; finite dirty-PSF area depends on truncation/sidelobes and can approach zero; a scalar area ratio neither restores missing uv modes nor calibrates correlated noise",
            "model_component_flux_full_jy": float(clean_model.sum()),
            "model_component_flux_aperture_jy": component_aperture,
            "residual_sum_aperture_jy_per_dirty_beam_pixel": residual_sum,
            "dirty_psf_finite_core_area_pixels": psf_area,
            "residual_aperture_flux_estimate_jy": residual_flux,
            "component_plus_residual_aperture_estimate_jy": (component_aperture+residual_flux
                                                              if residual_flux is not None else None),
            "apparent_input_flux_inside_aperture_jy": float((sky*clean_pb)[aperture].sum()),
            "aperture_radius_pixels": float(aperture_radius),
            "residual_scaling": "finite central PSF aperture convention; not a missing-spacing correction",
        }
    else:
        metrics["extended_clean_flux_bookkeeping"] = None
    pass_gates = {}
    pass_gates["all_celestial_grids_aligned_within_0p05_pixel"] = (
        max(alignment_errors.values()) <= alignment_tolerance)
    pass_gates["psf_has_unit_peak_within_1e-3"] = (
        abs(psf_peak-1.) <= TOLERANCES["psf_peak_absolute_error_max"])
    pass_gates["psf_core_matches_required_header_beam_within_10_percent"] = (
        beam_error <= TOLERANCES["psf_vs_image_header_beam_relative_error_max"])
    if case["fwhm_arcsec"] == 0:
        source_y, source_x = np.unravel_index(np.argmax(sky), sky.shape)
        peak_y, peak_x = np.unravel_index(np.argmax(image), image.shape)
        error_pix = pixel_position_error(sky_header, (source_x, source_y),
                                         image_header, (peak_x, peak_y))
        apparent_expected = float(case["input_flux_jy"]) * float(pb[source_y, source_x])
        if not np.isfinite(apparent_expected) or apparent_expected <= 0:
            raise ValueError("Invalid PB-attenuated point-source reference flux")
        flux_error = float(abs(float(image[peak_y, peak_x])/apparent_expected - 1))
        metrics.update({"input_point_flux_jy": case["input_flux_jy"],
                        "apparent_point_flux_expected_jy_beam": apparent_expected,
                        "point_centroid_error_pixels": error_pix,
                        "point_peak_flux_relative_error": flux_error})
        pass_gates["point_peak_flux_within_10_percent"] = bool(flux_error <= TOLERANCES["point_peak_flux_relative_error_max"])
        pass_gates["point_centroid_within_half_pixel"] = bool(error_pix <= TOLERANCES["point_centroid_error_pixels_max"])
    else:
        source_y, source_x = np.unravel_index(np.argmax(sky), sky.shape)
        radius = min(4*float(case["fwhm_arcsec"])/pixel_arcsec,
                     min(image.shape)/2-2)
        moments = positive_core_moments(image, source_x, source_y, radius)
        moments["major_fwhm_arcsec"] = moments["major_fwhm_pixels"]*pixel_arcsec
        moments["minor_fwhm_arcsec"] = moments["minor_fwhm_pixels"]*pixel_arcsec
        moments["input_intrinsic_fwhm_arcsec"] = float(case["fwhm_arcsec"])
        moments["circular_gaussian_plus_fitted_psf_reference_arcsec"] = [
            float(np.hypot(case["fwhm_arcsec"], psf_fit["major_fwhm_arcsec"])),
            float(np.hypot(case["fwhm_arcsec"], psf_fit["minor_fwhm_arcsec"]))]
        metrics["extended_positive_core_moments"] = moments
    metrics["header_beam_vs_psf_fit_relative_error"] = beam_error
    closure = metrics["niter_zero_image_residual_relative_l2"]
    pass_gates["niter_zero_image_equals_residual_within_1e-3"] = bool(
        closure <= TOLERANCES["niter_zero_image_residual_relative_l2_max"])
    pass_gates["measurement_set_metadata_audited"] = bool(channel.get("measurement_set_audit"))
    pass_gates["product_units_have_supported_explicit_or_provenance_backed_interpretations"] = True
    pass_gates["clean_restored_image_equals_restored_model_plus_residual_within_2_percent"] = bool(
        clean_closure <= clean_pixel_tolerance)
    stop = clean_stop_diagnostics(channel.get("clean_return_summary"),
                                  channel["clean_imaging"]["niter"],
                                  float(u.Quantity(channel["clean_imaging"]["threshold"]).to_value(u.Jy)),
                                  float(np.max(np.abs(clean_residual))))
    metrics["clean_stop_diagnostics"] = stop
    metrics["clean_iterations_reported"] = stop["iterations_reported"]
    metrics["clean_iteration_limit_reached"] = stop["iteration_limit_reached"]
    metrics["clean_stop_state"] = stop["state"]
    metrics["reported_vs_measured_threshold_flag_agrees"] = (
        channel.get("clean_threshold_met_by_final_residual") == stop["exported_residual_threshold_met"])
    pass_gates["clean_stop_state_is_auditable"] = stop["stop_reason_auditable"]

    # Deliberately a comparison, not an equivalence test: CASA's vla.c.cfg
    # tracks and this package's artificial Fourier lattice are different instruments.
    sky_wcs = WCS(sky_header).celestial
    cell = float(np.sqrt(abs(np.linalg.det(sky_wcs.pixel_scale_matrix))) * 3600)
    grid = Grid(npix=sky.shape[0], pixel_arcsec=cell, subpixels=1)
    inst_cfg = dict(fast_config["instrument"])
    inst_cfg.update(frequencies_ghz=(channel_frequency(products, case),),
                    bandwidths_mhz=(channel_bandwidth(products, case),))
    inst = Instrument(**inst_cfg)
    omega = abs(np.linalg.det(sky_wcs.pixel_scale_matrix)) * (np.pi/180)**2
    visibility = expected_measurement(sky[None] / omega, grid, inst)
    weights, _ = operator(grid, inst)
    proxy_image = dirty_image(visibility, weights)[0]
    proxy_psf = point_spread_function(weights)[0]
    metrics["fast_proxy_dirty_image_relative_l2"] = relative_l2(image, proxy_image)
    metrics["fast_proxy_psf_relative_l2"] = relative_l2(psf, proxy_psf)
    metrics["fast_proxy_scope"] = "descriptive only: different uv support, antenna/time sampling, CASA primary beam and thermal noise; no pass tolerance"
    metrics["validation_gates"] = pass_gates
    metrics["status"] = "pass" if pass_gates and all(pass_gates.values()) else "fail"
    return metrics


def channel_frequency(products, case):
    # Bundle has one input plane; reading its manifest avoids inventing a CASA
    # output frequency from filenames or a requested configuration.
    bundle = Path(case["bundle_path"])
    manifest = json.loads((bundle / "casa_inputs.json").read_text())
    return float(manifest["channels"][0]["frequency_ghz"])


def channel_bandwidth(products, case):
    bundle = Path(case["bundle_path"])
    manifest = json.loads((bundle / "casa_inputs.json").read_text())
    return float(manifest["channels"][0]["bandwidth_mhz"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--runs", required=True, help="Directory containing genuine CASA run subdirectories")
    parser.add_argument("--fast-config", default="configs/compact_proxy_v2.json")
    parser.add_argument("--out", required=True, help="New report path")
    parser.add_argument("--runtime-provenance", help="Independent hashed historical CASA package/run evidence; native reports stay unchanged")
    parser.add_argument("--noise-diagnostics", help="Preserved independent noise/weight diagnostic report; validation evidence only")
    args = parser.parse_args()
    fixtures = Path(args.fixtures).resolve()
    runs = Path(args.runs).resolve()
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(output)
    runtime_provenance = None
    if args.runtime_provenance:
        runtime_provenance = json.loads(Path(args.runtime_provenance).read_text())
        runtime_provenance["_path"] = str(Path(args.runtime_provenance).resolve())
    manifest_path = fixtures / "cases.json"
    case_data = None
    suite = None
    suite_failure = None
    try:
        case_data = json.loads(manifest_path.read_text())
        suite = validate_case_manifest(case_data)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
        suite_failure = f"{type(error).__name__}: {error}"
        if isinstance(case_data, dict) and isinstance(case_data.get("cases"), list):
            provided = [row.get("bundle") for row in case_data["cases"] if isinstance(row, dict)]
            suite = {"required_cases": list(REQUIRED_CASES), "provided_cases": provided,
                     "missing_required_cases": sorted(set(REQUIRED_CASES)-set(provided)),
                     "complete_required_suite": set(provided) == set(REQUIRED_CASES),
                     "test_double_only": case_data.get("test_double_only") is True}
    results = []
    if suite_failure is None:
        config = load_config(args.fast_config)
        for row in case_data["cases"]:
            case = dict(row)
            case["bundle_path"] = str((fixtures / row["bundle"]).resolve())
            try:
                results.append(validate_case(case, fixtures, runs / row["bundle"], config, runtime_provenance))
            except Exception as error:
                results.append({"case": row["bundle"], "status": "fail",
                                "test_double_only": bool(suite["test_double_only"]),
                                "failure": f"{type(error).__name__}: {error}"})
    case_results_pass = bool(results) and all(x["status"] == "pass" for x in results)
    if suite_failure is not None or not case_results_pass:
        status = "fail"
    elif suite["test_double_only"]:
        status = "test_double_only"
    else:
        status = "pass"
    report = {"schema_version": "1.1", "status": status,
              "product_contract_validation_pass": status == "pass",
              "scientific_validation_pass": False,
              "validation_scope": "Status covers the declared FITS/product, normalization, point-reference and restoration-closure gates only. Extended morphology/flux and fast-proxy differences are descriptive; passing these gates is not overall scientific or instrument-fidelity validation.",
              "purpose": ("synthetic test-double exercise of acceptance/reporting; not CASA validation"
                          if suite and suite["test_double_only"] else
                          "post-run checks of CASA products; no visibility generation or end-to-end scientific fidelity is inferred"),
              "test_double_only": bool(suite and suite["test_double_only"]),
              "required_suite": suite,
              "suite_failure": suite_failure,
              "tolerances": TOLERANCES,
              "input_manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
              "fast_config": str(Path(args.fast_config).resolve()),
              "fast_config_sha256": sha256(args.fast_config) if Path(args.fast_config).is_file() else None,
              "software_versions": versions(),
              "source_code_digest": code_digest(),
              "source_file_sha256": source_manifest(),
              "validator_sha256": sha256(__file__),
              "external_runtime_evidence": ({"path": runtime_provenance["_path"],
                                             "sha256": sha256(runtime_provenance["_path"]),
                                             "versions": runtime_provenance["versions"],
                                             "noise_checks": runtime_provenance.get("noise_checks"),
                                             "absolute_inverse_variance_calibration_assumed": False,
                                             "weight_interpretation": "Use the attached measured noisy-minus-noiseless noise statistics and independent diagnostics. Native WEIGHT/SIGMA consistency alone does not establish physical inverse variance; no physical noise scale is inferred from those columns here."}
                                            if runtime_provenance else None),
              "external_noise_diagnostics": ({"path": str(Path(args.noise_diagnostics).resolve()),
                                               "sha256": sha256(args.noise_diagnostics),
                                               "content": json.loads(Path(args.noise_diagnostics).read_text())}
                                              if args.noise_diagnostics else None),
              "cases": results}
    serialized = strict_json_dumps(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized)
    print(json.dumps({"status": report["status"], "report": str(output),
                      "case_statuses": {r["case"]: r["status"] for r in results}}, indent=2))
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
