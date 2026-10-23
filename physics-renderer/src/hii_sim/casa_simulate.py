"""Standalone CASA 6.7.4 integration, with portable protected input bundles.

This run directory contains simulation truth/provenance and is a protected
research artifact, never a predictor-facing export.
"""
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import sys
import platform
import math
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def casa_runtime_versions(tasks, tools_module=None):
    """Capture installed CASA distribution versions without guessing a release."""
    from importlib import metadata

    def installed(distribution):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None

    task_version = installed("casatasks") or getattr(tasks, "__version__", None)
    tool_version = installed("casatools")
    if tool_version is None and tools_module is not None:
        tool_version = getattr(tools_module, "__version__", None)
    return {
        "casatasks": str(task_version) if task_version is not None else "unknown",
        "casatools": str(tool_version) if tool_version is not None else "unknown",
        "version_source": "importlib.metadata; module __version__ fallback only",
    }


def resolve_input(input_dir, relative_path):
    """Resolve a relocatable member; reject old host-absolute paths."""
    value = Path(relative_path)
    if value.is_absolute() or PureWindowsPath(relative_path).is_absolute():
        raise ValueError("Re-prepare inputs: CASA v2 requires bundle-relative paths")
    root = Path(input_dir).resolve()
    target = (root / value).resolve()
    if root not in target.parents or not target.is_file():
        raise ValueError("Missing or out-of-bundle CASA input: " + relative_path)
    return target


def discover_measurement_sets(project_dir, noisy=True):
    """Inspect products, never infer a path from an array configuration name."""
    products = sorted(p for p in Path(project_dir).rglob("*.ms")
                      if p.is_dir() and (p / "table.dat").is_file())
    selected = [p for p in products if p.name.endswith(".noisy.ms") == noisy]
    if len(selected) != 1:
        kind = "noisy" if noisy else "noise-free"
        raise RuntimeError("Expected exactly one %s Measurement Set in %s; found %s"
                           % (kind, project_dir, [str(p) for p in products]))
    return selected[0], products


def _table_read(factory, path, columns):
    tool = factory()
    try:
        if not tool.open(str(path)):
            raise RuntimeError("CASA table tool could not open " + str(path))
        names = set(tool.colnames())
        absent = set(columns) - names
        if absent:
            raise RuntimeError("Measurement Set lacks required columns %s: %s" %
                               (sorted(absent), path))
        return {name: np.asarray(tool.getcol(name)) for name in columns}
    finally:
        try:
            tool.close()
        except Exception:
            pass


def audit_measurement_set(path, table_tool_factory):
    """Audit MAIN sample/whole-row flags; weights remain native, uncalibrated."""
    path = Path(path)
    main = _table_read(table_tool_factory, path,
                       ("DATA", "FLAG", "FLAG_ROW", "UVW", "WEIGHT", "TIME", "ANTENNA1", "ANTENNA2"))
    data = main["DATA"]
    flags = main["FLAG"].astype(bool)
    row_flags = main["FLAG_ROW"].astype(bool)
    weights = main["WEIGHT"].astype(float)
    uvw = main["UVW"].astype(float)
    if (data.ndim != 3 or data.shape != flags.shape or weights.shape != (data.shape[0], data.shape[2])
            or uvw.shape != (3, data.shape[2]) or main["TIME"].shape != (data.shape[2],)
            or main["ANTENNA1"].shape != (data.shape[2],) or main["ANTENNA2"].shape != (data.shape[2],)):
        raise RuntimeError("Unexpected Measurement Set DATA/FLAG/WEIGHT/UVW shapes")
    if row_flags.shape != (data.shape[2],):
        raise RuntimeError("Unexpected Measurement Set FLAG_ROW shape; expected one Boolean per MAIN row")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise RuntimeError("Selected Measurement Set has no valid finite positive-weight data")
    effective_flags = flags | row_flags[None, None, :]
    usable = ((~effective_flags) & np.isfinite(data.real) & np.isfinite(data.imag)
              & (weights[:, None, :] > 0))
    if not usable.any():
        raise RuntimeError("Selected Measurement Set has no valid unflagged positive-weight data after FLAG and FLAG_ROW")
    row_baselines = np.hypot(uvw[0], uvw[1])
    usable_row_baselines = row_baselines[usable.any(axis=(0, 1))]
    positive_usable_baselines = usable_row_baselines[usable_row_baselines > 0]
    all_positive_weights = weights[weights > 0]
    usable_weights = weights[usable.any(axis=1)]
    spw = _table_read(table_tool_factory, path / "SPECTRAL_WINDOW", ("CHAN_FREQ", "CHAN_WIDTH"))
    antenna = _table_read(table_tool_factory, path / "ANTENNA", ("POSITION",))
    return {
        "rows": int(data.shape[-1]), "data_shape": list(data.shape),
        "correlations": int(data.shape[0]), "channels": int(data.shape[1]),
        "flag_fraction": float(flags.mean()),
        "sample_flag_fraction": float(flags.mean()),
        "row_flag_fraction": float(row_flags.mean()),
        "effective_flag_fraction": float(effective_flags.mean()),
        "flag_diagnostic_scope": "flag_fraction is the legacy alias of sample_flag_fraction (FLAG only); row_flag_fraction is FLAG_ROW; effective_flag_fraction counts FLAG OR broadcast FLAG_ROW",
        "valid_unflagged_fraction": float(usable.mean()),
        "usable_sample_count": int(usable.sum()),
        "unflagged_amplitude_rms_jy": float(np.sqrt(np.mean(np.abs(data[usable])**2))),
        "weight_positive_fraction": float(np.mean(weights > 0)),
        "weight_min_positive": float(all_positive_weights.min()),
        "weight_median_positive": float(np.median(all_positive_weights)),
        "weight_max": float(all_positive_weights.max()),
        "weight_summary_scope": "weight_* summaries cover all MAIN WEIGHT cells before flags; positive summaries exclude zero. Native magnitudes are not interpreted as calibrated inverse variances.",
        "usable_weight_summary": {"correlation_row_count": int(usable_weights.size),
                                  "min": float(usable_weights.min()),
                                  "median": float(np.median(usable_weights)),
                                  "max": float(usable_weights.max()),
                                  "scope": "one native WEIGHT per correlation/row with at least one usable channel after FLAG, FLAG_ROW, finite-DATA and positive-weight selection"},
        "uvw_baseline_min_m": float(row_baselines.min()),
        "uvw_baseline_median_m": float(np.median(row_baselines)),
        "uvw_baseline_max_m": float(row_baselines.max()),
        "usable_uvw_baseline_min_positive_m": (float(positive_usable_baselines.min())
                                              if positive_usable_baselines.size else None),
        "usable_uvw_baseline_max_m": float(usable_row_baselines.max()),
        "usable_zero_projected_baseline_rows": int(np.sum(usable_row_baselines == 0)),
        "time_range_seconds": [float(main["TIME"].min()), float(main["TIME"].max())],
        "antenna_count": int(len(np.unique(np.r_[main["ANTENNA1"], main["ANTENNA2"]]))),
        "channel_frequency_hz": np.asarray(spw["CHAN_FREQ"]).reshape(-1).astype(float).tolist(),
        "channel_width_hz": np.asarray(spw["CHAN_WIDTH"]).reshape(-1).astype(float).tolist(),
        "antenna_position_count": int(np.asarray(antenna["POSITION"]).shape[-1]),
        "validation": "metadata audit of selected MS; no claim of calibration or visibility fidelity",
    }


def observed_noise_rms(fits_path, border_fraction=0.15):
    """Robust off-edge scatter of the niter=0 image; no truth mask is used."""
    with fits.open(fits_path, memmap=False) as hdus:
        image = np.asarray(hdus[0].data, dtype=float).squeeze()
    if image.ndim != 2 or not np.isfinite(image).all():
        raise RuntimeError("Cannot estimate observed noise from non-finite 2-D image")
    ny, nx = image.shape
    edge_y, edge_x = max(1, int(ny * border_fraction)), max(1, int(nx * border_fraction))
    yy, xx = np.indices(image.shape)
    region = (yy < edge_y) | (yy >= ny-edge_y) | (xx < edge_x) | (xx >= nx-edge_x)
    sample = image[region]
    center = float(np.median(sample))
    rms = float(1.482602218505602 * np.median(np.abs(sample-center)))
    if not np.isfinite(rms) or rms <= 0:
        raise RuntimeError("Observed off-edge noise estimate is zero or invalid")
    return {"rms_jy_per_beam": rms, "method": "1.4826022185*MAD from fixed outer 15% image border",
            "pixels": int(sample.size), "region": "fixed image border; no source/truth mask"}


def clean_summary_fields(summary):
    """Preserve CASA's return tree, including task/minor-cycle stop evidence.

    Field/channel/stokes dictionaries stay separate. Non-finite sentinels are
    represented explicitly rather than emitted as nonstandard JSON numbers.
    """
    if not isinstance(summary, dict):
        return None

    def serializable(value):
        if isinstance(value, np.ndarray):
            return serializable(value.tolist())
        if isinstance(value, np.generic):
            return serializable(value.item())
        if isinstance(value, dict):
            return {str(key): serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite_float": repr(value)}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError("Unsupported CASA convergence summary value: " + type(value).__name__)

    return serializable(summary) or None


def clean_stop_diagnostics(summary, niter, threshold_jy, exported_peak_jy):
    """Separate task exit evidence from a strict whole-export residual check.

    A task exit code or a residual below threshold alone is not a scientific
    convergence certificate. Do not reinterpret per-plane minor stopCode as
    the task's global stopcode, or flatten iteration histories across planes.
    """
    if (not math.isfinite(threshold_jy) or threshold_jy < 0 or
            not math.isfinite(exported_peak_jy) or exported_peak_jy < 0):
        raise ValueError("Stopping diagnostics require finite non-negative threshold and residual peak")
    summary = summary if isinstance(summary, dict) else {}
    iterations = summary.get("iterdone", summary.get("iterDone"))
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)):
        iterations = None
    code = summary.get("stopcode")
    if isinstance(code, bool) or not isinstance(code, (int, np.integer)):
        code = None
    description = summary.get("stopDescription")
    hit_limit = iterations is not None and iterations >= niter
    exported_met = exported_peak_jy <= threshold_jy
    state = ("task_reported_stop" if code is not None else
             "iteration_limit" if hit_limit else
             "exported_residual_below_threshold_task_reason_unknown" if exported_met else
             "unresolved_task_stop_reason")
    return {"state": state, "task_global_stopcode": int(code) if code is not None else None,
            "task_stop_description": description if isinstance(description, str) else None,
            "iterations_reported": int(iterations) if iterations is not None else None,
            "iteration_limit_reached": bool(hit_limit),
            "exported_residual_peak_jy_dirty_beam": float(exported_peak_jy),
            "threshold_jy": float(threshold_jy),
            "exported_residual_threshold_met": bool(exported_met),
            "exported_residual_minus_threshold_jy": float(exported_peak_jy-threshold_jy),
            "exported_peak_over_threshold": float(exported_peak_jy/threshold_jy) if threshold_jy > 0 else None,
            "stop_reason_auditable": bool(code is not None or hit_limit or exported_met),
            "scientific_convergence_verified": False,
            "comparison_scope": "strict global absolute exported residual peak; CASA may use mask/plane/cycle statistics at different stages; no tolerance or threshold alteration"}


def run_simobserve_in_output(tasks, output_root, project_name, **kwargs):
    """Use CASA's relative project-name contract while preserving caller cwd."""
    if (not isinstance(project_name, str) or not project_name or
            Path(project_name).name != project_name or PureWindowsPath(project_name).name != project_name):
        raise ValueError("CASA simobserve project must be a simple basename")
    output_root = Path(output_root).resolve()
    if not output_root.is_dir():
        raise ValueError("CASA output root must exist before simobserve")
    original_cwd = Path.cwd()
    try:
        os.chdir(output_root)
        return tasks.simobserve(project=project_name, **kwargs)
    finally:
        os.chdir(original_cwd)


def _check_reconstruction_options(policy, multiplier, noisy):
    if policy not in {"hogbom", "multiscale"}:
        raise ValueError("Supported reconstruction policies are hogbom and coarse multiscale")
    if (isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or
            not math.isfinite(multiplier) or multiplier <= 0):
        raise ValueError("noise_multiplier must be finite and positive")
    if not isinstance(noisy, bool):
        raise ValueError("noisy must be an explicit Boolean")
    if policy == "multiscale" and not noisy:
        raise ValueError("The supported coarse policy requires noisy observations; use legacy hogbom for noiseless controls")


def _observing_settings(settings):
    """Allow observing controls only; never accept a starting sky/model/mask."""
    allowed = {"datacolumn", "specmode", "gridder", "imsize", "cell", "phasecenter",
               "weighting", "pblimit", "pbcor", "interactive", "parallel"}
    if not isinstance(settings, dict) or set(settings)-allowed:
        raise ValueError("Unsupported observing settings; source/model/mask and deconvolution controls are prohibited")
    required = {"imsize", "cell", "phasecenter"}
    if not required <= set(settings):
        raise ValueError("Observing settings require imsize, cell and phasecenter")
    defaults = dict(datacolumn="data", specmode="mfs", gridder="standard", weighting="natural",
                    pblimit=0.2, pbcor=False, interactive=False, parallel=False)
    result = defaults | settings
    for key in ("datacolumn", "specmode", "gridder", "weighting", "pbcor", "interactive", "parallel"):
        if result[key] != defaults[key]:
            raise ValueError("Current reconstruction scope requires %s=%r" % (key, defaults[key]))
    return result


def _policy_module():
    if __package__:
        from . import casa_reconstruction_policy
        return casa_reconstruction_policy
    # CASA -c can execute the runner outside a package. Load its adjacent,
    # explicitly versioned policy file without depending on host cwd/sys.path.
    import importlib.util
    path = Path(__file__).with_name("casa_reconstruction_policy.py")
    spec = importlib.util.spec_from_file_location("hii_observable_casa_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruction_implementation():
    files = [Path(__file__), Path(__file__).with_name("casa_reconstruction_policy.py")]
    return {path.name: sha256(path) for path in files if path.is_file()}


def measurement_set_checksums(ms):
    """Content audit only; CASA's transient table.lock files are excluded."""
    root = Path(ms)
    return {path.relative_to(root).as_posix(): sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file() and path.name != "table.lock"}


def reconstruct_measurements(ms, output_root, tag, observing_settings, *, tasks,
                             ms_audit, policy="multiscale", noise_multiplier=3.0, noisy=True):
    """Shared measurement-only dirty/CLEAN path for simulated or existing MS.

    Caller owns a fresh output directory and failure report. No truth bundle,
    physical labels or starting sky image enters this function.
    """
    _check_reconstruction_options(policy, noise_multiplier, noisy)
    settings = _observing_settings(observing_settings)
    ms, out = Path(ms).resolve(), Path(output_root).resolve()
    if out == ms or ms in out.parents:
        raise ValueError("Reconstruction output must be outside the input Measurement Set")
    if (not isinstance(tag, str) or not tag or Path(tag).name != tag or
            PureWindowsPath(tag).name != tag or tag in {".", ".."}):
        raise ValueError("Channel tag must be a simple basename")
    if not out.is_dir() or not (ms / "table.dat").is_file():
        raise ValueError("Reconstruction requires an existing MS and fresh output directory")
    if any(out.glob(tag+".*")):
        raise FileExistsError("Reconstruction prefix already has products: " + str(out/tag))

    def export(prefix, roles, clean=False):
        products = {}
        for role in roles:
            native = Path(str(prefix)+"."+role)
            if not native.is_dir():
                raise RuntimeError("CASA did not produce required image: " + str(native))
            target = out / (tag+(".clean." if clean else ".")+role+".fits")
            tasks.exportfits(imagename=str(native), fitsimage=str(target), dropdeg=True, overwrite=False)
            if not target.is_file():
                raise RuntimeError("CASA exportfits did not produce " + str(target))
            products[role] = {"path": target.name, "sha256": sha256(target)}
        return products

    dirty = settings | dict(vis=str(ms), imagename=str(out/(tag+".dirty")),
                            deconvolver="hogbom", niter=0, savemodel="none", restart=False)
    tasks.tclean(**dirty)
    dirty_products = export(out/(tag+".dirty"), ("image", "psf", "pb", "residual"))
    noise = (observed_noise_rms(out/dirty_products["residual"]["path"]) if noisy else
             {"rms_jy_per_beam": 0.0, "method": "no thermal noise requested; no estimated noise floor",
              "pixels": 0, "region": "not applicable"})
    configured_policy = None
    if policy == "multiscale":
        with fits.open(out/dirty_products["image"]["path"], memmap=False) as hdus:
            shape = np.asarray(hdus[0].data).squeeze().shape
            header = hdus[0].header.copy()
        matrix = np.asarray(WCS(header).celestial.pixel_scale_matrix, dtype=float)
        lengths = np.linalg.norm(matrix, axis=0)
        if (len(shape) != 2 or shape[0] != shape[1] or not np.isfinite(matrix).all()
                or np.any(lengths <= 0) or not np.isclose(lengths[0], lengths[1], rtol=1e-6)
                or abs(np.dot(matrix[:, 0], matrix[:, 1])) > 1e-6*np.prod(lengths)):
            raise ValueError("Coarse multiscale scope requires a square image and square orthogonal celestial pixels")
        frequencies = ms_audit["channel_frequency_hz"]
        minimum = ms_audit.get("usable_uvw_baseline_min_positive_m")
        if (len(frequencies) != 1 or not math.isfinite(frequencies[0]) or frequencies[0] <= 0
                or minimum is None or not math.isfinite(minimum) or minimum <= 0):
            raise ValueError("Coarse multiscale requires one observed channel and a positive usable projected baseline")
        configured_policy = _policy_module().configured_multiscale_policy(
            noise_multiplier=noise_multiplier, beam_major_arcsec=float(header["BMAJ"])*3600.,
            cell_arcsec=float(np.sqrt(abs(np.linalg.det(matrix))))*3600., npix=int(shape[0]),
            min_usable_uv_lambda=minimum*frequencies[0]/299792458.,
            observed_border_rms_jy_beam=noise["rms_jy_per_beam"])
        controls = configured_policy["tclean_overrides"]
    else:
        controls = dict(deconvolver="hogbom", niter=5000, gain=0.1,
                        threshold="%.12gJy" % (noise_multiplier*noise["rms_jy_per_beam"]),
                        fullsummary=True, savemodel="none", restart=False)
    clean = dirty | controls | dict(imagename=str(out/(tag+".clean")))
    summary = clean_summary_fields(tasks.tclean(**clean))
    clean_products = export(out/(tag+".clean"), ("image", "model", "residual", "psf", "pb"), clean=True)
    with fits.open(out/clean_products["residual"]["path"], memmap=False) as hdus:
        residual = np.asarray(hdus[0].data, dtype=float).squeeze()
    if residual.ndim != 2 or not np.isfinite(residual).all():
        raise RuntimeError("Exported CLEAN residual must be a finite 2-D image")
    peak = float(np.max(np.abs(residual)))
    threshold = float(clean["threshold"][:-2])
    return {"channel": tag, "measurement_set_audit": ms_audit,
            "dirty_imaging": {k: v for k, v in dirty.items() if k not in {"vis", "imagename"}},
            "dirty_products": dirty_products,
            "clean_imaging": {k: v for k, v in clean.items() if k not in {"vis", "imagename"}},
            "clean_products": clean_products, "observed_noise_estimate": noise,
            "reconstruction_policy": policy, "configured_observable_policy": configured_policy,
            "threshold_policy": {"noise_multiplier": float(noise_multiplier),
                                 "noise_proxy_jy_dirty_beam": noise["rms_jy_per_beam"],
                                 "serialized_threshold": clean["threshold"], "task_threshold_jy": threshold,
                                 "calibrated_thermal_sigma": False},
            "clean_return_summary": summary, "clean_final_residual_peak_jy_beam": peak,
            "clean_threshold_met_by_final_residual": bool(peak <= threshold),
            "clean_stop_diagnostics": clean_stop_diagnostics(summary, clean["niter"], threshold, peak),
            "clean_stop_policy": "5000-iteration cap; explicit multiplier times observed border MAD; strict exported check and raw task exit are distinct",
            "clean_flux_units": {"model": "Jy/pixel", "image": "native composite: Jy/clean beam model plus Jy/dirty beam residual",
                                 "residual": "Jy/dirty beam"},
            "reconstruction_implementation_sha256": reconstruction_implementation(),
            "scientific_fidelity_certified": False}


def run(input_dir, output_dir, tasks=None, table_tool_factory=None, *,
        reconstruction_policy="hogbom", noise_multiplier=3.0):
    """Run real CASA tasks, or explicitly supplied test doubles in unit tests."""
    if tasks is None:
        import casatasks as tasks
    root = Path(input_dir).resolve()
    manifest_path = root / "casa_inputs.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "2.0":
        raise ValueError("Re-prepare legacy CASA input with the current prepare-casa command")
    _check_reconstruction_options(reconstruction_policy, noise_multiplier, bool(manifest["thermalnoise"]))
    pointings = resolve_input(root, manifest["pointings"])
    if sha256(pointings) != manifest["pointings_sha256"]:
        raise ValueError("CASA pointing file checksum mismatch")
    prepared = []
    for channel in manifest["channels"]:
        path = resolve_input(root, channel["fits"])
        if sha256(path) != channel["sha256"]:
            raise ValueError("CASA sky input checksum mismatch: " + channel["fits"])
        prepared.append((channel, path))
    tools_module = None
    if table_tool_factory is None:
        import casatools as tools_module
        table_tool_factory = tools_module.table
    package_versions = casa_runtime_versions(tasks, tools_module)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "2.0", "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "casa_version": package_versions["casatasks"],
              "casa_packages": package_versions,
              "imaging_producer": "casatasks.tclean",
              "python_version": sys.version,
              "platform": platform.platform(),
              "runner_sha256": sha256(__file__), "input_manifest_sha256": sha256(manifest_path),
              "configuration": manifest["casa_array"], "noise": manifest["thermalnoise"],
              "calibration": "No gain/phase corruption or calibration solutions",
              "measurement_product": "noisy" if manifest["thermalnoise"] else "noise-free",
              "reconstruction_policy": reconstruction_policy, "noise_multiplier": float(noise_multiplier),
              "reconstruction_implementation_sha256": reconstruction_implementation(),
              "products": []}
    (out / "casa_inputs.used.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for channel, model_path in prepared:
            tag = channel["tag"]
            if Path(tag).name != tag or PureWindowsPath(tag).name != tag:
                raise ValueError("Invalid channel tag")
            project = out / tag
            run_simobserve_in_output(
                tasks, out, tag, skymodel=str(model_path),
                incenter="%.9fGHz" % channel["frequency_ghz"],
                inwidth="%.6fMHz" % channel["bandwidth_mhz"],
                antennalist=manifest["casa_array"], obsmode="int",
                setpointings=False, ptgfile=str(resolve_input(root, manifest["pointings"])),
                direction=manifest["direction"], totaltime=manifest["duration"],
                integration=manifest["integration"], hourangle=manifest["hourangle"],
                refdate=manifest["reference_date"], outframe="TOPO",
                thermalnoise=manifest["thermalnoise"], seed=int(channel["seed"]),
                graphics="", overwrite=False)
            ms, all_ms = discover_measurement_sets(project, noisy=bool(manifest["thermalnoise"]))
            ms_audit = audit_measurement_set(ms, table_tool_factory)
            settings = dict(imsize=manifest["npix"], cell=manifest["cell"], phasecenter=manifest["direction"])
            channel_report = reconstruct_measurements(
                ms, out, tag, settings, tasks=tasks, ms_audit=ms_audit,
                policy=reconstruction_policy, noise_multiplier=noise_multiplier,
                noisy=bool(manifest["thermalnoise"]))
            channel_report.update(measurement_set=ms.relative_to(out).as_posix(),
                                  discovered_measurement_sets=[p.relative_to(out).as_posix() for p in all_ms])
            report["products"].append(channel_report)
        report["status"] = "completed_unvalidated"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "casa_products.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def reimage_saved_ms(ms, settings_path, output_dir, tasks=None, table_tool_factory=None, *,
                     reconstruction_policy=None, noise_multiplier=None):
    """Reconstruct saved measurements without opening any simulation inputs."""
    settings_path, ms = Path(settings_path).resolve(), Path(ms).resolve()
    config = json.loads(settings_path.read_text())
    if set(config)-{"schema_version", "imaging", "reconstruction"} or config.get("schema_version") != "1.0":
        raise ValueError("Saved-MS settings require observing-only schema_version 1.0")
    settings = _observing_settings(config["imaging"])
    options = config.get("reconstruction", {})
    if set(options)-{"policy", "noise_multiplier", "noisy"}:
        raise ValueError("Unknown reconstruction option")
    policy = reconstruction_policy if reconstruction_policy is not None else options.get("policy", "multiscale")
    multiplier = noise_multiplier if noise_multiplier is not None else options.get("noise_multiplier", 3.0)
    noisy = options.get("noisy", True)
    _check_reconstruction_options(policy, multiplier, noisy)
    out = Path(output_dir).resolve()
    if out == ms or ms in out.parents:
        raise ValueError("Reconstruction output must be outside the input Measurement Set")
    if not (ms/"table.dat").is_file():
        raise ValueError("Input must be an existing Measurement Set")
    if tasks is None:
        import casatasks as tasks
    tools_module = None
    if table_tool_factory is None:
        import casatools as tools_module
        table_tool_factory = tools_module.table
    package_versions = casa_runtime_versions(tasks, tools_module)
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "2.0", "status": "running",
              "product_origin": "measurement_only_reconstruction_of_existing_ms",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "casa_version": package_versions["casatasks"], "casa_packages": package_versions,
              "imaging_producer": "casatasks.tclean", "runner_sha256": sha256(__file__),
              "python_version": sys.version, "platform": platform.platform(),
              "input_measurement_set": str(ms), "observing_settings_sha256": sha256(settings_path),
              "reconstruction_policy": policy, "noise_multiplier": float(multiplier),
              "reconstruction_implementation_sha256": reconstruction_implementation(),
              "products": [], "scientific_fidelity_certified": False}
    effective = {"schema_version": "1.0", "imaging": settings,
                 "reconstruction": {"policy": policy, "noise_multiplier": float(multiplier), "noisy": noisy}}
    (out/"observing_settings.used.json").write_text(json.dumps(effective, indent=2)+"\n")
    try:
        report["input_ms_content_sha256_before"] = measurement_set_checksums(ms)
        audit = audit_measurement_set(ms, table_tool_factory)
        result = reconstruct_measurements(ms, out, "ch00", settings, tasks=tasks, ms_audit=audit,
                                          policy=policy, noise_multiplier=multiplier, noisy=noisy)
        result.update(measurement_set=str(ms), discovered_measurement_sets=[str(ms)])
        report["products"].append(result)
        report["input_ms_content_sha256_after"] = measurement_set_checksums(ms)
        report["input_ms_unchanged"] = (report["input_ms_content_sha256_before"] == report["input_ms_content_sha256_after"])
        if not report["input_ms_unchanged"]:
            raise RuntimeError("Input Measurement Set content changed during reconstruction")
        report["status"] = "completed_unvalidated"
    except Exception as error:
        report.update(status="failed", error=type(error).__name__+": "+str(error))
        raise
    finally:
        if "input_ms_content_sha256_before" in report and "input_ms_content_sha256_after" not in report:
            try:
                report["input_ms_content_sha256_after"] = measurement_set_checksums(ms)
                report["input_ms_unchanged"] = (report["input_ms_content_sha256_before"] == report["input_ms_content_sha256_after"])
            except OSError as error:
                report["input_ms_integrity_error"] = type(error).__name__+": "+str(error)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (out/"casa_products.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main(argv=None):
    import argparse
    args = (sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]) if argv is None else argv
    if args and args[0] == "reimage":
        parser = argparse.ArgumentParser(description="Measurement-only CASA reimaging; creates a fresh output directory")
        parser.add_argument("--ms", required=True)
        parser.add_argument("--settings", required=True)
        parser.add_argument("--out", required=True)
        parser.add_argument("--policy", choices=("hogbom", "multiscale"))
        parser.add_argument("--noise-multiplier", type=float)
        options = parser.parse_args(args[1:])
        return reimage_saved_ms(options.ms, options.settings, options.out,
                               reconstruction_policy=options.policy, noise_multiplier=options.noise_multiplier)
    parser = argparse.ArgumentParser(description="CASA simulation with shared measurement-only reconstruction")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--policy", choices=("hogbom", "multiscale"), default="hogbom")
    parser.add_argument("--noise-multiplier", type=float, default=3.0)
    options = parser.parse_args(args)
    return run(options.input_dir, options.output_dir, reconstruction_policy=options.policy,
               noise_multiplier=options.noise_multiplier)


if __name__ == "__main__":
    main()
