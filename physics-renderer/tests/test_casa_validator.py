"""Safety checks for the real-product CASA post-validator."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS


def load_validator():
    path = Path(__file__).parents[1] / "tools" / "validate_casa_run.py"
    spec = importlib.util.spec_from_file_location("validate_casa_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extended_bookkeeping_labels_truth_aperture_without_changing_numbers(tmp_path):
    validator=load_validator()
    fixtures,runs=make_test_double(tmp_path)
    config=json.loads(Path("configs/compact_proxy_v2.json").read_text())
    config["instrument"].update(min_baseline_m=0.,max_short_spacing_ratio=None)
    case={"bundle":"point_noiseless","bundle_path":str(fixtures/"point_noiseless"),
          "fwhm_arcsec":45.,"test_double_only":True}
    result=validator.validate_case(case,fixtures,runs/"point_noiseless",config)
    bookkeeping=result["extended_clean_flux_bookkeeping"]
    assert bookkeeping["uses_true_case_fwhm_for_aperture"] is True
    assert bookkeeping["uses_truth_sky_peak_for_aperture_center"] is True
    assert bookkeeping["operational_total_flux_estimator"] is False
    assert bookkeeping["calibrated_photometric_uncertainty"] is False
    assert bookkeeping["missing_flux_correction"] is False
    assert "validation_only" in bookkeeping["usage"]
    assert "missing uv modes" in bookkeeping["blind_rescaling_warning"]
    assert bookkeeping["aperture_radius_pixels"]==15.
    assert bookkeeping["model_component_flux_full_jy"]==float(np.float32(.01))
    assert bookkeeping["model_component_flux_aperture_jy"]==float(np.float32(.01))
    assert bookkeeping["residual_aperture_flux_estimate_jy"]==0.
    assert bookkeeping["component_plus_residual_aperture_estimate_jy"]==float(np.float32(.01))
    assert bookkeeping["residual_scaling"]=="finite central PSF aperture convention; not a missing-spacing correction"


def test_nonzero_residual_accounting_matches_immutable_0a31_validator(tmp_path):
    validator=load_validator();fixtures,runs=make_test_double(tmp_path)
    run=runs/"point_noiseless";report_path=run/"casa_products.json"
    report=json.loads(report_path.read_text())
    for role in ("image","residual"):
        entry=report["products"][0]["clean_products"][role];path=run/entry["path"]
        with fits.open(path,memmap=False) as hdus:
            data=hdus[0].data.copy()+np.float32(.0001);header=hdus[0].header.copy()
        fits.PrimaryHDU(data,header=header).writeto(path,overwrite=True)
        entry["sha256"]=validator.sha256(path)
    report_path.write_text(json.dumps(report))
    config=json.loads(Path("configs/compact_proxy_v2.json").read_text())
    config["instrument"].update(min_baseline_m=0.,max_short_spacing_ratio=None)
    case={"bundle":"point_noiseless","bundle_path":str(fixtures/"point_noiseless"),
          "fwhm_arcsec":45.,"test_double_only":True}
    snapshot=Path("output/audit/gaussian_gl4_repair_20261021/snapshot/tools/validate_casa_run.py")
    spec=importlib.util.spec_from_file_location("historical_validator_0a31",snapshot)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    original=old.validate_case(case,fixtures,run,config)["extended_clean_flux_bookkeeping"]
    revised=validator.validate_case(case,fixtures,run,config)["extended_clean_flux_bookkeeping"]
    assert original["residual_aperture_flux_estimate_jy"]>0
    # Every historical numeric value, unit-bearing key and convention remains identical.
    assert {key:revised[key] for key in original}==original


def test_validator_refuses_incomplete_casa_run_without_products(tmp_path):
    validator = load_validator()
    run = tmp_path / "case"
    run.mkdir()
    (run / "casa_products.json").write_text(json.dumps({
        "status": "running", "products": []
    }))
    with pytest.raises(RuntimeError, match="not task-complete"):
        validator.validate_case({"bundle": "point_noiseless", "fwhm_arcsec": 0},
                                tmp_path, run, {})


def test_predeclared_casa_tolerances_are_explicit():
    tolerances = load_validator().TOLERANCES
    assert tolerances == {
        "point_peak_flux_relative_error_max": 0.10,
        "point_centroid_error_pixels_max": 0.5,
        "psf_vs_image_header_beam_relative_error_max": 0.10,
        "niter_zero_image_residual_relative_l2_max": 1e-3,
        "product_grid_alignment_max_pixels": 0.05,
        "psf_peak_absolute_error_max": 1e-3,
    }


def test_required_suite_definition_and_explicit_narrow_test_double():
    validator = load_validator()
    cases = [{"bundle": name, **definition, "input_flux_jy": 0.010}
             for name, definition in validator.REQUIRED_CASES.items()]
    suite = validator.validate_case_manifest({"cases": cases})
    assert suite["complete_required_suite"] is True
    assert suite["missing_required_cases"] == []
    narrow = validator.validate_case_manifest({
        "test_double_only": True,
        "cases": [{"bundle": "point_noiseless", "fwhm_arcsec": 0.,
                   "thermalnoise": "", "input_flux_jy": .01,
                   "test_double_only": True}],
    })
    assert narrow["complete_required_suite"] is False
    assert narrow["test_double_only"] is True


@pytest.mark.parametrize("case_data,match", [
    ({"cases": []}, "suite is empty"),
    ({"cases": [{"bundle": "point_noiseless", "fwhm_arcsec": 0.,
                 "thermalnoise": "", "input_flux_jy": .01}]},
     "Missing required CASA validation cases"),
])
def test_empty_or_incomplete_suite_writes_useful_failure_report(tmp_path, monkeypatch, case_data, match):
    validator = load_validator()
    fixtures, runs = tmp_path / "fixtures", tmp_path / "runs"
    fixtures.mkdir()
    (fixtures / "cases.json").write_text(json.dumps(case_data))
    output = tmp_path / "failure.json"
    monkeypatch.setattr(sys, "argv", ["validate_casa_run.py", "--fixtures", str(fixtures),
                                      "--runs", str(runs), "--out", str(output)])
    with pytest.raises(SystemExit, match="1"):
        validator.main()
    report = json.loads(output.read_text(), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    assert report["status"] == "fail"
    assert report["scientific_validation_pass"] is False
    assert report["product_contract_validation_pass"] is False
    assert match in report["suite_failure"]
    assert report["required_suite"]["missing_required_cases"]
    assert report["cases"] == []


def test_strict_json_rejects_nonfinite_numpy_metrics():
    validator = load_validator()
    with pytest.raises(ValueError, match="Non-finite"):
        validator.strict_json_dumps({"metric": np.float64(np.nan)})


def test_validation_report_records_exact_code_config_and_measured_clean_closure(monkeypatch, tmp_path):
    report = run_test_report(monkeypatch, tmp_path)
    assert report["source_code_digest"]
    assert report["source_file_sha256"]["src/hii_sim/casa_simulate.py"]
    assert report["validator_sha256"]
    assert report["fast_config_sha256"]
    measured = report["cases"][0]["clean_restoration_closure_relative_l2"]
    assert measured < 1e-6
    assert report["cases"][0]["clean_restoration_closure_relative_l2"] != 0.02


def test_positive_core_moment_definition_on_gaussian_array():
    validator = load_validator()
    y, x = np.indices((65, 65))
    sigma = 8 / np.sqrt(8*np.log(2))
    image = np.exp(-((x-32)**2+(y-32)**2)/(2*sigma**2))
    result = validator.positive_core_moments(image, 32, 32, 24)
    assert result["major_fwhm_pixels"] == pytest.approx(8, rel=0.01)
    assert result["minor_fwhm_pixels"] == pytest.approx(8, rel=0.01)
    assert result["centroid_offset_pixels"] == pytest.approx(0, abs=1e-12)


def test_declared_beam_kernel_uses_bpa_and_rectangular_pixel_geometry():
    validator = load_validator()
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [180., 30.]
    wcs.wcs.crpix = [33., 33.]
    wcs.wcs.cdelt = [-1/3600, 2/3600]
    header = wcs.to_header()
    header.update(BMAJ=4/3600, BMIN=2/3600, BPA=0.)
    beam, info = validator.restoring_beam_kernel((65,65), header)
    assert info["major_fwhm_arcsec"] == pytest.approx(4.)
    assert beam[33,32] == pytest.approx(.5, rel=.01)  # north: major axis, 2 arcsec
    assert beam[32,33] == pytest.approx(.5, rel=.01)  # west: minor axis, 1 arcsec


def make_test_double(tmp_path, *, shift_image_deg=0., shift_clean_group_deg=0.,
                     clean_beam_scale=1., zero_psf=False, beam=True, reported_threshold_met=True):
    """Write synthetic FITS test doubles. These are not CASA/visibility products."""
    validator = load_validator()
    fixtures, runs = tmp_path / "fixtures", tmp_path / "runs"
    bundle, run = fixtures / "point_noiseless", runs / "point_noiseless"
    bundle.mkdir(parents=True)
    run.mkdir(parents=True)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [180., 30.]
    wcs.wcs.crpix = [17., 17.]
    wcs.wcs.cdelt = [-2.5/3600, 2.5/3600]
    base_header = wcs.to_header()
    base_header["BUNIT"] = "Jy/pixel"
    sky = np.zeros((32, 32), dtype=np.float32)
    sky[16, 16] = .01
    fits.PrimaryHDU(sky, header=base_header).writeto(bundle / "ch00.model.fits")
    (bundle / "casa_inputs.json").write_text(json.dumps({"channels": [{"frequency_ghz": 1.44, "bandwidth_mhz": 8.}]}))
    radius2 = (np.indices((32, 32))[0]-16)**2 + (np.indices((32, 32))[1]-16)**2
    fwhm_pixel = 4.
    psf = np.exp(-4*np.log(2)*radius2/fwhm_pixel**2).astype(np.float32)
    if zero_psf:
        psf[:] = 0.
    image = .01*psf
    pb = np.ones((32, 32), dtype=np.float32)
    image_header = base_header.copy()
    image_header["BUNIT"] = "Jy/beam"
    if shift_image_deg:
        image_header["CRVAL1"] += shift_image_deg
    if beam:
        image_header["BMAJ"] = fwhm_pixel*2.5/3600
        image_header["BMIN"] = fwhm_pixel*2.5/3600
        image_header["BPA"] = 0.
    files = {}
    clean_files = {}
    for role, data in (("image", image), ("psf", psf), ("pb", pb), ("residual", image)):
        path = run / (role+".fits")
        header = image_header if role == "image" else base_header.copy()
        if role == "psf":
            header["BUNIT"] = "1"
        elif role == "pb":
            header["BUNIT"] = "1"
        else:
            header["BUNIT"] = "Jy/beam"
        fits.PrimaryHDU(data, header=header).writeto(path)
        files[role] = {"path": path.name, "sha256": validator.sha256(path)}
    clean_data = {"image": image, "model": sky, "residual": np.zeros_like(image),
                  "psf": psf, "pb": pb}
    for role, data in clean_data.items():
        path = run / ("clean_"+role+".fits")
        header = image_header.copy() if role == "image" else base_header.copy()
        if shift_clean_group_deg:
            header["CRVAL1"] += shift_clean_group_deg
        if role == "image" and beam:
            header["BMAJ"] *= clean_beam_scale
            header["BMIN"] *= clean_beam_scale
            header["BPA"] = 0.
        header["BUNIT"] = "Jy/pixel" if role == "model" else ("1" if role in {"pb", "psf"} else "Jy/beam")
        fits.PrimaryHDU(data, header=header).writeto(path)
        clean_files[role] = {"path": path.name, "sha256": validator.sha256(path)}
    (run / "casa_products.json").write_text(json.dumps({
        "status": "completed_unvalidated", "casa_version": "TEST-DOUBLE-NOT-CASA",
        "test_double": True,
        "products": [{"channel": "ch00", "measurement_set": "TEST-DOUBLE-NOT-MS",
                      "discovered_measurement_sets": ["TEST-DOUBLE-NOT-MS"],
                      "measurement_set_audit": {"rows": 8, "test_double_only": True},
                      "products": files, "dirty_products": files, "clean_products": clean_files,
                      "clean_imaging": {"niter": 5000, "threshold": "0.001Jy"},
                      "clean_threshold_met_by_final_residual": reported_threshold_met,
                      "clean_final_residual_peak_jy_beam": 0.0}],
    }))
    (fixtures / "cases.json").write_text(json.dumps({"cases": [{
        "bundle": "point_noiseless", "input_flux_jy": .01,
        "fwhm_arcsec": 0., "thermalnoise": "", "test_double_only": True,
    }], "test_double_only": True}))
    return fixtures, runs


def run_test_report(monkeypatch, tmp_path, *, shift_image_deg=0., shift_clean_group_deg=0.,
                    clean_beam_scale=1., zero_psf=False, beam=True, reported_threshold_met=True):
    validator = load_validator()
    fixtures, runs = make_test_double(tmp_path, shift_image_deg=shift_image_deg,
                                      shift_clean_group_deg=shift_clean_group_deg,
                                      clean_beam_scale=clean_beam_scale,
                                      zero_psf=zero_psf, beam=beam, reported_threshold_met=reported_threshold_met)
    report = tmp_path / "report.json"
    config_path = tmp_path / "fast_config.json"
    config = json.loads((Path(__file__).parents[1] / "configs" / "compact_proxy_v2.json").read_text())
    config["instrument"]["min_baseline_m"] = 0.
    config["instrument"]["max_short_spacing_ratio"] = None
    config["instrument"]["max_baseline_m"] = 3400.
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["validate_casa_run.py", "--fixtures", str(fixtures),
                                      "--runs", str(runs), "--fast-config", str(config_path),
                                      "--out", str(report)])
    if shift_image_deg or shift_clean_group_deg or clean_beam_scale != 1. or zero_psf or not beam:
        with pytest.raises(SystemExit, match="1"):
            validator.main()
    else:
        validator.main()
    # strict parser rejects NaN/Infinity constants, and a standard encoder can
    # round-trip the loaded report. The explicit marker prevents CASA claims.
    def reject_constant(token):
        raise AssertionError(f"non-standard JSON constant: {token}")
    data = json.loads(report.read_text(), parse_constant=reject_constant)
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert data["test_double_only"] is True
    assert "not CASA validation" in data["purpose"]
    assert all(x["casa_version"] == "TEST-DOUBLE-NOT-CASA" for x in data["cases"] if x["status"] == "pass")
    return data


def test_full_acceptance_report_passes_for_aligned_test_double(monkeypatch, tmp_path):
    report = run_test_report(monkeypatch, tmp_path)
    assert report["status"] == "test_double_only"
    assert report["scientific_validation_pass"] is False
    result = report["cases"][0]
    assert result["status"] == "pass"
    assert result["point_centroid_error_pixels"] == pytest.approx(0.)
    assert result["validation_gates"]["psf_core_matches_required_header_beam_within_10_percent"]
    assert result["matched_beam_apparent_sky_vs_reconstructed_model_relative_l2"] < 1e-6


def test_validator_measures_exported_threshold_check_instead_of_trusting_report_flag(monkeypatch, tmp_path):
    report = run_test_report(monkeypatch, tmp_path, reported_threshold_met=False)
    result = report["cases"][0]
    assert result["clean_threshold_met_by_final_residual"] is False  # preserved native statement
    assert result["clean_stop_diagnostics"]["exported_residual_threshold_met"] is True
    assert result["reported_vs_measured_threshold_flag_agrees"] is False
    assert result["clean_stop_diagnostics"]["task_global_stopcode"] is None
    assert result["clean_stop_diagnostics"]["scientific_convergence_verified"] is False


def test_shifted_celestial_wcs_fails_full_acceptance_report(monkeypatch, tmp_path):
    report = run_test_report(monkeypatch, tmp_path, shift_image_deg=1.)
    assert report["status"] == "fail"
    assert "Celestial WCS grids are misaligned" in report["cases"][0]["failure"]


def test_whole_clean_group_shifted_against_dirty_and_input_fails(monkeypatch, tmp_path):
    report = run_test_report(monkeypatch, tmp_path, shift_clean_group_deg=1.)
    assert report["status"] == "fail"
    assert "Celestial WCS grids are misaligned" in report["cases"][0]["failure"]


def test_declared_clean_restoring_beam_controls_closure(monkeypatch, tmp_path):
    report = run_test_report(monkeypatch, tmp_path, clean_beam_scale=1.5)
    assert report["status"] == "fail"
    result = report["cases"][0]
    assert result["clean_restoration_closure_relative_l2"] > .02
    assert result["clean_restoring_beam_vs_psf_fit"]["major_relative_difference"] == pytest.approx(.5)


@pytest.mark.parametrize("kwargs,match", [
    ({"zero_psf": True}, "Invalid PSF normalization"),
    ({"beam": False}, "requires finite BMAJ, BMIN and BPA"),
])
def test_invalid_psf_or_missing_beam_fails_full_report(monkeypatch, tmp_path, kwargs, match):
    report = run_test_report(monkeypatch, tmp_path, **kwargs)
    assert report["status"] == "fail"
    assert match in report["cases"][0]["failure"]


@pytest.mark.parametrize("stage,role", [("dirty", "residual"), ("clean", "residual"),
                                         ("dirty", "psf"), ("clean", "pb")])
def test_blank_casa_units_require_specific_producer_and_preserve_raw_header(stage, role):
    validator = load_validator()
    header = fits.Header({"BUNIT": ""})
    kwargs = dict(stage=stage, imaging={"niter": 0 if stage == "dirty" else 5000},
                  casa_version="6.7.4.8", producer="casatasks.tclean", actual_casa=True,
                  dirty_residual_matches_image=True)
    result = validator.effective_product_unit(role, header["BUNIT"], **kwargs)
    assert header["BUNIT"] == result["raw_fits_bunit"] == ""
    assert result["effective_unit"] == ("Jy/dirty beam" if role == "residual" else
                                        "dimensionless normalized PSF response" if role == "psf" else
                                        "dimensionless primary-beam response")
    for change in ({"actual_casa": False}, {"casa_version": "unknown"},
                   {"producer": "unsupported.renderer"}):
        with pytest.raises(ValueError):
            validator.effective_product_unit(role, "", **(kwargs | change))


@pytest.mark.parametrize("role,bunit", [("residual", "Jy/pixel"), ("residual", "Jy"),
                                        ("psf", "Jy/beam"), ("pb", "Jy/beam")])
def test_explicit_conflicting_units_are_never_overridden_by_casa_provenance(role, bunit):
    with pytest.raises(ValueError):
        load_validator().effective_product_unit(
            role, bunit, stage="clean", imaging={"niter": 5000}, casa_version="6.7.4.8",
            producer="casatasks.tclean", actual_casa=True)


def test_blank_dirty_residual_requires_numerical_normalization_evidence():
    with pytest.raises(ValueError, match="exact pixel equality"):
        load_validator().effective_product_unit(
            "residual", "", stage="dirty", imaging={"niter": 0}, casa_version="6.7.4.8",
            producer="casatasks.tclean", actual_casa=True, dirty_residual_matches_image=False)


def test_runtime_evidence_checksums_bind_version_and_producer_to_case(tmp_path):
    validator = load_validator()
    products = {"casa_version": "unknown; preserve startup log",
                "runner_sha256": "00533bd4ddeef2119fde3357d4f10658715546e838ec6783c0f5f48f97657e06"}
    report = tmp_path / "casa_products.json"
    report.write_text(json.dumps(products))
    evidence_path = tmp_path / "runtime.json"
    evidence_path.write_text("synthetic unit-test provenance; not a CASA execution")
    evidence = {"actual_CASA": True, "test_double": False,
                "versions": {"casatasks": "6.7.4.8", "casatools": "6.7.4.8"},
                "runner_sha256": products["runner_sha256"], "_path": str(evidence_path),
                "cases": {"point_noisy": {"status": "completed_unvalidated",
                          "report_sha256": validator.sha256(report),
                          "selected_ms": "/old/host/point_noisy/ch00/example.noisy.ms"}}}
    channel = {"measurement_set": "ch00/example.noisy.ms"}
    context = validator.verified_runtime_context(evidence, products, report, "point_noisy", channel)
    assert context["casa_version"] == "6.7.4.8"
    assert context["evidence"]["runtime_provenance_sha256"] == validator.sha256(evidence_path)
    assert products["casa_version"].startswith("unknown")  # native report is untouched
    for change in ({"actual_CASA": False}, {"runner_sha256": "different"},
                   {"cases": {}}, {"versions": {"casatasks": "unsupported"}}):
        with pytest.raises(ValueError):
            validator.verified_runtime_context(evidence | change, products, report, "point_noisy", channel)
    for change in ({"imaging_producer": "unsupported.renderer"},
                   {"casa_version": "6.2.0"}, {"casa_packages": {"casatools": "6.2.0"}}):
        conflicting = products | change
        report.write_text(json.dumps(conflicting))
        evidence["cases"]["point_noisy"]["report_sha256"] = validator.sha256(report)
        with pytest.raises(ValueError, match="unsupported|conflict"):
            validator.verified_runtime_context(evidence, conflicting, report, "point_noisy", channel)
    report.write_text(json.dumps(products))
    evidence["cases"]["point_noisy"]["report_sha256"] = validator.sha256(report)
    report.write_text(report.read_text()+" ")
    with pytest.raises(ValueError, match="checksum-anchor"):
        validator.verified_runtime_context(evidence, products, report, "point_noisy", channel)


def saved_ms_contract_fixture(tmp_path):
    """Synthetic path/provenance contracts only; not a CASA execution."""
    validator = load_validator()
    native = tmp_path/"casa_products.json"
    runtime_path = tmp_path/"runtime.json"
    ms = "/external/observations/calibrated.ms"
    products = {"product_origin": "measurement_only_reconstruction_of_existing_ms",
                "input_measurement_set": ms, "casa_version": "6.7.4.8",
                "casa_packages": {"casatasks": "6.7.4.8", "casatools": "6.7.4.8"},
                "imaging_producer": "casatasks.tclean", "runner_sha256": "synthetic-contract-source"}
    channel = {"measurement_set": ms}
    evidence = {"actual_CASA": True, "test_double": False,
                "versions": {"casatasks": "6.7.4.8", "casatools": "6.7.4.8"},
                "runner_sha256": products["runner_sha256"],
                "cases": {"extended90_noisy": {"status": "completed_unvalidated", "selected_ms": ms}},
                "_path": str(runtime_path)}
    def bind():
        native.write_text(json.dumps(products))
        evidence["cases"]["extended90_noisy"]["report_sha256"] = validator.sha256(native)
        runtime_path.write_text(json.dumps(evidence))
    bind()
    return validator, products, channel, evidence, native, bind


@pytest.mark.parametrize("paths", [
    ("/external/observations/calibrated.ms", "/external/observations/./calibrated.ms/",
     "/external//observations/calibrated.ms"),
    ("C:/observations/calibrated.ms", r"C:\observations\calibrated.ms", "C:/observations/./calibrated.ms"),
    (r"\\server\share\calibrated.ms", "//server/share/calibrated.ms", r"\\server\share\.\calibrated.ms"),
])
def test_saved_ms_origin_requires_normalized_absolute_three_way_agreement(tmp_path, paths):
    v, products, channel, evidence, native, bind = saved_ms_contract_fixture(tmp_path)
    products["input_measurement_set"], channel["measurement_set"] = paths[:2]
    evidence["cases"]["extended90_noisy"]["selected_ms"] = paths[2]
    bind()
    before = native.read_bytes()
    context = v.verified_runtime_context(evidence, products, native, "extended90_noisy", channel)
    assert context["actual_casa"] is True
    assert "exact normalized absolute agreement" in context["evidence"]["measurement_set_path_contract"]
    assert native.read_bytes() == before


@pytest.mark.parametrize("slot", ["input", "channel", "external"])
def test_saved_ms_origin_rejects_different_absolute_paths_even_with_same_suffix(tmp_path, slot):
    v, products, channel, evidence, native, bind = saved_ms_contract_fixture(tmp_path)
    different = "/different_host_root/observations/calibrated.ms"
    if slot == "input": products["input_measurement_set"] = different
    if slot == "channel": channel["measurement_set"] = different
    if slot == "external": evidence["cases"]["extended90_noisy"]["selected_ms"] = different
    bind()
    with pytest.raises(ValueError, match="paths disagree"):
        v.verified_runtime_context(evidence, products, native, "extended90_noisy", channel)


@pytest.mark.parametrize("slot", ["input", "channel", "external"])
@pytest.mark.parametrize("bad", ["", None, 17, "relative/data.ms", "C:drive_relative.ms",
                                "/external/../data.ms", "file:///external/data.ms", "/",
                                "/external/data\x00.ms", r"\\?\C:\data.ms"])
def test_saved_ms_origin_rejects_missing_relative_traversal_or_malformed_path(tmp_path, slot, bad):
    v, products, channel, evidence, native, bind = saved_ms_contract_fixture(tmp_path)
    if slot == "input": products["input_measurement_set"] = bad
    if slot == "channel": channel["measurement_set"] = bad
    if slot == "external": evidence["cases"]["extended90_noisy"]["selected_ms"] = bad
    bind()
    with pytest.raises(ValueError, match="[Pp]ath|traversal"):
        v.verified_runtime_context(evidence, products, native, "extended90_noisy", channel)


@pytest.mark.parametrize("defect", ["unknown_origin", "null_origin", "native_package",
                                   "external_package", "producer", "missing_producer",
                                   "runner", "report_checksum"])
def test_saved_ms_path_support_preserves_producer_package_and_checksum_guards(tmp_path, defect):
    v, products, channel, evidence, native, bind = saved_ms_contract_fixture(tmp_path)
    if defect == "unknown_origin": products["product_origin"] = "arbitrary_origin"
    if defect == "null_origin": products["product_origin"] = None
    if defect == "native_package": products["casa_packages"]["casatools"] = "6.2.0"
    if defect == "external_package": evidence["versions"]["casatasks"] = "6.2.0"
    if defect == "producer": products["imaging_producer"] = "different.renderer"
    if defect == "missing_producer": products.pop("imaging_producer")
    if defect == "runner": evidence["runner_sha256"] = "different-source"
    bind()
    if defect == "report_checksum": native.write_text(native.read_text()+" ")
    with pytest.raises(ValueError):
        v.verified_runtime_context(evidence, products, native, "extended90_noisy", channel)


def test_legacy_relative_origin_still_requires_case_suffix_and_rejects_absolute_channel(tmp_path):
    v, products, channel, evidence, native, bind = saved_ms_contract_fixture(tmp_path)
    products.pop("product_origin")
    channel["measurement_set"] = "ch00/source.ms"
    evidence["cases"]["extended90_noisy"]["selected_ms"] = "/old/extended90_noisy/ch00/source.ms"
    bind()
    context = v.verified_runtime_context(evidence, products, native, "extended90_noisy", channel)
    assert "legacy relative" in context["evidence"]["measurement_set_path_contract"]
    evidence["cases"]["extended90_noisy"]["selected_ms"] = "/old/other_case/ch00/source.ms"
    with pytest.raises(ValueError, match="does not match"):
        v.verified_runtime_context(evidence, products, native, "extended90_noisy", channel)
    channel["measurement_set"] = "/old/extended90_noisy/ch00/source.ms"
    with pytest.raises(ValueError, match="must be relative"):
        v.verified_runtime_context(evidence, products, native, "extended90_noisy", channel)


def test_saved_ms_origin_requires_external_runtime_even_if_native_units_are_explicit(tmp_path):
    v, products, channel, evidence, native, bind = saved_ms_contract_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires independently checksum-bound runtime"):
        v.verified_runtime_context(None, products, native, "extended90_noisy", channel)
