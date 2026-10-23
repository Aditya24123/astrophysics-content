"""Observable policy and stop-report contracts; no CASA execution or truth fit."""
import json

import numpy as np
import pytest

from hii_sim.casa_reconstruction_policy import multiscale_policy, configured_multiscale_policy
from hii_sim.casa_simulate import clean_stop_diagnostics, clean_summary_fields


def example_policy(**changes):
    return multiscale_policy(**(dict(beam_major_arcsec=16.65, cell_arcsec=2.5,
                                     npix=256, min_usable_uv_lambda=374.,
                                     observed_border_rms_jy_beam=0.0002) | changes))


def test_scales_and_threshold_come_only_from_measured_beam_grid_uv_and_noise():
    policy = example_policy()
    settings = policy["tclean_overrides"]
    assert settings["scales"] == [0, 7, 20, 60]
    assert settings["threshold"] == "0.0006Jy"
    assert settings["niter"] == 5000 and settings["nsigma"] == 0.
    assert settings["smallscalebias"] == 0. and settings["gain"] == .1
    assert settings["fullsummary"] and not settings["restart"]
    assert settings["savemodel"] == "none" and settings["mask"] == ""
    assert "startmodel" not in settings
    with pytest.raises(TypeError):
        example_policy(true_source_fwhm=90.)


def test_scale_caps_respect_changed_field_and_shortest_usable_spacing():
    assert example_policy(npix=64)["tclean_overrides"]["scales"] == [0, 7]
    policy = example_policy(min_usable_uv_lambda=3000.)
    assert policy["tclean_overrides"]["scales"] == [0, 7]
    assert policy["scale_diameter_cap_pixels"] < 20.
    assert example_policy(beam_major_arcsec=10.)["tclean_overrides"]["scales"] == [0, 4, 12, 36]


@pytest.mark.parametrize("changes", [{"observed_border_rms_jy_beam": 0.},
                                      {"min_usable_uv_lambda": -1.},
                                      {"beam_major_arcsec": float("nan")},
                                      {"cell_arcsec": 0.}, {"npix": True},
                                      {"min_usable_uv_lambda": 1.e8}])
def test_policy_rejects_unsupported_or_unconstrained_inputs(changes):
    with pytest.raises(ValueError):
        example_policy(**changes)


def test_casa_summary_preserves_nested_planes_and_numpy_stop_history_in_strict_json():
    summary = {"stopcode": np.int32(2), "stopDescription": "Reached cyclethreshold",
               "iterdone": np.int64(9394), "summarymajor": np.array([0, 200, 9394]),
               "summaryminor": {0: {0: {0: {"peakRes": np.array([1.e-3, 6.6e-8]),
                                           "stopCode": np.array([1, 2]),
                                           "cycleThresh": [np.float64(np.nan)]}}}}}
    result = clean_summary_fields(summary)
    decoded = json.loads(json.dumps(result, allow_nan=False))
    assert decoded["stopcode"] == 2 and decoded["iterdone"] == 9394
    assert decoded["summaryminor"]["0"]["0"]["0"]["stopCode"] == [1, 2]
    assert decoded["summaryminor"]["0"]["0"]["0"]["cycleThresh"] == [{"nonfinite_float": "nan"}]


def test_task_threshold_exit_is_distinct_from_strict_exported_residual_failure():
    threshold, peak = 6.678947247564793e-8, 6.728051005211455e-8
    summary = {"stopcode": 2, "stopDescription": "Reached cyclethreshold", "iterdone": 9394}
    result = clean_stop_diagnostics(summary, 20000, threshold, peak)
    assert result["task_global_stopcode"] == 2
    assert result["state"] == "task_reported_stop"
    assert result["exported_residual_threshold_met"] is False
    assert result["exported_peak_over_threshold"] == pytest.approx(peak/threshold)
    assert result["exported_residual_minus_threshold_jy"] > 0.
    assert result["scientific_convergence_verified"] is False
    # Historical summary lost the stopcode: don't invent one or relax threshold.
    result = clean_stop_diagnostics({"iterdone": 9394}, 20000, threshold, peak)
    assert result["state"] == "unresolved_task_stop_reason"
    assert result["stop_reason_auditable"] is False


def test_minor_stopcodes_never_become_global_exit_or_convergence_claims():
    result = clean_stop_diagnostics({"summaryminor": {"stopCode": [2]}}, 5000, 1.e-4, 2.e-4)
    assert result["task_global_stopcode"] is None
    assert result["state"] == "unresolved_task_stop_reason"
    below = clean_stop_diagnostics(None, 5000, 1.e-4, 1.e-4)
    assert below["exported_residual_threshold_met"] is True
    assert below["scientific_convergence_verified"] is False
    cap = clean_stop_diagnostics({"iterdone": 5000}, 5000, 0., 1.e-8)
    assert cap["iteration_limit_reached"] and cap["state"] == "iteration_limit"
    assert cap["exported_residual_threshold_met"] is False


def test_configurable_coarse_wrapper_preserves_frozen_v1_and_reports_multiplier():
    frozen = example_policy()
    configured = configured_multiscale_policy(**frozen["inputs"], noise_multiplier=2.25)
    assert configured["tclean_overrides"]["scales"] == frozen["tclean_overrides"]["scales"]
    assert configured["tclean_overrides"]["threshold"] == "0.00045Jy"
    assert configured["noise_multiplier"] == 2.25
    assert example_policy() == frozen
    for invalid in (True, 0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="noise_multiplier"):
            configured_multiscale_policy(**frozen["inputs"], noise_multiplier=invalid)
