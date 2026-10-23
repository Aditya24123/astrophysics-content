"""Control-flow mocks only: these tests do not simulate CASA visibilities."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from astropy.io import fits
from astropy.coordinates import Angle, FK5, SkyCoord
import astropy.units as u

from hii_sim.casa_bridge import casa_direction_j2000, prepare_casa_inputs
from hii_sim.casa_simulate import (audit_measurement_set, discover_measurement_sets,
                                   resolve_input, run, run_simobserve_in_output, casa_runtime_versions,
                                   reimage_saved_ms, reconstruct_measurements)
from hii_sim.io import write_image
from hii_sim.schema import Grid, Instrument, Sky


def bundle(tmp_path, noise="tsys-atm"):
    grid = Grid(npix=32, pixel_arcsec=2.5)
    sky = np.zeros((1, 32, 32)); sky[0, 16, 16] = .01 / grid.omega_sr
    truth = tmp_path / "truth.fits"
    write_image(truth, sky, grid, Sky(), Instrument(frequencies_ghz=(1.44,), bandwidths_mhz=(8.,)),
                "Jy/sr", "truth")
    with fits.open(truth, mode="update") as hdus:
        hdus[1].header["EM0"] = 123456  # Must never be copied to the CASA sky header.
    out = tmp_path / "input"
    manifest = prepare_casa_inputs(truth, out, thermalnoise=noise)
    return out, manifest


@pytest.mark.parametrize("declination", [29.999997472, -29.999997472])
def test_casa_direction_uses_explicit_sexagesimal_units_and_preserves_coordinates(declination):
    original = SkyCoord(ra=180.00000317*u.deg, dec=declination*u.deg,
                        frame=FK5(equinox="J2000"))
    text = casa_direction_j2000(original)
    frame, ra_token, dec_token = text.split()
    assert frame == "J2000"
    assert all(letter in ra_token for letter in "hms")
    assert all(letter in dec_token for letter in "dms")
    assert ":" not in ra_token + dec_token
    assert dec_token[0] in "+-"
    parsed_ra = Angle(ra_token, unit=u.hourangle).to_value(u.deg)
    parsed_dec = Angle(dec_token, unit=u.deg).to_value(u.deg)
    assert parsed_ra == pytest.approx(original.ra.deg, abs=1e-8)
    assert parsed_dec == pytest.approx(original.dec.deg, abs=1e-8)


class MockCasa:
    """Create placeholder filesystem products with deliberately changed names."""
    __version__ = "TEST-DOUBLE-NOT-CASA"

    def __init__(self, noisy=True, raise_on_simobserve=False):
        self.noisy = noisy
        self.raise_on_simobserve = raise_on_simobserve
        self.imaging = []
        self.simobserve_cwd = []

    def simobserve(self, **kwargs):
        self.simobserve_cwd.append(Path.cwd())
        assert Path(kwargs["project"]).name == kwargs["project"]
        assert "/" not in kwargs["project"] and "\\" not in kwargs["project"]
        if self.raise_on_simobserve:
            Path(kwargs["project"]).mkdir()
            raise RuntimeError("synthetic simobserve failure")
        project = Path(kwargs["project"])
        products = ["unexpected-array-name.ms"]
        if self.noisy:
            products.append("unexpected-array-name.noisy.ms")
        for name in products:
            path = project / name
            path.mkdir(parents=True)
            (path / "table.dat").write_text("MOCK ONLY")
        assert kwargs["setpointings"] is False
        assert Path(kwargs["ptgfile"]).is_file()
        assert Path(kwargs["skymodel"]).is_file()

    def tclean(self, **kwargs):
        self.imaging.append(kwargs)
        for suffix in ("image", "psf", "pb", "residual"):
            Path(kwargs["imagename"] + "." + suffix).mkdir()
        if kwargs.get("niter", 0) > 0:
            Path(kwargs["imagename"] + ".model").mkdir()
            return {"iterdone": 3}
        return {"iterdone": 0}

    def exportfits(self, **kwargs):
        name = kwargs["imagename"].rsplit(".", 1)[-1]
        y, x = np.indices((32, 32))
        if name == "psf":
            data = np.exp(-((x-16)**2+(y-16)**2)/8.)
        elif name == "pb":
            data = np.ones((32, 32))
        elif name == "model":
            data = np.zeros((32, 32)); data[16, 16] = .01
        else:
            scale = 1e-6 if ".clean." in kwargs["fitsimage"] else 1e-4
            data = np.random.default_rng(2).normal(0, scale, (32, 32))
        header = fits.Header({"BUNIT": "Jy/pixel" if name == "model" else "Jy/beam",
                              "CTYPE1": "RA---SIN", "CTYPE2": "DEC--SIN",
                              "CRPIX1": 16.5, "CRPIX2": 16.5, "CRVAL1": 10., "CRVAL2": 20.,
                              "CDELT1": -2.5/3600, "CDELT2": 2.5/3600,
                              "BMAJ": 8./3600, "BMIN": 8./3600})
        fits.PrimaryHDU(data, header).writeto(kwargs["fitsimage"], checksum=True)


class MockTableTool:
    def open(self, path):
        self.path = Path(path)
        return True

    def colnames(self):
        if self.path.name == "SPECTRAL_WINDOW":
            return ["CHAN_FREQ", "CHAN_WIDTH"]
        if self.path.name == "ANTENNA":
            return ["POSITION"]
        return ["DATA", "FLAG", "FLAG_ROW", "UVW", "WEIGHT", "TIME", "ANTENNA1", "ANTENNA2"]

    def getcol(self, name):
        if name == "CHAN_FREQ": return np.array([[1.44e9]])
        if name == "CHAN_WIDTH": return np.array([[8e6]])
        if name == "POSITION": return np.zeros((3, 4))
        if name == "DATA": return np.ones((2, 1, 8), complex)*.01
        if name == "FLAG": return np.zeros((2, 1, 8), bool)
        if name == "FLAG_ROW": return np.zeros(8, bool)
        if name == "UVW": return np.array([[40, 50, 60, 70, 80, 90, 100, 110],
                                           [0]*8, [0]*8], float)
        if name == "WEIGHT": return np.ones((2, 8))
        if name == "TIME": return np.arange(8, dtype=float)
        if name == "ANTENNA1": return np.array([0,0,0,1,1,1,2,2])
        if name == "ANTENNA2": return np.array([1,2,3,2,3,0,3,0])
        raise KeyError(name)

    def close(self):
        pass


def table_factory():
    return MockTableTool()


def test_bundle_moves_without_original_absolute_paths(tmp_path):
    prepared, manifest = bundle(tmp_path)
    relocated = tmp_path / "relocated"
    shutil.move(str(prepared), relocated)
    channel = manifest["channels"][0]
    model = resolve_input(relocated, channel["fits"])
    with fits.open(model) as hdus:
        assert hdus[0].header["BUNIT"] == "Jy/pixel"
        assert "EM0" not in hdus[0].header
        assert hdus[0].data.sum() == pytest.approx(.01, rel=1e-6)
    assert not Path(channel["fits"]).is_absolute()
    assert "sky_input" not in manifest
    frame, ra_token, dec_token = manifest["direction"].split()
    assert frame == "J2000" and all(c in ra_token for c in "hms")
    assert all(c in dec_token for c in "dms") and ":" not in ra_token + dec_token
    tasks = MockCasa()
    nested_output = tmp_path / "runs" / "point_noiseless" / "attempt_01"
    original_cwd = Path.cwd()
    report = run(relocated, nested_output, tasks=tasks, table_tool_factory=table_factory)
    assert tasks.imaging[0]["vis"].endswith("unexpected-array-name.noisy.ms")
    assert Path(tasks.imaging[0]["vis"]).parent == nested_output.resolve() / "ch00"
    assert tasks.simobserve_cwd == [nested_output.resolve()]
    assert Path.cwd() == original_cwd
    assert report["status"] == "completed_unvalidated"
    assert report["casa_version"] == "TEST-DOUBLE-NOT-CASA"
    assert report["products"][0]["measurement_set_audit"]["rows"] == 8
    assert len(tasks.imaging) == 2 and tasks.imaging[0]["niter"] == 0
    assert tasks.imaging[1]["fullsummary"] is True
    assert tasks.imaging[1]["threshold"].endswith("Jy")
    assert set(report["products"][0]["clean_products"]) == {"image", "model", "residual", "psf", "pb"}


def test_absent_noisy_product_fails_without_falling_back(tmp_path):
    prepared, _ = bundle(tmp_path)
    tasks = MockCasa(noisy=False)
    with pytest.raises(RuntimeError, match="exactly one noisy"):
        run(prepared, tmp_path / "run", tasks=tasks, table_tool_factory=table_factory)
    assert tasks.imaging == []
    assert json.loads((tmp_path / "run/casa_products.json").read_text())["status"] == "failed"


def test_noise_free_is_explicit_and_existing_outputs_preserved(tmp_path):
    prepared, _ = bundle(tmp_path, noise="")
    tasks = MockCasa(noisy=False)
    report = run(prepared, tmp_path / "run", tasks=tasks, table_tool_factory=table_factory)
    assert report["measurement_product"] == "noise-free"
    assert tasks.imaging[1]["threshold"] == "0Jy"
    assert report["products"][0]["clean_return_summary"] == {"iterdone": 3}
    with pytest.raises(FileExistsError):
        run(prepared, tmp_path / "run", tasks=tasks, table_tool_factory=table_factory)


def test_simobserve_failure_restores_cwd_and_retains_absolute_output_location(tmp_path):
    prepared, _ = bundle(tmp_path)
    output = tmp_path / "nested" / "failed_attempt"
    original = Path.cwd()
    tasks = MockCasa(raise_on_simobserve=True)
    with pytest.raises(RuntimeError, match="synthetic simobserve failure"):
        run(prepared, output, tasks=tasks, table_tool_factory=table_factory)
    assert Path.cwd() == original
    assert tasks.simobserve_cwd == [output.resolve()]
    report = json.loads((output / "casa_products.json").read_text())
    assert report["status"] == "failed"
    assert (output / "ch00").is_dir()


def test_simobserve_helper_rejects_path_project_names(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    tasks = MockCasa()
    for unsafe in ("nested/ch00", "nested\\ch00", "../ch00", ""):
        with pytest.raises(ValueError, match="simple basename"):
            run_simobserve_in_output(tasks, output, unsafe, skymodel="unused")


def test_checksum_mismatch_and_path_escape_rejected(tmp_path):
    prepared, manifest = bundle(tmp_path)
    (prepared / manifest["channels"][0]["fits"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run(prepared, tmp_path / "run", tasks=MockCasa())
    for bad in ("../truth.fits", "C:\\old\\model.fits", str((tmp_path / "truth.fits").resolve())):
        with pytest.raises(ValueError):
            resolve_input(prepared, bad)


def test_ambiguous_noisy_products_rejected(tmp_path):
    for name in ("first.noisy.ms", "second.noisy.ms"):
        product = tmp_path / name
        product.mkdir()
        (product / "table.dat").write_text("MOCK ONLY")
    with pytest.raises(RuntimeError, match="exactly one noisy"):
        discover_measurement_sets(tmp_path)


def test_ms_audit_uses_only_unflagged_positive_weight_samples():
    audit = audit_measurement_set(Path("mock.ms"), table_factory)
    assert audit["rows"] == 8
    assert audit["uvw_baseline_min_m"] == pytest.approx(40)
    assert audit["flag_fraction"] == 0
    assert audit["sample_flag_fraction"] == audit["row_flag_fraction"] == audit["effective_flag_fraction"] == 0
    assert audit["usable_sample_count"] == 16
    assert audit["weight_min_positive"] == 1
    assert audit["channel_frequency_hz"] == [1.44e9]


def test_ms_audit_rejects_all_rows_flagged_despite_clear_sample_flags():
    class AllRowFlags(MockTableTool):
        def getcol(self, name):
            if name == "FLAG_ROW": return np.ones(8, bool)
            return super().getcol(name)
    with pytest.raises(RuntimeError, match="no valid unflagged positive-weight data after FLAG and FLAG_ROW"):
        audit_measurement_set(Path("mock.ms"), AllRowFlags)


def test_ms_audit_partial_row_flags_exclude_samples_from_rms_and_usable_weights():
    class PartialRowFlags(MockTableTool):
        def getcol(self, name):
            if name == "FLAG_ROW": return np.array([True, True, False, False, False, False, False, False])
            if name == "DATA":
                data = super().getcol(name)
                data[:, :, :2] = 100.  # Row flags must exclude both correlations.
                data[0, 0, 2] = 200.   # Independent sample flag.
                data[0, 0, 3] = 300.   # Zero-weight sample.
                data[0, 0, 4] = np.nan # Non-finite unflagged sample.
                return data
            if name == "FLAG":
                flags = super().getcol(name)
                flags[0, 0, 0] = True  # Overlap must not double-count row/sample flags.
                flags[0, 0, 2] = True
                return flags
            if name == "WEIGHT":
                weights = super().getcol(name)
                weights[:, :2] = 1000.
                weights[0, 2] = 2000.
                weights[0, 3] = 0.
                weights[0, 4] = 3000.
                return weights
            return super().getcol(name)
    audit = audit_measurement_set(Path("mock.ms"), PartialRowFlags)
    assert audit["sample_flag_fraction"] == audit["flag_fraction"] == pytest.approx(2/16)
    assert audit["row_flag_fraction"] == pytest.approx(2/8)
    assert audit["effective_flag_fraction"] == pytest.approx(5/16)
    assert audit["valid_unflagged_fraction"] == pytest.approx(9/16)
    assert audit["usable_sample_count"] == 9
    assert audit["unflagged_amplitude_rms_jy"] == pytest.approx(.01)
    assert audit["weight_positive_fraction"] == pytest.approx(15/16)
    assert audit["weight_max"] == 3000.  # All-column diagnostic intentionally includes excluded data.
    assert audit["usable_weight_summary"]["correlation_row_count"] == 9
    assert audit["usable_weight_summary"]["min"] == audit["usable_weight_summary"]["max"] == 1.
    assert audit["uvw_baseline_min_m"] == 40.  # Full-table legacy geometry.
    assert audit["usable_uvw_baseline_min_positive_m"] == 60.  # Whole-row flags excluded.


@pytest.mark.parametrize("shape", [(), (1, 8), (8, 1), (7,), (9,)])
def test_ms_audit_rejects_malformed_row_flag_shape(shape):
    class MalformedRowFlags(MockTableTool):
        def getcol(self, name):
            if name == "FLAG_ROW": return np.zeros(shape, bool)
            return super().getcol(name)
    with pytest.raises(RuntimeError, match="FLAG_ROW shape"):
        audit_measurement_set(Path("mock.ms"), MalformedRowFlags)


def test_preview_offsets_and_fourier_units():
    spec = importlib.util.spec_from_file_location("preview", Path(__file__).parents[1]/"tools/preview_sample.py")
    preview = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preview)
    n, cell = 96, 2.5
    left, right, bottom, top = preview.sky_extent(n, cell)
    x = np.linspace(left, right, n, endpoint=False) + (right-left)/(2*n)
    y = np.linspace(bottom, top, n, endpoint=False) + (top-bottom)/(2*n)
    assert x[n//2] == 0 and y[n//2] == 0
    assert x[0] > 0 and x[-1] < 0  # East is left, matching FITS CDELT1 < 0.
    uu, vv = preview.uv_coordinates_klambda(n, cell)
    expected = 1/np.deg2rad(n*cell/3600)/1000
    assert uu[0, 1] == pytest.approx(-expected)
    assert vv[1, 0] == pytest.approx(expected)


def test_casa_distribution_versions_are_recorded_without_module_version_attribute(monkeypatch):
    from importlib import metadata
    monkeypatch.setattr(metadata, "version", lambda name: {"casatasks": "6.7.4.8", "casatools": "6.7.4.8"}[name])
    versions = casa_runtime_versions(object(), object())
    assert versions["casatasks"] == versions["casatools"] == "6.7.4.8"


def test_casa_version_fallback_does_not_invent_missing_tool_version(monkeypatch):
    from importlib import metadata
    def absent(name):
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(metadata, "version", absent)
    versions = casa_runtime_versions(MockCasa())
    assert versions["casatasks"] == "TEST-DOUBLE-NOT-CASA"
    assert versions["casatools"] == "unknown"


def reimage_config(path, manifest):
    config = {"schema_version": "1.0",
              "imaging": {"imsize": manifest["npix"], "cell": manifest["cell"], "phasecenter": manifest["direction"]},
              "reconstruction": {"policy": "multiscale", "noise_multiplier": 3., "noisy": True}}
    path.write_text(json.dumps(config))
    return config


def test_runner_and_saved_ms_reconstruction_share_numerical_controls_without_truth_inputs(tmp_path):
    prepared, manifest = bundle(tmp_path)
    simulated = run(prepared, tmp_path/"simulated", tasks=MockCasa(), table_tool_factory=table_factory,
                    reconstruction_policy="multiscale")
    config = tmp_path/"observing.json"
    reimage_config(config, manifest)
    ms = tmp_path/"simulated"/simulated["products"][0]["measurement_set"]
    original = (ms/"table.dat").read_bytes()
    # Remove access to the simulation bundle before reimaging observations.
    relocated_truth = tmp_path/"unavailable_truth"
    shutil.move(str(prepared), relocated_truth)
    class NoSimulation(MockCasa):
        def simobserve(self, **kwargs):
            raise AssertionError("Saved-MS reconstruction must not simulate")
    tasks = NoSimulation()
    reimaged = reimage_saved_ms(ms, config, tmp_path/"reimaged", tasks=tasks, table_tool_factory=table_factory)
    left, right = simulated["products"][0], reimaged["products"][0]
    assert left["dirty_imaging"] == right["dirty_imaging"]
    assert left["clean_imaging"] == right["clean_imaging"]
    assert left["configured_observable_policy"] == right["configured_observable_policy"]
    assert right["clean_imaging"]["scales"] == [0, 3]  # Small measured field clips 3b/9b.
    assert right["threshold_policy"] == left["threshold_policy"]
    assert right["threshold_policy"]["calibrated_thermal_sigma"] is False
    for stage in ("dirty_products", "clean_products"):
        for role in left[stage]:
            with fits.open(tmp_path/"simulated"/left[stage][role]["path"]) as a, fits.open(tmp_path/"reimaged"/right[stage][role]["path"]) as b:
                np.testing.assert_array_equal(a[0].data, b[0].data)
    assert len(tasks.imaging) == 2
    assert reimaged["input_ms_unchanged"] is True
    assert reimaged["input_ms_content_sha256_before"] == reimaged["input_ms_content_sha256_after"]
    assert (ms/"table.dat").read_bytes() == original
    report_bytes = (tmp_path/"reimaged/casa_products.json").read_bytes()
    with pytest.raises(FileExistsError):
        reimage_saved_ms(ms, config, tmp_path/"reimaged", tasks=tasks, table_tool_factory=table_factory)
    assert (tmp_path/"reimaged/casa_products.json").read_bytes() == report_bytes
    with pytest.raises(FileExistsError):
        reconstruct_measurements(ms, tmp_path/"reimaged", "ch00", json.loads(config.read_text())["imaging"], tasks=tasks,
                                 ms_audit=right["measurement_set_audit"])
    with pytest.raises(ValueError, match="outside the input Measurement Set"):
        reconstruct_measurements(ms, ms, "ch00", json.loads(config.read_text())["imaging"], tasks=tasks,
                                 ms_audit=right["measurement_set_audit"])


def test_saved_ms_multiplier_override_is_exact_in_used_settings_task_and_report(tmp_path):
    prepared, manifest = bundle(tmp_path)
    simulated = run(prepared, tmp_path/"sim", tasks=MockCasa(), table_tool_factory=table_factory)
    ms = tmp_path/"sim"/simulated["products"][0]["measurement_set"]
    config = tmp_path/"observing.json";reimage_config(config, manifest)
    tasks = MockCasa()
    report = reimage_saved_ms(ms, config, tmp_path/"new", tasks=tasks, table_tool_factory=table_factory,
                              noise_multiplier=2.25)
    product = report["products"][0]
    assert report["noise_multiplier"] == product["threshold_policy"]["noise_multiplier"] == 2.25
    expected = "%.12gJy" % (2.25*product["observed_noise_estimate"]["rms_jy_per_beam"])
    assert tasks.imaging[1]["threshold"] == expected == product["threshold_policy"]["serialized_threshold"]
    assert product["threshold_policy"]["task_threshold_jy"] == float(expected[:-2])
    used = json.loads((tmp_path/"new/observing_settings.used.json").read_text())
    assert used["reconstruction"]["noise_multiplier"] == 2.25


@pytest.mark.parametrize("key,value", [("startmodel", "truth.fits"), ("mask", "truth.mask"), ("true_em", 1.e6)])
def test_saved_ms_settings_reject_source_and_truth_controls_before_any_task(tmp_path, key, value):
    config = tmp_path/"bad.json"
    config.write_text(json.dumps({"schema_version":"1.0", "imaging":{"imsize":32,"cell":"2.5arcsec","phasecenter":"J2000 0h 0d",key:value}}))
    tasks = MockCasa()
    with pytest.raises(ValueError, match="prohibited"):
        reimage_saved_ms(tmp_path/"nonexistent.ms", config, tmp_path/"out", tasks=tasks, table_tool_factory=table_factory)
    assert not tasks.imaging and not (tmp_path/"out").exists()


def test_saved_ms_failure_retains_report_and_dirty_products(tmp_path):
    prepared, manifest = bundle(tmp_path)
    sim = run(prepared, tmp_path/"sim", tasks=MockCasa(), table_tool_factory=table_factory)
    ms = tmp_path/"sim"/sim["products"][0]["measurement_set"]
    config = tmp_path/"observing.json";reimage_config(config, manifest)
    class CleanFailure(MockCasa):
        def tclean(self, **kwargs):
            if kwargs.get("niter", 0) > 0:
                raise RuntimeError("synthetic CLEAN failure")
            return super().tclean(**kwargs)
    with pytest.raises(RuntimeError, match="synthetic CLEAN failure"):
        reimage_saved_ms(ms, config, tmp_path/"failed", tasks=CleanFailure(), table_tool_factory=table_factory)
    report = json.loads((tmp_path/"failed/casa_products.json").read_text())
    assert report["status"] == "failed" and "synthetic CLEAN failure" in report["error"]
    assert report["input_ms_unchanged"] is True
    assert (tmp_path/"failed/ch00.residual.fits").is_file()


def test_saved_ms_detects_content_changes_and_keeps_failure_evidence(tmp_path):
    prepared, manifest = bundle(tmp_path)
    sim = run(prepared, tmp_path/"sim", tasks=MockCasa(), table_tool_factory=table_factory)
    ms = tmp_path/"sim"/sim["products"][0]["measurement_set"]
    config = tmp_path/"observing.json";reimage_config(config, manifest)
    class UnexpectedWrite(MockCasa):
        def tclean(self, **kwargs):
            if kwargs.get("niter", 0) > 0:
                (Path(kwargs["vis"])/"table.dat").write_text("TEST-DOUBLE: UNEXPECTED MUTATION")
            return super().tclean(**kwargs)
    with pytest.raises(RuntimeError, match="content changed"):
        reimage_saved_ms(ms, config, tmp_path/"failed", tasks=UnexpectedWrite(), table_tool_factory=table_factory)
    report = json.loads((tmp_path/"failed/casa_products.json").read_text())
    assert report["status"] == "failed" and report["input_ms_unchanged"] is False
    assert report["input_ms_content_sha256_before"] != report["input_ms_content_sha256_after"]


def test_standalone_policy_import_and_legacy_and_reimage_cli_options(monkeypatch):
    path = Path(__file__).parents[1]/"src/hii_sim/casa_simulate.py"
    spec = importlib.util.spec_from_file_location("standalone_casa_runner", path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module._policy_module().configured_multiscale_policy(
        beam_major_arcsec=16.65, cell_arcsec=2.5, npix=256,
        min_usable_uv_lambda=374., observed_border_rms_jy_beam=.0002)["tclean_overrides"]["scales"] == [0,7,20,60]
    seen = []
    monkeypatch.setattr(module, "run", lambda *args, **kwargs: seen.append((args,kwargs)))
    monkeypatch.setattr(module, "reimage_saved_ms", lambda *args, **kwargs: seen.append((args,kwargs)))
    module.main(["prepared", "new"])
    assert seen[-1] == (("prepared", "new"), {"reconstruction_policy":"hogbom", "noise_multiplier":3.})
    module.main(["reimage", "--ms", "saved.ms", "--settings", "observing.json", "--out", "fresh", "--policy", "multiscale", "--noise-multiplier", "2.25"])
    assert seen[-1] == (("saved.ms", "observing.json", "fresh"), {"reconstruction_policy":"multiscale", "noise_multiplier":2.25})
