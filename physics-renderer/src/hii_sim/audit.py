"""Run bounded scientific audits in a new directory; never regenerate a dataset."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .io import atomic_json, source_manifest, code_digest, versions, digest
from .schema import load_config, Grid, Instrument, Sky
from .validation import validate
from .convergence import convergence_suite
from .instrument_validation import validate_bandwidth
from .reconstruction_validation import validate_reconstruction
from .config_defaults import DEFAULT_CONFIG, LEGACY_CONFIG


def audit(config_path, output, widefield=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    start_manifest = source_manifest()
    config = load_config(config_path)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "versions": versions(),
                  "source_manifest": start_manifest, "code_sha256": code_digest(),
                  "test_sources_sha256": {p.as_posix(): digest(p) for p in Path("tests").glob("*.py")},
                  "config_sha256": digest(config_path), "configuration": config,
                  "command": {"config": str(config_path), "output": str(output), "widefield": widefield}}
    atomic_json(out/"provenance.json", provenance)
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        print("Checking physics and six sensitivity regimes", flush=True)
        physics = validate(config_path, out/"physics")
        print("Checking 32 spatial-integration cases and spectral refinement", flush=True)
        convergence = convergence_suite(config["grid"]["pixel_arcsec"])
        atomic_json(out/"convergence.json", {"provenance": provenance, **convergence})
        print("Checking fixed-baseline bandwidth approximation by direct DFT", flush=True)
        instrument = validate_bandwidth(Grid(**config["grid"]), Instrument(**config["instrument"]), Sky(**config["sky"]))
        atomic_json(out/"instrument.json", {"provenance": provenance, **instrument})
        print("Checking measurement-only CLEAN against point/extended fixtures", flush=True)
        reconstruction = validate_reconstruction(out/"reconstruction", grid=Grid(**config["grid"]),
                                    inst=Instrument(**config["instrument"]), widefield=widefield)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        sensitivity = physics["checks"]["sensitivity"]
        values = np.array([item["scaled_response_snr_for_delta_0p01"] for item in sensitivity.values()])
        im = axes[0].imshow(np.log10(np.maximum(values, 1e-12)), aspect="auto", cmap="viridis", vmin=-3)
        axes[0].set_xticks(range(5), ["ln EM", "ln Te", "ln size", "q", "PA (rad)"], rotation=25)
        axes[0].set_yticks(range(len(sensitivity)), list(sensitivity))
        axes[0].set_title("Local response to 0.01 scaled change\nthermal covariance + calibration nuisances")
        fig.colorbar(im, ax=axes[0], label="log10 response / noise (not recoverability)")
        for profile in ("gaussian_em", "ellipsoid_em"):
            cases = [r for r in convergence["cases"] if r["source"]["profile"] == profile]
            errors = [max(r["errors_vs_subpixels_16"][str(n)]["relative_pixel_l2"] for r in cases) for n in (1,2,4,8)]
            axes[1].loglog([1,2,4,8], errors, "o-", label=profile)
        axes[1].set(xlabel="Subpixels per pixel axis", ylabel="Worst relative image L2 error vs 16", title="32 crossed numerical integration cases")
        axes[1].legend()
        fig.savefig(out/"audit_diagnostics.png", dpi=160); plt.close(fig)
        unchanged = source_manifest() == start_manifest
        report = {"status": "completed_with_diagnostics" if unchanged else "invalidated_code_changed_during_run",
                  "assertion_checks": "physics/normalization/reproducibility assertions passed; diagnostic error values require scope-specific interpretation",
                  "source_unchanged_during_run": unchanged, "wall_seconds": time.perf_counter()-wall,
                  "cpu_seconds": time.process_time()-cpu, "code_sha256": provenance["code_sha256"],
                  "spatial_cases": convergence["case_count"], "sensitivity_regimes": len(sensitivity),
                  "bandwidth_cases": len(instrument["bandwidth_cases"]), "reconstruction_cases": len(reconstruction["cases"]),
                  "CASA": "not executed; see separate CASA blocker report", "bulk_dataset_generation": False,
                  "products_sha256": {p.relative_to(out).as_posix(): digest(p) for p in out.rglob("*") if p.is_file()}}
        atomic_json(out/"audit_summary.json", report)
        if not unchanged:
            raise RuntimeError("Source changed during audit; rerun into a new output directory")
        return report
    except Exception as exc:
        atomic_json(out/"failure.json", {"type": type(exc).__name__, "message": str(exc), "provenance": provenance})
        raise


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Configuration file (default: {DEFAULT_CONFIG}; use {LEGACY_CONFIG} only for explicit legacy audit)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--widefield", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = audit(args.config, args.out, args.widefield)
    print(result)


if __name__ == "__main__":
    main()
