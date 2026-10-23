"""Plot an actual paired sky/measurement sample as a review-only PNG preview."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm

from hii_sim.loaders import PairedDataset
from hii_sim.io import code_digest, digest


def sky_extent(npix, pixel_arcsec):
    """Pixel-edge offsets for FITS reference pixel npix//2 (east left)."""
    low = (-npix//2 - .5) * pixel_arcsec
    high = (npix - npix//2 - .5) * pixel_arcsec
    return (-low, -high, low, high)


def uv_coordinates_klambda(npix, pixel_arcsec):
    """DFT conjugates of sky radians are already wavelengths, at every band."""
    uv = np.fft.fftfreq(npix, d=np.deg2rad(pixel_arcsec/3600))
    vv, uu = np.meshgrid(uv, uv, indexing="ij")
    # The array x coordinate increases west; astronomical u is east-positive.
    return -uu/1000, vv/1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="output/visual_demo_v1")
    parser.add_argument("--out", default="output/visual_demo_v1/preview.png")
    parser.add_argument("--split", default="train")
    parser.add_argument("--frequency-index", type=int, default=2)
    parser.add_argument("--allow-legacy", action="store_true",
                        help="Review compromised schema1 data explicitly; never use it for blind evaluation")
    args = parser.parse_args()

    root = Path(args.dataset)
    dataset = PairedDataset(root, split=args.split, verify=True, allow_legacy=args.allow_legacy)
    inputs, targets = dataset[0]
    frequency_index = args.frequency_index
    bookkeeping = inputs["bookkeeping"]
    inst_meta = inputs["metadata"]
    instrument = inst_meta["instrument"]
    grid = inst_meta["grid"]
    frequencies = instrument["frequencies_ghz"]
    frequency = frequencies[frequency_index]
    dirty = inputs["observations"]["dirty_image"][frequency_index]
    truth = targets["truth_sky_jy_sr"][frequency_index]
    weights_all = inputs["metadata"]["weights"]
    weights = weights_all[frequency_index]
    psf = inputs["metadata"]["psf"][frequency_index]
    measured = inputs["observations"]["vis_jy"]
    support = weights > 0
    support_all = weights_all > 0
    if measured.size != int(support_all.sum()):
        raise ValueError("Visibility vector does not match the published multi-frequency uv support")
    visibility_cube = np.zeros_like(weights_all, dtype=np.complex128)
    visibility_cube[support_all] = measured
    measured_plane = visibility_cube[frequency_index, support]

    n = int(grid["npix"])
    pixel_arcsec = float(grid["pixel_arcsec"])
    extent = sky_extent(n, pixel_arcsec)
    uu, vv = uv_coordinates_klambda(n, pixel_arcsec)
    u_klambda = uu[support]
    v_klambda = vv[support]
    amplitude_db = 20*np.log10(np.maximum(np.abs(measured_plane), 1e-12))

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0), constrained_layout=True)
    positive = truth[truth > 0]
    vmin = max(float(np.max(truth))*1e-5, float(np.min(positive)))
    art = axes[0, 0].imshow(truth, origin="lower", extent=extent, cmap="magma",
                            norm=LogNorm(vmin=vmin, vmax=float(np.max(truth))))
    axes[0, 0].set_title("Generated intrinsic sky intensity")
    axes[0, 0].set_xlabel("East offset (arcsec; east is left)")
    axes[0, 0].set_ylabel("North offset (arcsec)")
    fig.colorbar(art, ax=axes[0, 0], label="Jy/sr")

    bound = float(np.percentile(np.abs(dirty), 99.5))
    art = axes[0, 1].imshow(dirty, origin="lower", extent=extent, cmap="RdBu_r",
                            norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    axes[0, 1].set_title("Dirty image from noisy sampled visibilities")
    axes[0, 1].set_xlabel("East offset (arcsec; east is left)")
    axes[0, 1].set_ylabel("North offset (arcsec)")
    fig.colorbar(art, ax=axes[0, 1], label="Jy/beam (unit-peak dirty PSF)")

    art = axes[1, 0].imshow(psf, origin="lower", extent=extent, cmap="RdBu_r",
                            vmin=-0.25, vmax=1)
    axes[1, 0].set_title("Sampling response (dirty PSF)")
    axes[1, 0].set_xlabel("East offset (arcsec; east is left)")
    axes[1, 0].set_ylabel("North offset (arcsec)")
    fig.colorbar(art, ax=axes[1, 0], label="Normalized response")

    scatter = axes[1, 1].scatter(u_klambda, v_klambda, c=amplitude_db, s=9,
                                 cmap="viridis", linewidths=0)
    axes[1, 1].set_aspect("equal", adjustable="datalim")
    axes[1, 1].set_title("Measured visibility amplitudes on sampled uv cells")
    axes[1, 1].set_xlabel("u (kλ)")
    axes[1, 1].set_ylabel("v (kλ)")
    fig.colorbar(scatter, ax=axes[1, 1], label="20 log10(|V| / 1 Jy) (dB)")

    fig.suptitle(f"Synthetic H II sample | {frequency:.2f} GHz | {args.split} realization",
                 fontsize=14)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    metadata = {
        "preview_only": True,
        "protected_review_artifact": True,
        "preview_script_sha256":digest(__file__),
        "package_code_digest":code_digest(),
        "axis_convention":"east left, north up; pixel-edge offsets relative to FITS reference pixel",
        "uv_unit_conversion":"DFT spatial frequency (wavelengths) / 1000; u sign reversed for east-positive sky",
        "source_id": bookkeeping["source_id"],
        "realization_id": bookkeeping["realization_id"],
        "split": args.split,
        "frequency_ghz": frequency,
        "intrinsic_unit": targets["truth_metadata"]["unit"],
        "dirty_image_unit": inputs["metadata"]["image"]["unit"],
        "dirty_image_origin": "measured visibilities reconstructed with published fast-path weights",
        "out": str(out),
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
