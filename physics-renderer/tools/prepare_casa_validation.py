"""Prepare bounded, protected numerical fixtures for a real CASA host.

Produces input FITS only; no fake Measurement Sets or validation-success flag.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from hii_sim.casa_bridge import prepare_casa_inputs
from hii_sim.io import write_image
from hii_sim.schema import Grid, Instrument, Sky


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="New protected fixture directory")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    grid = Grid(npix=256, pixel_arcsec=2.5)
    inst = Instrument(frequencies_ghz=(1.44,), bandwidths_mhz=(8.,))
    east, north = grid.coordinates()
    cases = []
    for name, fwhm in (("point", 0.), ("extended45", 45.), ("extended90", 90.)):
        plane = np.zeros((grid.npix, grid.npix))
        if fwhm == 0:
            plane[grid.npix//2, grid.npix//2] = 1
        else:
            plane = np.exp(-4*np.log(2)*(east**2+north**2)/fwhm**2)
        plane *= .01 / plane.sum() / grid.omega_sr
        truth = out/(name+".fits")
        write_image(truth, plane[None], grid, Sky(), inst, "Jy/sr", "validation_truth")
        for noise in ("", "tsys-atm"):
            case = name + ("_noisy" if noise else "_noiseless")
            prepare_casa_inputs(truth, out/case, thermalnoise=noise)
            cases.append({"bundle":case,"input_flux_jy":.01,"fwhm_arcsec":fwhm,
                          "thermalnoise":noise})
    (out/"cases.json").write_text(json.dumps({"protected":True,"status":"inputs_prepared_CASA_not_run",
        "purpose":"Point and extended flux, PSF, missing-scale and residual checks",
        "scope":"Single narrow 8 MHz channel at 1.44 GHz; not full 128 MHz channel validation",
        "cases":cases},indent=2)+"\n")
    print(out.resolve())


if __name__ == "__main__":
    main()
