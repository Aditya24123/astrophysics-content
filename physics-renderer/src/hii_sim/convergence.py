"""Expanded bounded numerical challenge suite, not population robustness proof."""
from dataclasses import asdict
from itertools import product
import numpy as np
from .physics import render
from .schema import Source, Grid, Sky


def convergence_suite(pixel_arcsec=2.5):
    cases = []
    # 32 crossed cases include non-Gaussian edges and compact elongated sources.
    for profile, em, size, ratio, phase in product(
            ("gaussian_em", "ellipsoid_em"), (1e4, 1e7), (12., 60.), (.3, .8), (0, 1)):
        source = Source(em, 8000, size, ratio, 37 if phase == 0 else 113, profile)
        sky = Sky(east_arcsec=.4*pixel_arcsec*phase, north_arcsec=-.3*pixel_arcsec*phase)
        reference = render(source, Grid(64, pixel_arcsec, 16), sky, [1.44], [0], 1)[0]
        errors = {}
        for subpixels in (1, 2, 4, 8):
            plane = render(source, Grid(64, pixel_arcsec, subpixels), sky, [1.44], [0], 1)[0]
            errors[str(subpixels)] = {"relative_pixel_l2": float(np.linalg.norm(plane-reference)/np.linalg.norm(reference)),
                "relative_flux_error": float(abs(plane.sum()-reference.sum())/reference.sum())}
        cases.append({"source": asdict(source), "sky": asdict(sky), "errors_vs_subpixels_16": errors})
    refinement = []
    for profile in ("gaussian_em", "ellipsoid_em"):
        worst = max((x for x in cases if x["source"]["profile"] == profile),
                    key=lambda x: x["errors_vs_subpixels_16"]["8"]["relative_pixel_l2"])
        source, sky = Source(**worst["source"]), Sky(**worst["sky"])
        p16 = render(source, Grid(64, pixel_arcsec, 16), sky, [1.44], [0], 1)[0]
        p32 = render(source, Grid(64, pixel_arcsec, 32), sky, [1.44], [0], 1)[0]
        refinement.append({"source": asdict(source), "relative_pixel_l2_16_vs_32": float(np.linalg.norm(p16-p32)/np.linalg.norm(p32)),
                           "relative_flux_16_vs_32": float(abs(p16.sum()-p32.sum())/p32.sum())})
    frequency = []
    for temperature, em in product((5500, 8000, 11500), (1e4, 5e5, 1e7)):
        source = Source(em, temperature, 12, .5, 37)
        p3 = render(source, Grid(64, pixel_arcsec, 4), Sky(), [1.06, 8.0], [128, 128], 3)
        p7 = render(source, Grid(64, pixel_arcsec, 4), Sky(), [1.06, 8.0], [128, 128], 7)
        frequency.append({"te_k": temperature, "em0_pc_cm6": em,
            "relative_l2_by_frequency": (np.linalg.norm(p3-p7, axis=(1,2))/np.linalg.norm(p7, axis=(1,2))).tolist()})
    return {"case_count": len(cases), "grid": {"npix": 64, "pixel_arcsec": pixel_arcsec},
            "frequency_ghz": 1.44, "reference_subpixels": 16, "cases": cases,
            "worst_per_profile_reference_refinement": refinement, "spectral_3_vs_7": frequency,
            "scope": "numerical integration stress cases, no assertion of population/instrument robustness; finite-grid comparisons"}
