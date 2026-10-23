"""FITS/WCS and atomic/checksummed storage. Unequal frequencies are never linearized."""
import hashlib
import json
import os
import platform
from pathlib import Path
from importlib.metadata import version
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def write_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


def versions():
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in ("numpy","scipy","astropy","matplotlib")}}


def source_manifest():
    """Version the executable package, command-line tools and dependency declarations."""
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src/hii_sim").glob("*.py"))
    paths += list((root / "tools").glob("*.py"))
    paths += [p for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt")
              if (p := root / name).is_file()]
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}


def code_digest():
    return canonical_hash(source_manifest())


def write_image(path, cube, grid, sky, inst, unit, kind):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [grid.npix//2+1]*2
    w.wcs.crval = [sky.ra_deg,sky.dec_deg]
    w.wcs.cdelt = [-grid.pixel_arcsec/3600,grid.pixel_arcsec/3600]
    header = w.to_header()
    header["BUNIT"] = unit
    header["RADESYS"] = "ICRS"
    primary = fits.PrimaryHDU()
    primary.header["SCHEMAV"] = "1.0"
    primary.header["PRODUCT"] = kind
    hdus = [primary]
    for i,(f,b,plane) in enumerate(zip(inst.frequencies_ghz,inst.bandwidths_mhz,cube)):
        h = header.copy(); h["FREQHZ"] = f*1e9; h["BANDWID"] = b*1e6
        h["SPECSYS"] = "TOPOCENT"
        if unit == "Jy/beam":
            h["BEAMTYPE"] = "DIRTY"
            h["COMMENT"] = "Unit-peak dirty PSF; Gaussian beam-area flux conversion invalid"
        hdus.append(fits.ImageHDU(np.asarray(plane,dtype=np.float32),header=h,name=f"CH{i:03}"))
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    fits.HDUList(hdus).writeto(tmp,overwrite=True,checksum=True)
    os.replace(tmp,path)


def read_image(path):
    with fits.open(path,checksum=True,memmap=False) as hdus:
        cube = np.stack([h.data for h in hdus[1:]])
        metadata = {"wcs": WCS(hdus[1].header).to_header_string(),
                    "frequencies_hz": [h.header["FREQHZ"] for h in hdus[1:]],
                    "bandwidths_hz": [h.header["BANDWID"] for h in hdus[1:]],
                    "unit": hdus[1].header["BUNIT"]}
    return cube, metadata
