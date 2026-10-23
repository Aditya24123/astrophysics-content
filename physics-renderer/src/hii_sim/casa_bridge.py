"""Prepare Astropy FITS sky planes for the separate CASA measurement path."""
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u
from astropy.wcs import WCS
from .io import digest, versions, code_digest


def casa_direction_j2000(coord):
    """Serialize FK5/J2000 coordinates with explicit CASA sexagesimal units."""
    fk5 = coord.transform_to(FK5(equinox="J2000"))
    ra = fk5.ra.to_string(unit=u.hourangle, sep=("h", "m", "s"), precision=8, pad=True)
    dec = fk5.dec.to_string(unit=u.deg, sep=("d", "m", "s"), precision=8,
                             pad=True, alwayssign=True)
    return "J2000 " + ra + " " + dec

def prepare_casa_inputs(truth_fits, output_dir, seed=42017, duration="30min",
                        integration="10s", hourangle="transit", thermalnoise="tsys-atm",
                        reference_date="2026/10/19"):
    """Create a new relocatable, protected bundle; never overwrite old inputs."""
    if thermalnoise not in ("", "tsys-atm"):
        raise ValueError("Supported CASA noise models: '' or 'tsys-atm'")
    source = Path(truth_fits); out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    channels = []
    with fits.open(source, checksum=True, memmap=False) as hdus:
        if len(hdus) < 2:
            raise ValueError("Expected image extensions CHnnn from hii-sim")
        ref_header = hdus[1].header
        cdelt = abs(float(ref_header["CDELT1"]))
        omega = abs(np.linalg.det(WCS(ref_header).celestial.pixel_scale_matrix)) * (np.pi/180)**2
        npix = hdus[1].data.shape[-1]
        for i, hdu in enumerate(hdus[1:]):
            tag = "ch%02d" % i
            target = out / (tag + ".model.fits")
            if hdu.header.get("BUNIT") != "Jy/sr" or hdu.data.shape != (npix, npix):
                raise ValueError("CASA bridge requires equal-sized square Jy/sr sky planes")
            # Copy celestial coordinates, not arbitrary truth/physical FITS keywords.
            header = WCS(hdu.header).celestial.to_header()
            header["BUNIT"] = "Jy/pixel"
            header["FREQ"] = float(hdu.header["FREQHZ"])
            fits.PrimaryHDU(np.asarray(hdu.data, dtype=np.float32)*omega,
                            header=header).writeto(target, overwrite=False, checksum=True)
            channels.append({"tag":tag,"fits":target.name,"sha256":digest(target),
                             "frequency_ghz":float(hdu.header["FREQHZ"])/1e9,
                             "bandwidth_mhz":float(hdu.header["BANDWID"])/1e6,
                             "spectral_channels":1,
                             "seed":int(seed+i)})
    center = SkyCoord.from_pixel(npix//2, npix//2, WCS(ref_header).celestial).transform_to(FK5(equinox="J2000"))
    direction = casa_direction_j2000(center)
    pointings = out / "pointings.txt"
    pointings.write_text(direction + "\n", encoding="utf-8")
    manifest = {"schema_version":"2.0","channels":channels,"npix":npix,"cell":"%.9farcsec" % (cdelt*3600),
                "mapsize_arcsec":"%.6farcsec" % (npix*cdelt*3600),
                "direction":direction,"pointings":pointings.name,
                "pointings_sha256":digest(pointings),
                "duration":duration,"integration":integration,"hourangle":hourangle,
                "thermalnoise":thermalnoise,"reference_date":reference_date,
                "sky_unit_conversion":"Jy/sr times tangent-plane pixel solid angle to Jy/pixel",
                "primary_beam":"applied by CASA VLA antenna voltage pattern",
                "casa_array":"vla.c.cfg"}
    (out/"casa_inputs.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (out/"preparation_provenance.json").write_text(json.dumps({
        "protected":True,"source_fits_sha256":digest(source),"software_versions":versions(),
        "code_digest":code_digest(),"bundle_role":"simulation sky truth, never predictor input"
    },indent=2)+"\n",encoding="utf-8")
    return manifest
