"""Fail-closed contract and raw-chain acceptance for the declared experiment."""
import numpy as np
from bayes_followup import GroupedLikelihood
from diagnostics_v2 import diagnostic_report

TARGETS=('em0_pc_cm6','te_k','major_fwhm_arcsec','axis_ratio')
UNITS=('pc cm^-6','K','arcsec (full major EM FWHM)','1')
COORDS=list(TARGETS)+['pa_branch','east_prior_unit','north_prior_unit','common_gain_z','slope_z']
RANGES={n:r for n,r in zip(TARGETS,[(10000,500000,'log'),(5000,12000,'linear'),(12,60,'log'),(.5,1,'linear')])}


def validate_config(config,inputs):
    sampling=config['sampling']
    if tuple(config['reported_targets'])!=TARGETS or any(tuple(sampling['ranges'][n])!=RANGES[n] for n in TARGETS):
        raise ValueError('Unsupported four-target prior/support/transform contract')
    if tuple(sampling['ranges']['pa_deg'])!=(0,180,'linear') or sampling['offset_limit_arcsec']!=10:
        raise ValueError('Legacy likelihood hardcodes PA/offset priors; unsupported nuisance config rejected')
    if config['grid']!={'npix':96,'pixel_arcsec':2.5,'subpixels':4,'spatial_rule':'gauss_legendre'} or config['instrument']['frequency_quadrature']!=3:
        raise ValueError('Unsupported numerical renderer configuration')
    if config.get('profile')!='gaussian_em' or config['sky'].get('background_rj_k',0)!=0:
        raise ValueError('Unsupported morphology/background')
    if set(inputs)!={'features','bookkeeping'} or set(inputs['features'])!={'measurements','observing'}:
        raise ValueError('Predictor feature allowlist violated')
    m,o=inputs['features']['measurements'],inputs['features']['observing']
    inst=config['instrument']
    if m.unit!='Jy' or o.thermal.covariance_unit!='Jy^2' or len(m.values_jy)!=3906:
        raise ValueError('Unsupported measurement units/layout; challenges must stay separate')
    if o.npix!=96 or o.pixel_arcsec!=2.5 or o.taper_role!='imaging_weight_not_inverse_variance':
        raise ValueError('Observing grid/taper interpretation mismatch')
    for name in ('frequencies_ghz','bandwidths_mhz'):
        if not np.array_equal(getattr(o,name),inst[name]):
            raise ValueError('Observing bands/config mismatch')
    if (o.thermal.frequency_correlation!=inst['noise_frequency_correlation'] or
        o.calibration.fractional_common_amplitude_std!=inst['calibration_fraction'] or
        o.calibration.spectral_log_slope_std!=inst['calibration_slope_std']):
        raise ValueError('Thermal/calibration prior mismatch')


class CheckedLikelihood(GroupedLikelihood):
    def __init__(self,inputs,config):
        validate_config(config,inputs)
        super().__init__(inputs,config)


def physical_targets(chain,config):
    physical=chain[:,:,:4].copy()
    for j,name in enumerate(TARGETS):
        lo,hi,kind=config['sampling']['ranges'][name]
        physical[:,:,j]=lo*(hi/lo)**chain[:,:,j] if kind=='log' else lo+(hi-lo)*chain[:,:,j]
    return physical


def full_diagnostics(chain,config):
    if chain.ndim!=3 or chain.shape[2]!=9 or not np.isfinite(chain).all():
        raise ValueError('Full posterior must have nine finite coordinates')
    if np.any((chain[:,:,[0,1,2,3,5,6]]<0)|(chain[:,:,[0,1,2,3,5,6]]>1)):
        raise ValueError('Posterior outside declared physical/offset prior support')
    converted=chain.copy()
    converted[:,:,:4]=physical_targets(chain,config)
    a=2*np.pi*chain[:,:,4]
    values=np.concatenate((converted,np.sin(a)[...,None],np.cos(a)[...,None]),axis=2)
    return diagnostic_report(values,COORDS+['pa_sin','pa_cos'])


def validate_case_set(cases,expected):
    pairs=[(c['source_id'],c['realization_id']) for c in cases]
    if len(pairs)!=len(expected) or len(set(pairs))!=len(pairs) or set(pairs)!=set(expected):
        raise ValueError('Missing, extra, duplicate or wrong-observation comparator cases')
