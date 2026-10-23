"""Fixed-formula/accounting tests; no CASA or fresh-noise experiment."""
from dataclasses import asdict
import inspect
from pathlib import Path
import runpy
import numpy as np
import pytest
from hii_sim.visibility_photometry import (measured_thermal_calibration,observing_weights,
    correlated_flux,ThermalCalibration,prediction_critical)


def fixture(n=12):
    row=np.arange(n,dtype=float)
    uv=np.stack((row*80.,row*17.,row*-3.))
    v=np.stack((.01+.001*row+1j*.002*row,.01-.0005*row-1j*.001*row))
    cal=ThermalCalibration(.002,(.0018,.0024),100)
    return v,uv,cal


def test_point_normalization_and_full_covariance():
    _,uv,cal=fixture()
    point=np.full((2,uv.shape[1]),.01+0j)
    result=correlated_flux(point,uv,1.44e9,cal)
    np.testing.assert_allclose(result.values_jy,.01,rtol=1e-15)
    assert result.covariance_jy2.shape==(2,2) and result.covariance_jy2[0,1]>0
    assert result.resolved_total_flux is False
    assert "conditional" in result.uncertainty
    assert np.linalg.eigvalsh(result.covariance_jy2).min()>=0
    # At zero baselines both operators coincide: equal variance and perfect covariance.
    zero=correlated_flux(point,np.zeros_like(uv),1.44e9,cal)
    np.testing.assert_allclose(zero.covariance_jy2,cal.sigma_component_jy**2/(2*len(point[0])))
    with pytest.raises(ValueError):result.values_jy[0]=0


def test_linearity_and_covariance_from_independent_receiver_basis():
    v,uv,cal=fixture(5)
    first=correlated_flux(v,uv,1.44e9,cal)
    second=correlated_flux(2*v+(.003-.007j),uv,1.44e9,cal)
    np.testing.assert_allclose(second.values_jy,2*first.values_jy+.003,rtol=1e-14)
    np.testing.assert_array_equal(first.covariance_jy2,second.covariance_jy2)
    # Exact pushforward of independent real receiver basis, no Monte Carlo gate.
    jac=[]
    for hand in range(2):
        for row in range(v.shape[1]):
            basis=np.zeros_like(v);basis[hand,row]=1.
            jac.append(correlated_flux(basis,uv,1.44e9,cal).values_jy)
    jac=np.stack(jac,axis=1)
    np.testing.assert_allclose(first.covariance_jy2,cal.sigma_component_jy**2*jac@jac.T,rtol=2e-15)
    halfwidth=prediction_critical()*np.sqrt(np.diag(first.covariance_jy2))*cal.sigma_99_9percent_interval_jy[1]/cal.sigma_component_jy
    np.testing.assert_allclose(first.marginal_conservative_95_intervals_jy,
        np.column_stack((first.values_jy-halfwidth,first.values_jy+halfwidth)))


def test_exact_frozen_formula_parity():
    frozen=runpy.run_path(str(Path("output/training_readiness_20261021/photometry_measurements_v1.py")))
    v,uv,_=fixture(101)
    old=frozen["measured_thermal_calibration"](v)
    current=measured_thermal_calibration(v)
    current_record=asdict(current)
    current_record["sigma_99_9percent_interval_jy"]=list(current_record["sigma_99_9percent_interval_jy"])
    assert current_record==old
    np.testing.assert_array_equal(observing_weights(uv,1.44e9),frozen["observing_weights"](uv,1.44e9))
    expected,covariance=frozen["photometry"](v,frozen["observing_weights"](uv,1.44e9),old["sigma_component_jy"])
    result=correlated_flux(v,uv,1.44e9,current)
    np.testing.assert_array_equal(result.values_jy,expected)
    np.testing.assert_array_equal(result.covariance_jy2,covariance)
    assert prediction_critical()==frozen["prediction_critical"]()


def test_calibration_uses_measured_hand_difference_only():
    # Analytic powers with known lower median; adding equal-hand signal cancels.
    z=np.sqrt(np.array([1.,4.,9.,16.,25.,36.]))+0j
    v=np.stack((np.sqrt(2)*z,np.zeros_like(z)))
    base=measured_thermal_calibration(v)
    shifted=measured_thermal_calibration(v+(3.+2j))
    assert base.sigma_component_jy==pytest.approx(np.sqrt(9/(2*np.log(2))))
    assert shifted.sigma_component_jy==pytest.approx(base.sigma_component_jy,rel=1e-15)
    assert base.sigma_99_9percent_interval_jy[0]<base.sigma_component_jy<base.sigma_99_9percent_interval_jy[1]
    assert base.physical_inverse_variance_from_native_WEIGHT is False


@pytest.mark.parametrize("name",["source_size","true_gain","true_centroid","noiseless_reference","sky_image","source_mask"])
def test_operational_api_rejects_protected_arguments(name):
    v,uv,cal=fixture()
    assert name not in inspect.signature(correlated_flux).parameters
    with pytest.raises(TypeError):correlated_flux(v,uv,1.44e9,cal,**{name:1.})


@pytest.mark.parametrize("defect",["channels","real_data","empty","nan","frequency_array","zero_frequency",
    "complex_uvw","wrong_rows","flags","row_flags","weight_hands","weight_rows","polarized","linear_hands","bad_calibration"])
def test_unsupported_or_malformed_domains_rejected(defect):
    v,uv,cal=fixture();frequency=1.44e9;kwargs={}
    if defect=="channels":v=v[:,:,None]
    if defect=="real_data":v=v.real
    if defect=="empty":v=v[:,:0];uv=uv[:,:0]
    if defect=="nan":v[0,0]=np.nan
    if defect=="frequency_array":frequency=[1.44e9]
    if defect=="zero_frequency":frequency=0.
    if defect=="complex_uvw":uv=uv.astype(complex)
    if defect=="wrong_rows":uv=uv[:,:-1]
    if defect=="flags":kwargs["flags"]=np.ones(v.shape,dtype=bool)
    if defect=="row_flags":kwargs["row_flags"]=np.ones(v.shape[1],dtype=bool)
    if defect in ("weight_hands","weight_rows"):
        weights=np.ones(v.shape)
        if defect=="weight_hands":weights[0]=2.
        else:weights[:,0]=2.
        kwargs["native_weights"]=weights
    if defect=="polarized":kwargs["unpolarized"]=False
    if defect=="linear_hands":kwargs["correlations"]=("XX","YY")
    if defect=="bad_calibration":cal=.002
    with pytest.raises(ValueError):correlated_flux(v,uv,frequency,cal,**kwargs)


def test_uniform_native_weights_and_unflagged_metadata_do_not_set_noise_scale():
    v,uv,cal=fixture()
    reference=correlated_flux(v,uv,1.44e9,cal)
    result=correlated_flux(v,uv,1.44e9,cal,flags=np.zeros(v.shape,dtype=bool),
                           row_flags=np.zeros(v.shape[1],dtype=bool),native_weights=np.full(v.shape,99.))
    np.testing.assert_array_equal(result.covariance_jy2,reference.covariance_jy2)
    with pytest.raises(ValueError):measured_thermal_calibration(np.zeros_like(v))
    with pytest.raises(ValueError):measured_thermal_calibration(v,flags=np.ones(v.shape,dtype=bool))
