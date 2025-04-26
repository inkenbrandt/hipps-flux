import pytest
import numpy as np
from eddy_covariance.freq_factor import (
    SensorSpecs, AtmConditions, FrequencyResponse, SpectralAnalysis
)

# Test data
@pytest.fixture
def sensor_specs():
    return SensorSpecs(
        path_length=0.15,  # m
        separation_distance=0.1,  # m
        time_constant=0.1,  # s
        measuring_height=3.0,  # m
        sampling_frequency=10.0  # Hz
    )

@pytest.fixture
def atm_conditions():
    return AtmConditions(
        wind_speed=2.0,  # m/s
        stability=-0.5,  # z/L
        temp=293.15,  # K
        pressure=101.3  # kPa
    )

@pytest.fixture
def freq_response(sensor_specs, atm_conditions):
    return FrequencyResponse(sensor_specs, atm_conditions)

def test_frequency_response_initialization(freq_response):
    """Test FrequencyResponse initialization"""
    assert freq_response.f_nyquist == 5.0  # Half sampling frequency
    assert freq_response.normalized_freq is None
    assert freq_response.transfer_funcs == {}

def test_calculate_normalized_frequency(freq_response):
    """Test normalized frequency calculation"""
    freq = freq_response._calculate_normalized_frequency(n_points=100)
    
    assert len(freq) == 100
    assert np.all(freq >= 0)
    assert np.all(np.diff(freq) > 0)  # Should be monotonically increasing
    assert freq[0] == pytest.approx(0.001 * 3.0 / 2.0)  # First point check

def test_block_averaging_transfer(freq_response):
    """Test block averaging transfer function"""
    H_b = freq_response.block_averaging_transfer()
    
    assert len(H_b) == 1000  # Default n_points
    assert np.all(H_b <= 1.0)  # Transfer function should be <= 1
    assert np.all(H_b >= 0.0)  # Transfer function should be >= 0
    assert H_b[0] == pytest.approx(1.0)  # Should be 1 at f=0
    
    # Check if stored in transfer_funcs
    assert 'block' in freq_response.transfer_funcs
    assert np.array_equal(freq_response.transfer_funcs['block'], H_b)

def test_line_averaging_transfer(freq_response):
    """Test line averaging transfer function"""
    H_l = freq_response.line_averaging_transfer()
    
    assert len(H_l) == 1000  # Default n_points
    assert np.all(H_l <= 1.0)  # Transfer function should be <= 1
    assert np.all(H_l >= 0.0)  # Transfer function should be >= 0
    assert H_l[0] == pytest.approx(1.0)  # Should be 1 at f=0
    
    # Check if stored in transfer_funcs
    assert 'line' in freq_response.transfer_funcs
    assert np.array_equal(freq_response.transfer_funcs['line'], H_l)

def test_time_response_transfer(freq_response):
    """Test time response transfer function"""
    H_r = freq_response.time_response_transfer()
    
    assert len(H_r) == 1000  # Default n_points
    assert np.all(H_r <= 1.0)  # Transfer function should be <= 1
    assert np.all(H_r >= 0.0)  # Transfer function should be >= 0
    assert H_r[0] == pytest.approx(1.0)  # Should be 1 at f=0
    
    # Check if stored in transfer_funcs
    assert 'time' in freq_response.transfer_funcs
    assert np.array_equal(freq_response.transfer_funcs['time'], H_r)

def test_sensor_separation_transfer(freq_response):
    """Test sensor separation transfer function"""
    H_s = freq_response.sensor_separation_transfer()
    
    assert len(H_s) == 1000  # Default n_points
    assert np.all(H_s <= 1.0)  # Transfer function should be <= 1
    assert np.all(H_s >= 0.0)  # Transfer function should be >= 0
    assert H_s[0] == pytest.approx(1.0)  # Should be 1 at f=0
    
    # Check if stored in transfer_funcs
    assert 'separation' in freq_response.transfer_funcs
    assert np.array_equal(freq_response.transfer_funcs['separation'], H_s)

def test_total_transfer_function(freq_response):
    """Test total transfer function calculation"""
    H_total = freq_response.total_transfer_function()
    
    assert len(H_total) == 1000  # Default n_points
    assert np.all(H_total <= 1.0)  # Combined transfer function should be <= 1
    assert np.all(H_total >= 0.0)  # Combined transfer function should be >= 0
    assert H_total[0] == pytest.approx(1.0)  # Should be 1 at f=0
    
    # Should be product of individual transfer functions
    H_expected = (freq_response.block_averaging_transfer() *
                 freq_response.line_averaging_transfer() *
                 freq_response.time_response_transfer() *
                 freq_response.sensor_separation_transfer())
    assert np.allclose(H_total, H_expected)

def test_calculate_correction_factor(freq_response):
    """Test correction factor calculation"""
    correction = freq_response.calculate_correction_factor(
        stability=-0.5,
        integration_method='simpson'
    )
    
    assert isinstance(correction, float)
    assert correction >= 1.0  # Correction factor should be >= 1
    assert np.isfinite(correction)
    
    # Test alternative integration method
    correction_trapz = freq_response.calculate_correction_factor(
        stability=-0.5,
        integration_method='trapz'
    )
    assert isinstance(correction_trapz, float)
    assert np.isclose(correction, correction_trapz, rtol=1e-2)

# SpectralAnalysis tests
def test_calculate_power_spectrum():
    """Test power spectrum calculation"""
    # Generate test signal
    t = np.linspace(0, 10, 1000)
    freq = 1.0  # Hz
    signal = np.sin(2 * np.pi * freq * t)
    
    freqs, power = SpectralAnalysis.calculate_power_spectrum(
        signal, sampling_freq=100.0
    )
    
    assert len(freqs) == len(power)
    assert np.abs(freqs[np.argmax(power)] - freq) < 0.1  # Peak near 1 Hz
    assert np.all(power >= 0)  # Power should be non-negative

def test_calculate_cospectra():
    """Test co-spectra calculation"""
    # Generate correlated test signals
    t = np.linspace(0, 10, 1000)
    signal1 = np.sin(2 * np.pi * t)
    signal2 = np.sin(2 * np.pi * t + np.pi/4)  # Phase shifted
    
    freqs, co_spec = SpectralAnalysis.calculate_cospectra(
        signal1, signal2, sampling_freq=100.0
    )
    
    assert len(freqs) == len(co_spec)
    assert np.any(co_spec != 0)  # Should have non-zero co-spectral components

# Error handling tests
def test_invalid_stability():
    """Test handling of invalid stability values"""
    with pytest.raises(ValueError):
        freq_response.calculate_correction_factor(stability=-10.0)

def test_invalid_integration_method():
    """Test handling of invalid integration method"""
    with pytest.raises(ValueError):
        freq_response.calculate_correction_factor(
            stability=-0.5,
            integration_method='invalid'
        )
