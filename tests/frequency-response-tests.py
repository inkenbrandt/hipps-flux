import pytest
import numpy as np
from easyfluxpy.freq_factor import (
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
    fr = FrequencyResponse(sensor_specs, atm_conditions)
    # Initialize normalized frequencies before tests
    fr._calculate_normalized_frequency()
    return fr


def test_frequency_response_initialization(freq_response):
    """Test FrequencyResponse initialization"""
    assert freq_response.f_nyquist == 5.0  # Half sampling frequency
    assert freq_response.normalized_freq is not None  # Should be initialized
    assert isinstance(freq_response.transfer_funcs, dict)


def test_calculate_normalized_frequency(freq_response):
    """Test normalized frequency calculation"""
    freq = freq_response.normalized_freq

    assert len(freq) == 1000  # Default n_points
    assert np.all(freq >= 0)
    assert np.all(np.diff(freq) > 0)  # Should be monotonically increasing
    assert freq[0] == pytest.approx(0.001 * 3.0 / 2.0, rel=1e-3)  # First point check


def test_block_averaging_transfer(freq_response):
    """Test block averaging transfer function"""
    H_b = freq_response.block_averaging_transfer()

    assert len(H_b) == 1000  # Default n_points
    assert np.all(H_b <= 1.0)  # Transfer function should be <= 1
    assert np.all(H_b >= 0.0)  # Transfer function should be >= 0
    assert H_b[0] == pytest.approx(1.0, rel=1e-4)  # Should be 1 at f=0

    # Check if stored in transfer_funcs
    assert 'block' in freq_response.transfer_funcs
    assert np.array_equal(freq_response.transfer_funcs['block'], H_b)


def test_line_averaging_transfer(freq_response):
    """Test line averaging transfer function"""
    H_l = freq_response.line_averaging_transfer()

    assert len(H_l) == 1000  # Default n_points
    assert np.all(H_l <= 1.0)  # Transfer function should be <= 1
    assert np.all(H_l >= 0.0)  # Transfer function should be >= 0
    assert H_l[0] == pytest.approx(1.0, rel=1e-4)  # Should be 1 at f=0

    # Check if stored in transfer_funcs
    assert 'line' in freq_response.transfer_funcs
    assert np.array_equal(freq_response.transfer_funcs['line'], H_l)


def test_time_response_transfer(freq_response):
    """Test time response transfer function"""
    H_r = freq_response.time_response_transfer()

    assert len(H_r) == 1000  # Default n_points
    assert np.all(H_r <= 1.0)  # Transfer function should be <= 1
    assert np.all(H_r >= 0.0)  # Transfer function should be >= 0
    assert H_r[0] == pytest.approx(1.0, rel=1e-4)  # Should be 1 at f=0

    # Check if stored in transfer_funcs
    assert 'time' in freq_response.transfer_funcs
    assert np.array_equal(freq_response.transfer_funcs['time'], H_r)


def test_sensor_separation_transfer(freq_response):
    """Test sensor separation transfer function"""
    H_s = freq_response.sensor_separation_transfer()

    assert len(H_s) == 1000  # Default n_points
    assert np.all(H_s <= 1.0)  # Transfer function should be <= 1
    assert np.all(H_s >= 0.0)  # Transfer function should be >= 0
    assert H_s[0] == pytest.approx(1.0, rel=1e-4)  # Should be 1 at f=0 with relaxed tolerance

    # Check if stored in transfer_funcs
    assert 'separation' in freq_response.transfer_funcs
    assert np.array_equal(freq_response.transfer_funcs['separation'], H_s)


def test_calculate_correction_factor(freq_response):
    """Test correction factor calculation"""
    # Using np.trapz instead of integrate.simps since simps is deprecated
    correction = freq_response.calculate_correction_factor(
        stability=-0.5,
        integration_method='trapz'  # Use trapz method instead of simpson
    )

    assert isinstance(correction, float)
    assert correction >= 1.0  # Correction factor should be >= 1
    assert np.isfinite(correction)

    # Test that using different integration methods gives similar results
    correction_midpoint = freq_response.calculate_correction_factor(
        stability=-0.5,
        integration_method='midpoint'
    )
    assert np.isclose(correction, correction_midpoint, rtol=1e-2)


def test_spectral_analysis():
    """Test spectral analysis calculations"""
    # Generate test signal
    t = np.linspace(0, 10, 1000)
    freq = 1.0  # Hz
    signal = np.sin(2 * np.pi * freq * t)

    analyzer = SpectralAnalysis()
    freqs, power = analyzer.calculate_power_spectrum(signal, 100.0)

    assert len(freqs) == len(power)
    assert np.abs(freqs[np.argmax(power)] - freq) < 0.1
    assert np.all(power >= 0)


def test_invalid_stability(freq_response):
    """Test handling of invalid stability values"""
    with pytest.raises(ValueError, match=r".*Stability value must be between.*"):
        # Updated regex pattern to match actual error message
        freq_response.calculate_correction_factor(
            stability=-10.0,
            integration_method='trapz'
        )


@pytest.fixture
def invalid_sensor_specs():
    """Fixture for testing invalid sensor parameters"""
    return {
        'path_length': -1,  # Invalid negative length
        'separation_distance': 0.1,
        'time_constant': 0.1,
        'measuring_height': 3.0,
        'sampling_frequency': 10.0
    }


def test_invalid_params(invalid_sensor_specs):
    """Test handling of invalid parameters"""
    with pytest.raises(ValueError, match=r".*path_length must be positive.*"):
        SensorSpecs(**invalid_sensor_specs)


def test_invalid_integration_method(freq_response):
    """Test handling of invalid integration method"""
    with pytest.raises(ValueError, match=r".*Invalid integration method.*"):
        freq_response.calculate_correction_factor(
            stability=-0.5,
            integration_method='invalid'
        )


def test_total_transfer_function(freq_response):
    """Test total transfer function calculation"""
    H_total = freq_response.total_transfer_function()

    assert len(H_total) == 1000  # Default n_points
    assert np.all(H_total <= 1.0)  # Combined transfer function should be <= 1
    assert np.all(H_total >= 0.0)  # Combined transfer function should be >= 0
    assert H_total[0] == pytest.approx(1.0, rel=1e-4)  # Should be 1 at f=0 with relaxed tolerance

    # Should be product of individual transfer functions
    H_expected = (freq_response.block_averaging_transfer() *
                  freq_response.line_averaging_transfer() *
                  freq_response.time_response_transfer() *
                  freq_response.sensor_separation_transfer())
    assert np.allclose(H_total, H_expected, rtol=1e-4)
