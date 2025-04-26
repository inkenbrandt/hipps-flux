```python
"""
Frequency response corrections for eddy covariance measurements.

This module implements frequency response corrections including:
1. Block averaging
2. Line averaging
3. Time response
4. Sensor separation
5. Path averaging
6. High/low pass filtering

References:
    Moore (1986) Frequency response corrections for eddy correlation systems
    Massman (2000) A simple method for estimating frequency response corrections
    Moncrieff et al. (1997) A system to measure surface fluxes of momentum...
"""

from typing import Optional, Tuple, List, Union
import numpy as np
from dataclasses import dataclass
from scipy import integrate

@dataclass
class SensorSpecs:
    """Sensor specifications for frequency response calculations"""
    path_length: float  # Sensor path length (m)
    separation_distance: float  # Sensor separation distance (m)
    time_constant: float  # Sensor time constant (s)
    measuring_height: float  # Measurement height (m)
    sampling_frequency: float  # Sampling frequency (Hz)
    
@dataclass
class AtmConditions:
    """Atmospheric conditions affecting frequency response"""
    wind_speed: float  # Mean wind speed (m/s)
    stability: float  # Atmospheric stability (z/L)
    temp: float  # Air temperature (K)
    pressure: float  # Air pressure (kPa)

class FrequencyResponse:
    """
    Calculate frequency response corrections following Moore (1986).
    """
    
    def __init__(
        self,
        sensor_specs: SensorSpecs,
        atm_conditions: AtmConditions
    ):
        """
        Initialize frequency response calculations.
        
        Args:
            sensor_specs: Sensor specifications
            atm_conditions: Atmospheric conditions
        """
        self.specs = sensor_specs
        self.atm = atm_conditions
        
        # Derived parameters
        self.f_nyquist = self.specs.sampling_frequency / 2
        self.normalized_freq = None  # Will store normalized frequencies
        self.transfer_funcs = {}  # Will store transfer functions
        
    def _calculate_normalized_frequency(self, n_points: int = 1000) -> np.ndarray:
        """
        Calculate normalized frequency array.
        
        Args:
            n_points: Number of frequency points
            
        Returns:
            Array of normalized frequencies
        """
        # Create frequency array from 0.001 to Nyquist
        f = np.logspace(-3, np.log10(self.f_nyquist), n_points)
        
        # Normalize by measurement height and wind speed
        self.normalized_freq = f * self.specs.measuring_height / self.atm.wind_speed
        return self.normalized_freq
        
    def block_averaging_transfer(self) -> np.ndarray:
        """
        Calculate transfer function for block averaging.
        
        Returns:
            Transfer function array
        """
        if self.normalized_freq is None:
            self._calculate_normalized_frequency()
            
        f = self.normalized_freq * self.atm.wind_speed / self.specs.measuring_height
        T = 1 / self.specs.sampling_frequency
        
        H_b = np.ones_like(f)
        nonzero = f > 0
        H_b[nonzero] = np.sin(np.pi * f[nonzero] * T) / (np.pi * f[nonzero] * T)
        H_b = H_b * H_b
        
        self.transfer_funcs['block'] = H_b
        return H_b
        
    def line_averaging_transfer(self) -> np.ndarray:
        """
        Calculate transfer function for line averaging.
        
        Returns:
            Transfer function array
        """
        if self.normalized_freq is None:
            self._calculate_normalized_frequency()
            
        f = self.normalized_freq * self.atm.wind_speed / self.specs.measuring_height
        
        # Path averaging parameter
        peak = 2 * np.pi * f * self.specs.path_length / self.atm.wind_speed
        
        H_l = np.ones_like(f)
        nonzero = peak > 0.01
        
        # Different formulations for vertical and horizontal wind components
        # Here implementing for vertical component (eq. 9 in Moore 1986)
        H_l[nonzero] = (4 / peak[nonzero]) * (
            1 + (peak[nonzero] + 3) / (2 * peak[nonzero] * np.exp(peak[nonzero])) -
            3 / (2 * peak[nonzero])
        )
        
        self.transfer_funcs['line'] = H_l
        return H_l
        
    def time_response_transfer(self) -> np.ndarray:
        """
        Calculate transfer function for sensor time response.
        
        Returns:
            Transfer function array
        """
        if self.normalized_freq is None:
            self._calculate_normalized_frequency()
            
        f = self.normalized_freq * self.atm.wind_speed / self.specs.measuring_height
        
        # First-order response
        H_r = 1 / (1 + (2 * np.pi * f * self.specs.time_constant)**2)
        
        self.transfer_funcs['time'] = H_r
        return H_r
        
    def sensor_separation_transfer(self) -> np.ndarray:
        """
        Calculate transfer function for sensor separation.
        
        Returns:
            Transfer function array
        """
        if self.normalized_freq is None:
            self._calculate_normalized_frequency()
            
        # Lateral separation parameter
        d_u = self.specs.separation_distance / self.atm.wind_speed
        
        # Exponential decay following Horst (1997)
        H_s = np.exp(-9.9 * (d_u * self.normalized_freq * 
                            self.atm.wind_speed / self.specs.measuring_height)**1.5)
        
        self.transfer_funcs['separation'] = H_s
        return H_s
        
    def total_transfer_function(self) -> np.ndarray:
        """
        Calculate total system transfer function.
        
        Returns:
            Combined transfer function array
        """
        H_total = (self.block_averaging_transfer() *
                  self.line_averaging_transfer() *
                  self.time_response_transfer() *
                  self.sensor_separation_transfer())
        
        return H_total
        
    def _cospectral_model(self, stability: float) -> np.ndarray:
        """
        Calculate theoretical cospectrum model.
        
        Args:
            stability: Atmospheric stability (z/L)
            
        Returns:
            Model cospectrum array
        """
        n = self.normalized_freq
        
        if stability > 0:  # Stable conditions
            A = 0.284 * (1 + 6.4 * stability)**0.75
            B = 9.345 * (1 + 6.4 * stability)**-0.825
            Co = n / (A + B * n**2.1)
            
        else:  # Unstable conditions
            n_mask = n < 0.54
            Co = np.zeros_like(n)
            
            # Different formulations for low and high frequencies
            Co[n_mask] = 12.92 * n[n_mask] / (1 + 26.7 * n[n_mask])**1.375
            Co[~n_mask] = 4.378 * n[~n_mask] / (1 + 3.8 * n[~n_mask])**2.4
            
        return Co
        
    def calculate_correction_factor(
        self,
        stability: float,
        integration_method: str = 'simpson'
    ) -> float:
        """
        Calculate frequency response correction factor.
        
        Args:
            stability: Atmospheric stability (z/L)
            integration_method: Numerical integration method ('simpson' or 'trapz')
            
        Returns:
            Correction factor (ratio of true to measured flux)
        """
        # Get model cospectrum and transfer function
        Co = self._cospectral_model(stability)
        H = self.total_transfer_function()
        
        # Integration bounds
        f_min = self.normalized_freq[0]
        f_max = self.normalized_freq[-1]
        
        if integration_method == 'simpson':
            numerator = integrate.simps(Co, x=self.normalized_freq)
            denominator = integrate.simps(Co * H, x=self.normalized_freq)
        else:
            numerator = np.trapz(Co, x=self.normalized_freq)
            denominator = np.trapz(Co * H, x=self.normalized_freq)
            
        return numerator / denominator

class SpectralAnalysis:
    """
    Spectral analysis tools for eddy covariance data.
    """
    
    @staticmethod
    def calculate_power_spectrum(
        data: np.ndarray,
        sampling_freq: float,
        window: str = 'hanning'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate power spectrum of time series.
        
        Args:
            data: Input time series
            sampling_freq: Sampling frequency (Hz)
            window: Window function ('hanning', 'hamming', 'blackman')
            
        Returns:
            Tuple containing:
            - Frequency array
            - Power spectrum array
        """
        # Remove mean
        data = data - np.mean(data)
        
        # Apply window
        if window == 'hanning':
            window_func = np.hanning(len(data))
        elif window == 'hamming':
            window_func = np.hamming(len(data))
        elif window == 'blackman':
            window_func = np.blackman(len(data))
        else:
            window_func = np.ones_like(data)
            
        windowed_data = data * window_func
        
        # Calculate FFT
        fft = np.fft.fft(windowed_data)
        
        # Get frequencies
        freqs = np.fft.fftfreq(len(data), d=1/sampling_freq)
        
        # Get power spectrum (positive frequencies only)
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        power = np.abs(fft[pos_mask])**2
        
        return freqs, power
        
    @staticmethod
    def calculate_cospectra(
        data1: np.ndarray,
        data2: np.ndarray,
        sampling_freq: float,
        window: str = 'hanning'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate co-spectrum between two time series.
        
        Args:
            data1: First time series
            data2: Second time series
            sampling_freq: Sampling frequency (Hz)
            window: Window function name
            
        Returns:
            Tuple containing:
            - Frequency array
            - Co-spectrum array
        """
        # Remove means
        data1 = data1 - np.mean(data1)
        data2 = data2 - np.mean(data2)
        
        # Apply window
        if window == 'hanning':
            window_func = np.hanning(len(data1))
        elif window == 'hamming':
            window_func = np.hamming(len(data1))
        elif window == 'blackman':
            window_func = np.blackman(len(data1))
        else:
            window_func = np.ones_like(data1)
            
        windowed1 = data1 * window_func
        windowed2 = data2 * window_func
        
        # Calculate FFTs
        fft1 = np.fft.fft(windowed1)
        fft2 = np.fft.fft(windowed2)
        
        # Get frequencies
        freqs = np.fft.fftfreq(len(data1), d=1/sampling_freq)
        
        # Calculate co-spectrum (real part of cross-spectrum)
        co_spec = np.real(fft1 * np.conj(fft2))
        
        # Get positive frequencies only
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        co_spec = co_spec[pos_mask]
        
        return freqs, co_spec

def apply_frequency_correction(
    flux: float,
    correction_factor: float
) -> float:
    """
    Apply frequency response correction to flux.
    
    Args:
        flux: Uncorrected flux
        correction_factor: Correction factor
        
    Returns:
        Corrected flux
    """
    return flux * correction_factor
```
