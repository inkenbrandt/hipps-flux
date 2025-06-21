"""
Data quality assessment for eddy covariance measurements.

This module implements data quality checks based on:
1. Steady state tests
2. Integral turbulence characteristics (ITC)
3. Wind direction relative to CSAT orientation
4. Statistical tests and outlier detection

References:
    Foken et al. (2004) Handbook of Micrometeorology
    Foken et al. (2012) Eddy Covariance: A Practical Guide
"""

from typing import Tuple, Optional, Union, Dict
import numpy as np
from dataclasses import dataclass
from enum import IntEnum

class QualityFlag(IntEnum):
    """Quality classification flags following Foken et al. (2004)"""
    CLASS_1 = 1  # Highest quality
    CLASS_2 = 2  # Good quality
    CLASS_3 = 3  # Moderate quality, usable
    CLASS_4 = 4  # Low quality, conditionally usable
    CLASS_5 = 5  # Poor quality, storage terms needed
    CLASS_6 = 6  # Poor quality, flux correction needed
    CLASS_7 = 7  # Poor quality, used for empirical relationships
    CLASS_8 = 8  # Poor quality, discarded in basic research
    CLASS_9 = 9  # Very poor quality, discarded

@dataclass
class StabilityParameters:
    """Parameters describing atmospheric stability conditions"""
    z: float  # Measurement height (m)
    L: float  # Obukhov length (m)
    u_star: float  # Friction velocity (m/s)
    sigma_w: float  # Standard deviation of vertical wind (m/s)
    sigma_T: float  # Standard deviation of temperature (K)
    T_star: float  # Temperature scale (K)
    latitude: float  # Site latitude (degrees)

@dataclass
class StationarityTest:
    """Results from stationarity test"""
    RN_uw: float  # Relative non-stationarity for momentum flux
    RN_wT: float  # Relative non-stationarity for sensible heat flux
    RN_wq: float  # Relative non-stationarity for latent heat flux
    RN_wc: float  # Relative non-stationarity for CO2 flux

class DataQuality:
    """
    Data quality assessment following Foken et al. (2004, 2012).
    
    Implements comprehensive quality control including:
    - Stationarity tests
    - Integral turbulence characteristics
    - Wind direction checks
    - Overall quality flags
    """
    
    def __init__(self, use_wind_direction: bool = True):
        """
        Initialize data quality assessment.
        
        Args:
            use_wind_direction: Whether to include wind direction in quality assessment
        """
        self.use_wind_direction = use_wind_direction
        
    def _calculate_integral_turbulence(
        self,
        stability: StabilityParameters
    ) -> Tuple[float, float]:
        """
        Calculate integral turbulence characteristics.
        
        Args:
            stability: StabilityParameters object
            
        Returns:
            Tuple containing:
            - ITC for momentum flux
            - ITC for scalar flux
        """
        z_L = stability.z / stability.L
        
        # Parameters depending on stability following Foken et al. (2004)
        if z_L <= -0.032:
            # Unstable conditions
            itc_w = 2.00 * abs(z_L)**0.125  # For vertical velocity
            itc_T = abs(z_L)**(-1/3)  # For temperature
            
        elif z_L <= 0.0:
            # Near-neutral unstable
            itc_w = 1.3
            itc_T = 0.5 * abs(z_L)**(-0.5)
            
        elif z_L < 0.4:
            # Near-neutral stable
            # Calculate Coriolis parameter
            f = 2 * 7.2921e-5 * np.sin(np.radians(stability.latitude))
            itc_w = 0.21 * np.log(abs(f)/stability.u_star) + 3.1
            itc_T = 1.4 * z_L**(-0.25)
            
        else:
            # Stable conditions
            itc_w = -(stability.sigma_w/stability.u_star)/9.1
            itc_T = -(stability.sigma_T/abs(stability.T_star))/9.1
            
        return itc_w, itc_T
        
    def _check_wind_direction(self, wind_direction: float) -> int:
        """
        Check wind direction relative to CSAT orientation.
        
        Args:
            wind_direction: Wind direction in degrees
            
        Returns:
            Quality class (1-9) based on wind direction
        """
        if not self.use_wind_direction:
            return QualityFlag.CLASS_1
            
        if (wind_direction < 151.0) or (wind_direction > 209.0):
            return QualityFlag.CLASS_1
        elif ((151.0 <= wind_direction < 171.0) or 
              (189.0 <= wind_direction <= 209.0)):
            return QualityFlag.CLASS_7
        else:  # 171.0 <= wind_direction <= 189.0
            return QualityFlag.CLASS_9
            
    def _evaluate_stationarity(
        self,
        stationarity: StationarityTest,
        flux_type: str
    ) -> int:
        """
        Evaluate stationarity test results.
        
        Args:
            stationarity: StationarityTest object
            flux_type: Type of flux ('momentum', 'heat', 'moisture', 'co2')
            
        Returns:
            Quality class (1-9) based on stationarity
        """
        # Get relevant RN value
        if flux_type == 'momentum':
            rn = stationarity.RN_uw
        elif flux_type == 'heat':
            rn = stationarity.RN_wT
        elif flux_type == 'moisture':
            rn = stationarity.RN_wq
        else:  # CO2
            rn = stationarity.RN_wc
            
        # Classify based on relative non-stationarity
        if rn < 0.16:
            return QualityFlag.CLASS_1
        elif rn < 0.31:
            return QualityFlag.CLASS_2
        elif rn < 0.76:
            return QualityFlag.CLASS_3
        elif rn < 1.01:
            return QualityFlag.CLASS_4
        elif rn < 2.51:
            return QualityFlag.CLASS_5
        elif rn < 10.0:
            return QualityFlag.CLASS_6
        else:
            return QualityFlag.CLASS_9
            
    def _evaluate_itc(
        self,
        measured: float,
        modeled: float
    ) -> int:
        """
        Evaluate integral turbulence characteristic test.
        
        Args:
            measured: Measured ITC
            modeled: Modeled ITC
            
        Returns:
            Quality class (1-9) based on ITC comparison
        """
        # Calculate relative difference
        itc_diff = abs((measured - modeled) / modeled)
        
        # Classify based on difference
        if itc_diff < 0.31:
            return QualityFlag.CLASS_1
        elif itc_diff < 0.76:
            return QualityFlag.CLASS_2
        elif itc_diff < 1.01:
            return QualityFlag.CLASS_3
        elif itc_diff < 2.51:
            return QualityFlag.CLASS_4
        elif itc_diff < 10.0:
            return QualityFlag.CLASS_5
        else:
            return QualityFlag.CLASS_9
            
    def assess_data_quality(
        self,
        stability: StabilityParameters,
        stationarity: StationarityTest,
        wind_direction: Optional[float] = None,
        flux_type: str = 'momentum'
    ) -> Dict[str, Union[int, float]]:
        """
        Perform comprehensive data quality assessment.
        
        Args:
            stability: StabilityParameters object
            stationarity: StationarityTest object
            wind_direction: Wind direction in degrees (optional)
            flux_type: Type of flux to assess ('momentum', 'heat', 'moisture', 'co2')
            
        Returns:
            Dictionary containing:
            - overall_flag: Final quality classification
            - stationarity_flag: Quality based on stationarity
            - itc_flag: Quality based on ITC
            - wind_dir_flag: Quality based on wind direction
            - itc_measured: Measured ITC value
            - itc_modeled: Modeled ITC value
        """
        # Calculate ITC
        itc_w, itc_T = self._calculate_integral_turbulence(stability)
        
        # Get measured ITC
        measured_itc = stability.sigma_w / stability.u_star
        if flux_type == 'momentum':
            modeled_itc = itc_w
        else:
            modeled_itc = itc_T
            
        # Evaluate individual tests
        station_flag = self._evaluate_stationarity(stationarity, flux_type)
        itc_flag = self._evaluate_itc(measured_itc, modeled_itc)
        wind_flag = (self._check_wind_direction(wind_direction) 
                    if wind_direction is not None else QualityFlag.CLASS_1)
        
        # Overall quality is worst of individual flags
        overall_flag = max(station_flag, itc_flag, wind_flag)
        
        return {
            'overall_flag': overall_flag,
            'stationarity_flag': station_flag,
            'itc_flag': itc_flag,
            'wind_dir_flag': wind_flag,
            'itc_measured': measured_itc,
            'itc_modeled': modeled_itc
        }
        
    def get_quality_label(self, flag: int) -> str:
        """Get descriptive label for quality flag."""
        labels = {
            1: "Highest quality",
            2: "Good quality",
            3: "Moderate quality",
            4: "Low quality",
            5: "Poor quality (storage)",
            6: "Poor quality (flux correction)",
            7: "Poor quality (empirical only)",
            8: "Poor quality (discard research)",
            9: "Very poor quality (discard)"
        }
        return labels.get(flag, "Unknown")

class OutlierDetection:
    """Statistical methods for detecting outliers in flux data."""
    
    @staticmethod
    def mad_outliers(
        data: np.ndarray,
        threshold: float = 3.5
    ) -> np.ndarray:
        """
        Detect outliers using Median Absolute Deviation (MAD) method.
        
        Args:
            data: Input data array
            threshold: Number of MADs for outlier threshold
            
        Returns:
            Boolean mask where True indicates outliers
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_zscore = 0.6745 * (data - median) / mad
        return np.abs(modified_zscore) > threshold
        
    @staticmethod
    def spike_detection(
        data: np.ndarray,
        window_size: int = 100,
        z_threshold: float = 4.0
    ) -> np.ndarray:
        """
        Detect spikes using moving window statistics.
        
        Args:
            data: Input data array
            window_size: Size of moving window
            z_threshold: Z-score threshold for spike detection
            
        Returns:
            Boolean mask where True indicates spikes
        """
        spikes = np.zeros_like(data, dtype=bool)
        
        for i in range(len(data)):
            # Get window indices
            start = max(0, i - window_size//2)
            end = min(len(data), i + window_size//2)
            
            # Calculate statistics for window
            window = data[start:end]
            mean = np.mean(window)
            std = np.std(window)
            
            # Check if point is spike
            if std > 0:  # Avoid division by zero
                z_score = abs(data[i] - mean) / std
                spikes[i] = z_score > z_threshold
                
        return spikes

def quality_filter(
    data: np.ndarray,
    quality_flags: np.ndarray,
    min_quality: int = 3
) -> np.ndarray:
    """
    Filter data based on quality flags.
    
    Args:
        data: Input data array
        quality_flags: Array of quality flags
        min_quality: Minimum acceptable quality class
        
    Returns:
        Filtered data array with lower quality data set to NaN
    """
    filtered = data.copy()
    filtered[quality_flags > min_quality] = np.nan
    return filtered
