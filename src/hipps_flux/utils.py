
"""
Utility functions for eddy covariance data processing.

This module provides helper functions for:
1. Unit conversions
2. Time series processing
3. Statistical calculations
4. Meteorological calculations
5. Data validation and quality checks
"""

from typing import Optional, Union, Tuple, List, Dict, Any
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats, signal

def detrend_timeseries(
    data: np.ndarray,
    method: str = 'linear',
    window_size: Optional[int] = None
) -> np.ndarray:
    """
    Detrend a time series using various methods.
    
    Args:
        data: Input time series
        method: Detrending method ('linear', 'mean', 'polynomial', 'moving')
        window_size: Window size for moving average (if method='moving')
        
    Returns:
        Detrended time series
    """
    if method == 'linear':
        x = np.arange(len(data))
        coeffs = np.polyfit(x[~np.isnan(data)], 
                          data[~np.isnan(data)], 1)
        trend = np.polyval(coeffs, x)
        return data - trend
        
    elif method == 'mean':
        return data - np.nanmean(data)
        
    elif method == 'polynomial':
        x = np.arange(len(data))
        coeffs = np.polyfit(x[~np.isnan(data)], 
                          data[~np.isnan(data)], 3)
        trend = np.polyval(coeffs, x)
        return data - trend
        
    elif method == 'moving':
        if window_size is None:
            window_size = len(data) // 10
        trend = pd.Series(data).rolling(
            window=window_size, center=True, min_periods=1).mean()
        return data - trend.values
        
    else:
        raise ValueError(f"Unknown detrending method: {method}")

def moving_window_stats(
    data: np.ndarray,
    window_size: int,
    stats_list: List[str] = ['mean', 'std']
) -> Dict[str, np.ndarray]:
    """
    Calculate statistics in moving windows.
    
    Args:
        data: Input time series
        window_size: Size of moving window
        stats_list: List of statistics to calculate
        
    Returns:
        Dictionary of calculated statistics arrays
    """
    results = {}
    series = pd.Series(data)
    
    for stat in stats_list:
        if stat == 'mean':
            results[stat] = series.rolling(
                window=window_size, center=True, min_periods=1).mean().values
        elif stat == 'std':
            results[stat] = series.rolling(
                window=window_size, center=True, min_periods=1).std().values
        elif stat == 'var':
            results[stat] = series.rolling(
                window=window_size, center=True, min_periods=1).var().values
        elif stat == 'median':
            results[stat] = series.rolling(
                window=window_size, center=True, min_periods=1).median().values
        elif stat == 'skew':
            results[stat] = series.rolling(
                window=window_size, center=True, min_periods=1).skew().values
        elif stat == 'kurt':
            results[stat] = series.rolling(
                window=window_size, center=True, min_periods=1).kurt().values
            
    return results

def unit_conversion(
    value: Union[float, np.ndarray],
    from_unit: str,
    to_unit: str
) -> Union[float, np.ndarray]:
    """
    Convert between different units commonly used in EC measurements.
    
    Args:
        value: Value(s) to convert
        from_unit: Original unit
        to_unit: Target unit
        
    Returns:
        Converted value(s)
    """
    # Define conversion factors
    conversions = {
        'umol/m2/s_to_mg/m2/s': {
            'co2': 44.01 / 1000,  # CO2 molecular weight
            'h2o': 18.02 / 1000,  # H2O molecular weight
        },
        'K_to_C': -273.15,
        'C_to_K': 273.15,
        'kPa_to_Pa': 1000,
        'Pa_to_kPa': 0.001,
        'mm/hr_to_m/s': 1 / (3600 * 1000),
        'm/s_to_mm/hr': 3600 * 1000
    }
    
    conversion_key = f"{from_unit}_to_{to_unit}"
    
    if conversion_key not in conversions:
        raise ValueError(f"Unsupported unit conversion: {conversion_key}")
        
    factor = conversions[conversion_key]
    
    if isinstance(factor, dict):
        raise ValueError("Please specify gas type for molar conversions")
        
    return value + factor if '_to_K' in conversion_key or '_to_C' in conversion_key else value * factor

def check_timestamps(
    timestamps: np.ndarray,
    freq: str = '10min'
) -> Tuple[bool, Optional[List[datetime]]]:
    """
    Check for gaps and duplicates in timestamp series.
    
    Args:
        timestamps: Array of timestamps
        freq: Expected frequency
        
    Returns:
        Tuple containing:
        - bool: True if timestamps are valid
        - List: Missing timestamps if any
    """
    # Convert to pandas datetime if not already
    if not isinstance(timestamps[0], (datetime, pd.Timestamp)):
        timestamps = pd.to_datetime(timestamps)
        
    # Create ideal timestamp series
    ideal_range = pd.date_range(
        start=timestamps[0],
        end=timestamps[-1],
        freq=freq
    )
    
    # Find missing timestamps
    missing = ideal_range.difference(timestamps)
    
    # Check for duplicates
    has_duplicates = len(timestamps) != len(set(timestamps))
    
    return (len(missing) == 0 and not has_duplicates, 
            missing.tolist() if len(missing) > 0 else None)

def despike_data(
    data: np.ndarray,
    threshold: float = 3.5,
    window_size: int = 100,
    method: str = 'mad'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove spikes from time series data.
    
    Args:
        data: Input time series
        threshold: Threshold for spike detection
        window_size: Size of moving window
        method: Despiking method ('mad' or 'zscore')
        
    Returns:
        Tuple containing:
        - Despiked data array
        - Boolean mask of identified spikes
    """
    despiked = data.copy()
    
    if method == 'mad':
        # Median Absolute Deviation method
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_zscore = 0.6745 * (data - median) / mad
        spikes = np.abs(modified_zscore) > threshold
        
    elif method == 'zscore':
        # Moving window z-score method
        spikes = np.zeros_like(data, dtype=bool)
        
        for i in range(len(data)):
            start = max(0, i - window_size//2)
            end = min(len(data), i + window_size//2)
            window = data[start:end]
            
            z_score = (data[i] - np.mean(window)) / np.std(window)
            spikes[i] = abs(z_score) > threshold
            
    else:
        raise ValueError(f"Unknown despiking method: {method}")
        
    # Replace spikes with interpolated values
    despiked[spikes] = np.interp(
        np.where(spikes)[0],
        np.where(~spikes)[0],
        data[~spikes]
    )
    
    return despiked, spikes

def time_lag_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 50
) -> Tuple[int, float]:
    """
    Calculate time lag between two signals using cross-correlation.
    
    Args:
        x: First time series
        y: Second time series
        max_lag: Maximum lag to consider
        
    Returns:
        Tuple containing:
        - Optimal lag
        - Maximum correlation coefficient
    """
    # Remove means
    x = x - np.mean(x)
    y = y - np.mean(y)
    
    # Calculate cross-correlation
    correlation = signal.correlate(x, y, mode='full')
    lags = signal.correlation_lags(len(x), len(y))
    
    # Find maximum correlation within max_lag
    valid_lags = np.abs(lags) <= max_lag
    max_idx = np.argmax(np.abs(correlation[valid_lags]))
    
    return (lags[valid_lags][max_idx], 
            correlation[valid_lags][max_idx] / (len(x) * np.std(x) * np.std(y)))

def calculate_turbulence_statistics(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray
) -> Dict[str, float]:
    """
    Calculate basic turbulence statistics.
    
    Args:
        u: Streamwise velocity component
        v: Crosswind velocity component
        w: Vertical velocity component
        
    Returns:
        Dictionary of turbulence statistics
    """
    # Calculate means
    u_mean = np.mean(u)
    v_mean = np.mean(v)
    w_mean = np.mean(w)
    
    # Calculate fluctuations
    u_prime = u - u_mean
    v_prime = v - v_mean
    w_prime = w - w_mean
    
    # Calculate variances
    u_var = np.mean(u_prime**2)
    v_var = np.mean(v_prime**2)
    w_var = np.mean(w_prime**2)
    
    # Calculate TKE
    tke = 0.5 * (u_var + v_var + w_var)
    
    # Calculate covariances
    uw_cov = np.mean(u_prime * w_prime)
    vw_cov = np.mean(v_prime * w_prime)
    
    # Calculate friction velocity
    u_star = np.sqrt(np.sqrt(uw_cov**2 + vw_cov**2))
    
    return {
        'u_mean': u_mean,
        'v_mean': v_mean,
        'w_mean': w_mean,
        'u_std': np.sqrt(u_var),
        'v_std': np.sqrt(v_var),
        'w_std': np.sqrt(w_var),
        'tke': tke,
        'uw_cov': uw_cov,
        'vw_cov': vw_cov,
        'u_star': u_star
    }

def validate_data_range(
    data: np.ndarray,
    variable: str,
    qc_limits: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Check if data falls within physically reasonable limits.
    
    Args:
        data: Input data array
        variable: Variable name
        qc_limits: Dictionary of min/max limits
        
    Returns:
        Tuple containing:
        - Mask of valid data
        - Dictionary with count of different types of invalid data
    """
    if qc_limits is None:
        qc_limits = {
            'wind_speed': (-30, 30),
            'temperature': (-40, 50),
            'co2': (200, 900),
            'h2o': (0, 40),
            'pressure': (80, 110)
        }
        
    if variable not in qc_limits:
        raise ValueError(f"No QC limits defined for variable: {variable}")
        
    min_val, max_val = qc_limits[variable]
    
    # Initialize masks
    valid_range = (data >= min_val) & (data <= max_val)
    is_finite = np.isfinite(data)
    
    # Count different types of invalid data
    invalid_counts = {
        'missing': np.sum(~is_finite),
        'below_range': np.sum((data < min_val) & is_finite),
        'above_range': np.sum((data > max_val) & is_finite),
        'total_invalid': np.sum(~valid_range | ~is_finite)
    }
    
    return valid_range & is_finite, invalid_counts

def interpolate_gaps(
    data: np.ndarray,
    max_gap: int = 3,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate gaps in time series data.
    
    Args:
        data: Input data array
        max_gap: Maximum gap size to interpolate
        method: Interpolation method
        
    Returns:
        Tuple containing:
        - Interpolated data array
        - Boolean mask of interpolated values
    """
    # Find gaps
    is_missing = ~np.isfinite(data)
    
    # Create mask of gaps to interpolate
    interpolate_mask = np.zeros_like(is_missing)
    
    # Find contiguous regions of missing data
    gaps = np.where(is_missing)[0]
    if len(gaps) == 0:
        return data, interpolate_mask
        
    # Group gaps into consecutive sequences
    gap_groups = np.split(gaps, np.where(np.diff(gaps) > 1)[0] + 1)
    
    # Interpolate gaps smaller than max_gap
    interpolated = data.copy()
    
    for gap in gap_groups:
        if len(gap) <= max_gap:
            interpolate_mask[gap] = True
            
    # Perform interpolation
    if method == 'linear':
        valid = ~is_missing
        interp_indices = np.where(interpolate_mask)[0]
        interpolated[interp_indices] = np.interp(
            interp_indices,
            np.where(valid)[0],
            data[valid]
        )
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
        
    return interpolated, interpolate_mask
