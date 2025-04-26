"""
Boundary layer calculations including planetary boundary layer height and temperature/density relationships.

This module implements the algorithms from Kljun et al. (2004, 2015) for planetary boundary layer 
height calculations and includes functions for temperature and water vapor conversions in the boundary layer.

References:
    Kljun et al. (2004) A Simple Parameterisation for Flux Footprint Predictions
    Kljun et al. (2015) A simple two-dimensional parameterisation for Flux Footprint Prediction (FFP)
"""

from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .constants import (
    RD,  # Gas constant for dry air
    RV,  # Gas constant for water vapor
    T_0C_K,  # 0°C in Kelvin
)

@dataclass
class BoundaryLayerParams:
    """Parameters describing boundary layer conditions"""
    height: float  # Planetary boundary layer height (m)
    temperature: float  # Air temperature (K)
    pressure: float  # Atmospheric pressure (kPa)
    h2o_density: float  # Water vapor density (g/m^3)

def planetary_boundary_layer_height(obukhov: float) -> Optional[float]:
    """
    Calculate planetary boundary layer height using Kljun et al. (2004, 2015) method.
    
    Args:
        obukhov: Obukhov length (m)
        
    Returns:
        Optional[float]: Calculated PBL height (m) or None if input is invalid
        
    Notes:
        The PBL height is calculated based on stability conditions indicated by the Obukhov length:
        - Unstable conditions: obukhov < 0
        - Stable conditions: obukhov > 0
        
        For unstable conditions:
        - Minimum height is 1000m when obukhov ≤ -1013.3
        - Maximum height is 2000m near neutral conditions
        
        For stable conditions:
        - Minimum height is 200m for weak stability
        - Maximum height is 1000m for strong stability
    """
    if not isinstance(obukhov, (int, float)):
        return None
        
    if np.isnan(obukhov):
        return None
        
    # Unstable conditions
    if obukhov < 0:
        if obukhov < -1013.3:
            return 1000.0  # minimum PBL height 
        elif obukhov <= -650:
            return 1200.0 - 200.0 * ((obukhov + 650.0) / (-1013.3 + 650.0))
        elif obukhov <= -30.0:
            return 1500.0 - 300.0 * ((obukhov + 30.0) / (-650.0 + 30.0))
        elif obukhov <= -5.0:
            return 2000.0 - 500.0 * ((obukhov + 5.0) / (-30.0 + 5.0))
        else:  # obukhov <= 0.0
            return 2000.0 + 20.0 * (obukhov + 5.0)
            
    # Stable conditions
    else:
        if obukhov > 1316.4:
            return 1000.0
        elif obukhov >= 1000.0:
            return 800.0 + 200.0 * ((obukhov - 1000.0) / (1316.4 - 1000.0))
        elif obukhov >= 130.0:
            return 250.0 + 550.0 * ((obukhov - 130.0) / (1000.0 - 130.0))
        elif obukhov >= 84.0:
            return 200.0 + 50.0 * ((obukhov - 84.0) / (130.0 - 84.0))
        else:  # obukhov > 0.0
            return 200.0 - (84.0 - obukhov) * (50.0/46.0)

def air_temperature_from_sonic(
    sonic_temp: float,
    h2o_density: float,
    pressure: float
) -> Optional[float]:
    """
    Calculate air temperature from sonic temperature, water vapor density, and pressure.
    
    Implementation of equation (14) for air temperature from sonic temperature.
    Uses specific heat capacities:
    - Cpd = 1004, Cvd = 717 (dry air)
    - Cpw = 1952, Cvw = 1463 (water vapor) in J deg^-1 kg^-1
    
    Args:
        sonic_temp: Sonic temperature (K)
        h2o_density: Water vapor density (g/m^3)
        pressure: Atmospheric pressure (kPa)
    
    Returns:
        Optional[float]: Air temperature (K) or None if inputs are invalid
    """
    if any(np.isnan([sonic_temp, h2o_density, pressure])):
        return None
        
    # Constants used in calculation
    cvw_cvd_plus1 = 3.040446  # Cvw/Cvd + 1
    cvw_cvd_minus1 = 1.040446  # Cvw/Cvd - 1
    cpw_cpv_factor = 1.696000  # 2*(2*Cpw/Cpv - Cvw/Cvd - 1)
    cpw_cpd = 1.944223  # Cpw/Cpd
    cvw_cvd = 2.040446  # Cvw/Cvd
    
    # Intermediate calculations
    t_c1 = pressure + (2*RV - cvw_cvd_plus1*RD)*h2o_density*sonic_temp
    
    t_c2 = (pressure**2 + 
            (cvw_cvd_minus1*RD*h2o_density*sonic_temp)**2 + 
            cpw_cpv_factor*RD*h2o_density*pressure*sonic_temp)
    
    t_c3 = 2*h2o_density*((RV - cpw_cpd*RD) + 
                          (RV - RD)*(RV - cvw_cvd*RD)*h2o_density*sonic_temp/pressure)
    
    # Final calculation
    try:
        return (t_c1 - np.sqrt(t_c2))/t_c3
    except (ValueError, RuntimeWarning):
        return None

def calculate_air_density(
    temperature: float,
    pressure: float,
    h2o_density: float
) -> Tuple[float, float, float]:
    """
    Calculate dry air density, water vapor pressure, and moist air density.
    
    Args:
        temperature: Air temperature (K)
        pressure: Atmospheric pressure (kPa)
        h2o_density: Water vapor density (g/m^3)
    
    Returns:
        Tuple containing:
        - e_air: Water vapor pressure (kPa)
        - rho_d: Dry air density (g/m^3)
        - rho_a: Moist air density (kg/m^3)
    """
    # Calculate water vapor pressure using ideal gas law
    e_air = h2o_density * RV * temperature
    
    # Calculate dry air density
    rho_d = (pressure - e_air)/(temperature * RD)
    
    # Calculate moist air density (convert to kg/m^3)
    rho_a = (rho_d + h2o_density)/1000.0
    
    return e_air, rho_d, rho_a

def calculate_saturation_vapor_pressure(
    temperature: float,
    pressure: float
) -> Tuple[float, float]:
    """
    Calculate saturation vapor pressure and enhancement factor.
    
    Args:
        temperature: Air temperature (°C)
        pressure: Atmospheric pressure (kPa)
        
    Returns:
        Tuple containing:
        - e_sat: Saturation vapor pressure (kPa)
        - enhance_factor: Enhancement factor (dimensionless)
    """
    # Calculate enhancement factor
    enhance_factor = (1.00041 + 
                     pressure*(3.48e-5 + 
                             7.4e-9*(temperature + 30.6 - 0.38*pressure)**2))
    
    # Calculate saturation vapor pressure based on temperature range
    if temperature >= 0:
        e_sat = (0.61121 * enhance_factor * 
                 np.exp((17.368 * temperature)/(temperature + 238.88)))
    else:
        e_sat = (0.61121 * enhance_factor * 
                 np.exp((17.966 * temperature)/(temperature + 247.15)))
        
    return e_sat, enhance_factor

def calculate_dewpoint_temperature(
    e_air: float,
    pressure: float,
    enhance_factor: Optional[float] = None
) -> float:
    """
    Calculate dew point temperature using enhanced vapor pressure.
    
    Args:
        e_air: Water vapor pressure (kPa)
        pressure: Atmospheric pressure (kPa)
        enhance_factor: Optional enhancement factor. If None, will be calculated.
        
    Returns:
        float: Dew point temperature (°C)
    """
    if enhance_factor is None:
        enhance_factor = 1.00072 + 3.46e-5 * pressure
    
    x_tmp = np.log(e_air/(0.61121 * enhance_factor))
    t_dp_first = 240.97 * x_tmp/(17.502 - x_tmp)
    
    # Recalculate enhancement factor with first dew point estimate
    enhance_factor = (1.00041 + 
                     pressure*(3.48e-5 + 
                             7.4e-9*(t_dp_first + 30.6 - 0.38*pressure)**2))
    
    x_tmp = np.log(e_air/(0.61121 * enhance_factor))
    
    # Final dew point calculation based on temperature range
    if t_dp_first >= 0:
        return 238.88 * x_tmp/(17.368 - x_tmp)
    else:
        return 247.15 * x_tmp/(17.966 - x_tmp)

class BoundaryLayerProcessor:
    """
    Process and analyze boundary layer measurements.
    
    This class provides methods for processing raw measurements into derived 
    boundary layer parameters and properties.
    """
    
    def __init__(self, params: BoundaryLayerParams):
        """
        Initialize with boundary layer parameters.
        
        Args:
            params: BoundaryLayerParams object containing measurement data
        """
        self.params = params
        self._validate_params()
        
    def _validate_params(self) -> None:
        """Validate input parameters."""
        if self.params.height <= 0:
            raise ValueError("Boundary layer height must be positive")
        if self.params.temperature < 0:
            raise ValueError("Temperature must be ≥ 0 K")
        if self.params.pressure <= 0:
            raise ValueError("Pressure must be positive")
        if self.params.h2o_density < 0:
            raise ValueError("Water vapor density cannot be negative")
            
    def process_measurements(self) -> dict:
        """
        Process raw measurements into derived quantities.
        
        Returns:
            dict: Dictionary containing processed values including:
                - air_temperature (K)
                - vapor_pressure (kPa)
                - saturation_vapor_pressure (kPa)
                - relative_humidity (%)
                - dewpoint_temperature (°C)
                - dry_air_density (g/m^3)
                - moist_air_density (kg/m^3)
        """
        # Convert absolute temperature to Celsius for some calculations
        temp_c = self.params.temperature - T_0C_K
        
        # Calculate vapor pressures
        e_air, rho_d, rho_a = calculate_air_density(
            self.params.temperature,
            self.params.pressure,
            self.params.h2o_density
        )
        
        e_sat, enhance_factor = calculate_saturation_vapor_pressure(
            temp_c,
            self.params.pressure
        )
        
        # Calculate relative humidity
        rh = 100.0 * e_air/e_sat
        
        # Calculate dew point
        t_dp = calculate_dewpoint_temperature(
            e_air,
            self.params.pressure,
            enhance_factor
        )
        
        return {
            'air_temperature': self.params.temperature,
            'vapor_pressure': e_air,
            'saturation_vapor_pressure': e_sat,
            'relative_humidity': rh,
            'dewpoint_temperature': t_dp,
            'dry_air_density': rho_d,
            'moist_air_density': rho_a
        }
