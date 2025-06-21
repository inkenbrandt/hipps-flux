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

from hipps_flux.constants import *

#    RD,  # Gas constant for dry air
#    RV,  # Gas constant for water vapor
#    T_0C_K,  # 0°C in Kelvin
#

# Gas constants
RD = R_SPECIFIC["dry_air"]  # Specific gas constant for dry air (J/kg/K)
RV = R_SPECIFIC["water_vapor"]  # Specific gas constant for water vapor (J/kg/K)
T_0C_K = T_ZERO_C

from typing import Optional, Tuple, Union
import numpy as np


def calculate_air_temperature(
    sonic_temp: Union[float, np.ndarray],
    h2o_density: Union[float, np.ndarray],
    pressure: Union[float, np.ndarray],
    Rd: float = 287.04,
    Rv: float = 461.5,
) -> Union[float, np.ndarray]:
    """Calculate air temperature with array support"""
    # Input validation
    # Handle scalar and array inputs
    inputs = [sonic_temp, h2o_density, pressure]
    if any(isinstance(x, np.ndarray) for x in inputs):
        # Convert scalars to arrays if needed
        inputs = [np.asarray(x) if not isinstance(x, np.ndarray) else x for x in inputs]
        sonic_temp, h2o_density, pressure = inputs

        # Check for NaN values
        if (
            np.any(np.isnan(sonic_temp))
            or np.any(np.isnan(h2o_density))
            or np.any(np.isnan(pressure))
        ):
            return None
    else:
        if any(np.isnan([sonic_temp, h2o_density, pressure])):
            return None

    if any(x <= 0 for x in [sonic_temp, pressure]):
        return None

    if h2o_density < 0:
        return None

    try:
        # Convert units
        h2o_density_kgm3 = h2o_density / 1000.0  # g/m³ to kg/m³
        pressure_pa = pressure * 1000.0  # kPa to Pa

        # Calculate water vapor pressure using ideal gas law
        # e = ρRvT
        e = h2o_density_kgm3 * Rv * sonic_temp  # Pa

        # Calculate air temperature using Kaimal and Gaynor formulation
        correction_factor = 0.32 * e / pressure_pa
        air_temp = sonic_temp / (1 + correction_factor)

        # Verify result is physical
        if not (173.15 < air_temp < 373.15):  # -100°C to 100°C in K
            return None

        return air_temp

    except (ValueError, RuntimeWarning, ZeroDivisionError):
        return None


@dataclass
class BoundaryLayerParams:
    """Parameters describing boundary layer conditions"""

    height: float  # Planetary boundary layer height (m)
    temperature: float  # Air temperature (K)
    pressure: float  # Atmospheric pressure (kPa)
    h2o_density: float  # Water vapor density (g/m^3)


def planetary_boundary_layer_height(
    obukhov: Union[float, np.ndarray],
) -> Union[Optional[float], np.ndarray]:
    """
    Calculate planetary boundary layer height using Kljun et al. (2004, 2015) method.
    Supports both single float values and numpy arrays as input.

    Args:
        obukhov: Obukhov length (m) - can be a single float or numpy array

    Returns:
        Union[Optional[float], np.ndarray]: Calculated PBL height (m)
        For single inputs: returns None if input is invalid
        For array inputs: invalid values are set to np.nan

    Notes:
        The PBL height is calculated based on stability conditions:
        - Unstable conditions: obukhov < 0
        - Stable conditions: obukhov > 0

        Special cases:
        - Very unstable (obukhov ≤ -1013.3): 1000m
        - Near neutral unstable (-15 to -0.1): 1980m to 2019m
        - Near neutral stable (0 to 50): 200m to 184.13m
        - Very stable (obukhov ≥ 1500): 1000m
    """
    # Convert input to numpy array if it isn't already
    is_scalar = np.isscalar(obukhov)
    obukhov_arr = np.asarray(obukhov)

    # Input validation
    if not np.issubdtype(obukhov_arr.dtype, np.number):
        return None if is_scalar else np.full_like(obukhov_arr, np.nan, dtype=float)

    # Initialize output array
    result = np.zeros_like(obukhov_arr, dtype=float)

    def linear_interpolation(
        x: np.ndarray, x1: float, x2: float, y1: float, y2: float
    ) -> np.ndarray:
        """Helper function for linear interpolation"""
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    # Handle unstable conditions (obukhov < 0)
    unstable = obukhov_arr < 0
    if np.any(unstable):
        # Very unstable
        mask = (obukhov_arr <= -1013.3) & unstable
        result[mask] = 1000.0

        # -1013.3 to -800.0
        mask = (-1013.3 < obukhov_arr) & (obukhov_arr <= -800.0) & unstable
        result[mask] = linear_interpolation(
            obukhov_arr[mask], -1013.3, -800.0, 1000.0, 1117.42
        )

        # -800.0 to -300.0
        mask = (-800.0 < obukhov_arr) & (obukhov_arr <= -300.0) & unstable
        result[mask] = linear_interpolation(
            obukhov_arr[mask], -800.0, -300.0, 1117.42, 1472.0
        )

        # -300.0 to -15.0
        mask = (-300.0 < obukhov_arr) & (obukhov_arr <= -15.0) & unstable
        result[mask] = linear_interpolation(
            obukhov_arr[mask], -300.0, -15.0, 1472.0, 1980.0
        )

        # -15.0 to -0.1
        mask = (-15.0 < obukhov_arr) & (obukhov_arr <= -0.1) & unstable
        result[mask] = linear_interpolation(
            obukhov_arr[mask], -15.0, -0.1, 1980.0, 2019.0
        )

        # -0.1 to 0.0
        mask = (-0.1 < obukhov_arr) & (obukhov_arr < 0) & unstable
        result[mask] = linear_interpolation(
            obukhov_arr[mask], -0.1, 0.0, 2019.0, 2000.0
        )

    # Handle stable conditions (obukhov >= 0)
    stable = obukhov_arr >= 0
    if np.any(stable):
        # Very stable
        mask = (obukhov_arr >= 1500.0) & stable
        result[mask] = 1000.0

        # 1100.0 to 1500.0
        mask = (1100.0 <= obukhov_arr) & (obukhov_arr < 1500.0) & stable
        result[mask] = linear_interpolation(
            obukhov_arr[mask], 1100.0, 1500.0, 843.29, 1000.0
        )

        # 500.0 to 1100.0
        mask = (500.0 <= obukhov_arr) & (obukhov_arr < 1100.0) & stable
        result[mask] = linear_interpolation(
            obukhov_arr[mask], 500.0, 1100.0, 432.18, 843.29
        )

        # 50.0 to 500.0
        mask = (50.0 <= obukhov_arr) & (obukhov_arr < 500.0) & stable
        result[mask] = linear_interpolation(
            obukhov_arr[mask], 50.0, 500.0, 184.13, 432.18
        )

        # 0.0 to 50.0
        mask = (0.0 <= obukhov_arr) & (obukhov_arr < 50.0) & stable
        result[mask] = linear_interpolation(obukhov_arr[mask], 0.0, 50.0, 200.0, 184.13)

    # Handle NaN values
    result[np.isnan(obukhov_arr)] = np.nan

    # Return scalar or array based on input type
    if is_scalar:
        return float(result.item()) if not np.isnan(result.item()) else None
    return result


def air_temperature_from_sonic(
    sonic_temp: Union[float, np.ndarray],
    h2o_density: Union[float, np.ndarray],
    pressure: Union[float, np.ndarray],
) -> Union[Optional[float], np.ndarray]:
    """
    Calculate air temperature from sonic temperature, water vapor density, and pressure.
    Supports both scalar values and numpy arrays as input.

    Implementation of equation (14) for air temperature from sonic temperature.
    Uses specific heat capacities:
    - Cpd = 1004, Cvd = 717 (dry air)
    - Cpw = 1952, Cvw = 1463 (water vapor) in J deg^-1 kg^-1

    Args:
        sonic_temp: Sonic temperature (K)
        h2o_density: Water vapor density (g/m^3)
        pressure: Atmospheric pressure (kPa)

    Returns:
        Union[Optional[float], np.ndarray]: Air temperature (K)
        For scalar inputs: returns None if inputs are invalid
        For array inputs: invalid values are set to np.nan
    """
    # Convert inputs to numpy arrays
    sonic_temp_arr = np.asarray(sonic_temp)
    h2o_density_arr = np.asarray(h2o_density)
    pressure_arr = np.asarray(pressure)

    # Check if inputs are scalar
    is_scalar = all(np.isscalar(x) for x in [sonic_temp, h2o_density, pressure])

    # Input validation for scalar case
    if is_scalar and any(np.isnan([sonic_temp, h2o_density, pressure])):
        return None

    # Constants used in calculation
    cvw_cvd_plus1 = 3.040446  # Cvw/Cvd + 1
    cvw_cvd_minus1 = 1.040446  # Cvw/Cvd - 1
    cpw_cpv_factor = 1.696000  # 2*(2*Cpw/Cpv - Cvw/Cvd - 1)
    cpw_cpd = 1.944223  # Cpw/Cpd
    cvw_cvd = 2.040446  # Cvw/Cvd

    # Broadcast arrays to same shape if needed
    try:
        # This will raise ValueError if shapes are incompatible
        broadcast_shape = np.broadcast_shapes(
            sonic_temp_arr.shape, h2o_density_arr.shape, pressure_arr.shape
        )
        sonic_temp_arr = np.broadcast_to(sonic_temp_arr, broadcast_shape)
        h2o_density_arr = np.broadcast_to(h2o_density_arr, broadcast_shape)
        pressure_arr = np.broadcast_to(pressure_arr, broadcast_shape)
    except ValueError:
        if is_scalar:
            return None
        return np.full_like(sonic_temp_arr, np.nan, dtype=float)

    # Initialize output array with NaN values
    result = np.full(broadcast_shape, np.nan, dtype=float)

    # Create mask for valid calculations (non-NaN values)
    valid_mask = ~(
        np.isnan(sonic_temp_arr) | np.isnan(h2o_density_arr) | np.isnan(pressure_arr)
    )

    if np.any(valid_mask):
        # Intermediate calculations for valid values only
        t_c1 = (
            pressure_arr[valid_mask]
            + (2 * RV - cvw_cvd_plus1 * RD)
            * h2o_density_arr[valid_mask]
            * sonic_temp_arr[valid_mask]
        )

        t_c2 = (
            pressure_arr[valid_mask] ** 2
            + (
                cvw_cvd_minus1
                * RD
                * h2o_density_arr[valid_mask]
                * sonic_temp_arr[valid_mask]
            )
            ** 2
            + cpw_cpv_factor
            * RD
            * h2o_density_arr[valid_mask]
            * pressure_arr[valid_mask]
            * sonic_temp_arr[valid_mask]
        )

        t_c3 = (
            2
            * h2o_density_arr[valid_mask]
            * (
                (RV - cpw_cpd * RD)
                + (RV - RD)
                * (RV - cvw_cvd * RD)
                * h2o_density_arr[valid_mask]
                * sonic_temp_arr[valid_mask]
                / pressure_arr[valid_mask]
            )
        )

        # Final calculation with error handling
        with np.errstate(invalid="ignore", divide="ignore"):
            temp_result = (t_c1 - np.sqrt(t_c2)) / t_c3
            # Handle any invalid results from the calculation
            valid_result = (
                ~np.isnan(temp_result) & ~np.isinf(temp_result) & (temp_result > 0)
            )
            result[valid_mask] = np.where(valid_result, temp_result, np.nan)

    # Return appropriate type based on input
    if is_scalar:
        return float(result.item()) if not np.isnan(result.item()) else None
    return result


def calculate_air_density(
    temperature: Union[float, np.ndarray],
    pressure: Union[float, np.ndarray],
    h2o_density: Union[float, np.ndarray],
) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Calculate dry air density, water vapor pressure, and moist air density.
    Supports both scalar values and numpy arrays as input.

    Args:
        temperature: Air temperature (K)
        pressure: Atmospheric pressure (kPa)
        h2o_density: Water vapor density (g/m³)

    Returns:
        For scalar inputs:
            Tuple[float, float, float] containing:
            - e_air: Water vapor pressure (kPa)
            - rho_d: Dry air density (g/m³)
            - rho_a: Moist air density (kg/m³)
        For array inputs:
            Tuple[np.ndarray, np.ndarray, np.ndarray] containing the same quantities

    Raises:
        ValueError: If inputs contain invalid values (NaN, negative, or zero for temperature/pressure)

    Notes:
        - Uses ideal gas law for water vapor pressure: e = ρRvT
        - Rv = 461.51 J/(kg·K) (specific gas constant for water vapor)
        - Rd = 287.04 J/(kg·K) (specific gas constant for dry air)
        - Input h2o_density is converted from g/m³ to kg/m³ for calculations
        - Output rho_d is converted to g/m³ for consistency with input units
    """
    # Convert inputs to numpy arrays
    temp_arr = np.asarray(temperature)
    pres_arr = np.asarray(pressure)
    h2o_arr = np.asarray(h2o_density)

    # Check if inputs are scalar
    is_scalar = all(np.isscalar(x) for x in [temperature, pressure, h2o_density])

    # Try broadcasting arrays to same shape
    try:
        broadcast_shape = np.broadcast_shapes(
            temp_arr.shape, pres_arr.shape, h2o_arr.shape
        )
        temp_arr = np.broadcast_to(temp_arr, broadcast_shape)
        pres_arr = np.broadcast_to(pres_arr, broadcast_shape)
        h2o_arr = np.broadcast_to(h2o_arr, broadcast_shape)
    except ValueError as e:
        raise ValueError(f"Input arrays have incompatible shapes: {e}")

    # Input validation
    if np.any(np.isnan([temp_arr, pres_arr, h2o_arr])):
        raise ValueError("Input contains NaN values")

    if np.any(temp_arr <= 0) or np.any(pres_arr <= 0):
        raise ValueError("Temperature and pressure must be positive")

    if np.any(h2o_arr < 0):
        raise ValueError("Water vapor density cannot be negative")

    # Constants (from R_SPECIFIC in constants.py)
    Rv = 461.51  # Specific gas constant for water vapor (J/kg/K)
    Rd = 287.04  # Specific gas constant for dry air (J/kg/K)

    # Convert water vapor density to kg/m³ for calculations
    h2o_density_kgm3 = h2o_arr / 1000.0

    # Calculate water vapor pressure (kPa)
    # e = ρRvT (ideal gas law)
    # Factor of 1000 converts J/m³ to kPa
    e_air = h2o_density_kgm3 * Rv * temp_arr / 1000.0

    # Calculate dry air density using ideal gas law
    # ρd = (P - e)/(RdT)
    # Pressure and vapor pressure are in kPa, so multiply by 1000 to get Pa
    rho_d = (pres_arr - e_air) * 1000.0 / (Rd * temp_arr)

    # Convert dry air density to g/m³
    rho_d *= 1000.0

    # Calculate total air density (kg/m³)
    # Add water vapor density (kg/m³) to dry air density (converted from g/m³)
    rho_a = rho_d / 1000.0 + h2o_density_kgm3

    # Return appropriate type based on input
    if is_scalar:
        return float(e_air.item()), float(rho_d.item()), float(rho_a.item())
    return e_air, rho_d, rho_a


def calculate_saturation_vapor_pressure(
    temperature: Union[float, np.ndarray], pressure: Union[float, np.ndarray]
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate saturation vapor pressure and enhancement factor.
    Supports both scalar values and numpy arrays as input.

    Args:
        temperature: Air temperature (°C)
        pressure: Atmospheric pressure (kPa)

    Returns:
        For scalar inputs:
            Tuple[float, float] containing:
            - e_sat: Saturation vapor pressure (kPa)
            - enhance_factor: Enhancement factor (dimensionless)
        For array inputs:
            Tuple[np.ndarray, np.ndarray] containing the same quantities
    """
    # Convert inputs to numpy arrays
    temp_arr = np.asarray(temperature)
    pres_arr = np.asarray(pressure)

    # Check if inputs are scalar
    is_scalar = all(np.isscalar(x) for x in [temperature, pressure])

    # Try broadcasting arrays to same shape
    try:
        broadcast_shape = np.broadcast_shapes(temp_arr.shape, pres_arr.shape)
        temp_arr = np.broadcast_to(temp_arr, broadcast_shape)
        pres_arr = np.broadcast_to(pres_arr, broadcast_shape)
    except ValueError as e:
        raise ValueError(f"Input arrays have incompatible shapes: {e}")

    # Calculate enhancement factor (vectorized)
    enhance_factor = 1.00041 + pres_arr * (
        3.48e-5 + 7.4e-9 * (temp_arr + 30.6 - 0.38 * pres_arr) ** 2
    )

    # Initialize saturation vapor pressure array
    e_sat = np.zeros_like(temp_arr, dtype=float)

    # Handle temperature ranges separately using masks
    # For temperatures >= 0°C
    mask_warm = temp_arr >= 0
    e_sat[mask_warm] = (
        0.61121
        * enhance_factor[mask_warm]
        * np.exp((17.368 * temp_arr[mask_warm]) / (temp_arr[mask_warm] + 238.88))
    )

    # For temperatures < 0°C
    mask_cold = ~mask_warm
    e_sat[mask_cold] = (
        0.61121
        * enhance_factor[mask_cold]
        * np.exp((17.966 * temp_arr[mask_cold]) / (temp_arr[mask_cold] + 247.15))
    )

    # Return appropriate type based on input
    if is_scalar:
        return float(e_sat.item()), float(enhance_factor.item())
    return e_sat, enhance_factor


def calculate_dewpoint_temperature(
    e_air: Union[float, np.ndarray],
    pressure: Union[float, np.ndarray],
    enhance_factor: Optional[Union[float, np.ndarray]] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate dew point temperature using enhanced vapor pressure.
    Supports both scalar values and numpy arrays as input.

    Args:
        e_air: Water vapor pressure (kPa)
        pressure: Atmospheric pressure (kPa)
        enhance_factor: Optional enhancement factor. If None, will be calculated.

    Returns:
        Union[float, np.ndarray]: Dew point temperature (°C)
        Returns float for scalar inputs, np.ndarray for array inputs
    """
    # Convert inputs to numpy arrays
    e_air_arr = np.asarray(e_air)
    pres_arr = np.asarray(pressure)

    # Check if inputs are scalar
    is_scalar = all(np.isscalar(x) for x in [e_air, pressure])
    if enhance_factor is not None:
        is_scalar = is_scalar and np.isscalar(enhance_factor)

    # Try broadcasting arrays to same shape
    try:
        broadcast_shape = np.broadcast_shapes(e_air_arr.shape, pres_arr.shape)
        e_air_arr = np.broadcast_to(e_air_arr, broadcast_shape)
        pres_arr = np.broadcast_to(pres_arr, broadcast_shape)

        if enhance_factor is not None:
            enhance_arr = np.asarray(enhance_factor)
            broadcast_shape = np.broadcast_shapes(broadcast_shape, enhance_arr.shape)
            e_air_arr = np.broadcast_to(e_air_arr, broadcast_shape)
            pres_arr = np.broadcast_to(pres_arr, broadcast_shape)
            enhance_arr = np.broadcast_to(enhance_arr, broadcast_shape)
        else:
            # Calculate initial enhancement factor
            enhance_arr = 1.00072 + 3.46e-5 * pres_arr

    except ValueError as e:
        raise ValueError(f"Input arrays have incompatible shapes: {e}")

    # First dewpoint estimate calculation
    x_tmp = np.log(e_air_arr / (0.61121 * enhance_arr))
    t_dp_first = 240.97 * x_tmp / (17.502 - x_tmp)

    # Recalculate enhancement factor with first dew point estimate
    enhance_arr = 1.00041 + pres_arr * (
        3.48e-5 + 7.4e-9 * (t_dp_first + 30.6 - 0.38 * pres_arr) ** 2
    )

    # Recalculate x_tmp with new enhancement factor
    x_tmp = np.log(e_air_arr / (0.61121 * enhance_arr))

    # Initialize output array
    t_dp = np.zeros_like(x_tmp)

    # Calculate final dew point based on temperature ranges using masks
    mask_warm = t_dp_first >= 0
    t_dp[mask_warm] = 238.88 * x_tmp[mask_warm] / (17.368 - x_tmp[mask_warm])

    mask_cold = ~mask_warm
    t_dp[mask_cold] = 247.15 * x_tmp[mask_cold] / (17.966 - x_tmp[mask_cold])

    # Return appropriate type based on input
    if is_scalar:
        return float(t_dp.item())
    return t_dp


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
            self.params.temperature, self.params.pressure, self.params.h2o_density
        )

        e_sat, enhance_factor = calculate_saturation_vapor_pressure(
            temp_c, self.params.pressure
        )

        # Calculate relative humidity
        rh = 100.0 * e_air / e_sat

        # Calculate dew point
        t_dp = calculate_dewpoint_temperature(
            e_air, self.params.pressure, enhance_factor
        )

        return {
            "air_temperature": self.params.temperature,
            "vapor_pressure": e_air,
            "saturation_vapor_pressure": e_sat,
            "relative_humidity": rh,
            "dewpoint_temperature": t_dp,
            "dry_air_density": rho_d,
            "moist_air_density": rho_a,
        }
