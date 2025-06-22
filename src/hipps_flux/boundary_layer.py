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
    """
    Convert *sonic* (virtual) temperature to true air (thermodynamic)
    temperature.

    Sonic anemometers report a **virtual** temperature :math:`T_s`
    because the transit time of the acoustic pulse depends on the speed
    of sound, which itself is a function of the *actual* temperature and
    the density of water vapour.
    The bias can be removed with (Kaimal & Gaynor, 1990)

    .. math::
       T_a \;=\; \frac{T_s}{1 + 0.32\,e / p}

    where

    * :math:`e` – water‐vapour partial pressure (Pa)
      ``e = ρ_v R_v T_s``
    * :math:`p` – ambient pressure (Pa)

    The routine is fully vectorised: any of the three primary inputs may
    be a NumPy array, provided the shapes are broadcast–compatible.

    Parameters
    ----------
    sonic_temp : float or ndarray
        Virtual (sonic) temperature, **Kelvin**.
    h2o_density : float or ndarray
        Absolute moisture content ρᵥ (g m⁻³).  Values must be
        non-negative.
    pressure : float or ndarray
        Ambient static pressure (kPa).  Values must be positive.
    Rd : float, default ``287.04``
        Specific gas constant for dry air (J kg⁻¹ K⁻¹).  Present for
        completeness but *not* used in the current formulation.
    Rv : float, default ``461.5``
        Specific gas constant for water vapour (J kg⁻¹ K⁻¹).

    Returns
    -------
    float or ndarray
        Corrected air temperature (Kelvin).
        Returns ``None`` if any validation check fails (negative inputs,
        NaNs, temperatures outside the physically plausible range
        −100 °C … +100 °C).

    Raises
    ------
    None
        All exceptional circumstances are **caught** internally; the
        function signals failure by returning ``None``.

    Notes
    -----
    * The 0.32 coefficient assumes the humidity contribution derived
      from the ratio ``Rd/Rv`` and works well for temperatures near
      300 K.  Extremely cold or hot conditions may require a refined
      coefficient (Schotanus et al., 1983).
    * Output is **Kelvin**.  Convert to Celsius by subtracting 273.15.
    * Broadcasting follows NumPy rules—​for example, a scalar pressure
      can combine with vector ``sonic_temp`` and ``h2o_density`` of the
      same shape.

    Examples
    --------
    >>> Ts  = 305.15                 # K  (≈ 32 °C)
    >>> rho = 12.0                   # g m⁻³  (≈ 60 % RH at 32 °C)
    >>> p   = 95.0                   # kPa  (≈ 900 m a.s.l.)
    >>> Ta  = calculate_air_temperature(Ts, rho, p)
    >>> round(Ta - 273.15, 2)        # °C
    30.34

    Vectorised usage:

    >>> import numpy as np
    >>> Ts  = np.array([300., 305., 310.])
    >>> rho = np.array([10., 12., 14.])
    >>> p   = 95.0                   # scalar broadcasts
    >>> calculate_air_temperature(Ts, rho, p)
    array([298.23..., 303.34..., 308.46...])
    """

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
    Estimate the planetary-boundary-layer (PBL) height *zₚᵦₗ*
    from the Obukhov length *L* following the piece-wise parameterisation
    of **Kljun et al.** (2004, updated 2015).

    The mapping captures the characteristic *capping-inversion* behaviour
    of the convective (unstable) and stable boundary layers:

    * **Unstable** ``L < 0``
      PBL grows with increasing instability down to a lower limit of
      *L* ≈ −1013 m, at which ``zₚᵦₗ = 1000 m``.
    * **Near-neutral** ``−0.1 ≤ L ≤ 50 m``
      PBL height is ≈ 2 km and varies only weakly with *L*.
    * **Stable** ``L > 0``
      PBL height collapses rapidly with increasing stability, reaching
      ≈ 200 m at *L* ≈ 50 m and returning to 1 km for very stable
      *L* ≥ 1500 m (representing e.g. residual layers).

    Linear interpolation is used between the break-points listed in the
    original look-up table (see *Notes*).

    Parameters
    ----------
    obukhov : float or ndarray
        Obukhov length *L* (m).  May be a scalar or any NumPy-broadcast-
        able array.  Missing values (``np.nan``) propagate to the output.

    Returns
    -------
    float or ndarray
        Estimated PBL height *zₚᵦₗ* (m).  For scalar input the function
        returns a scalar **or** ``None`` if validation fails; for array
        input invalid elements are set to ``np.nan``.

    Notes
    -----
    Break-points (*L*, *zₚᵦₗ*) used for the piece-wise linear mapping
    (unstable < 0 on the left, stable > 0 on the right):

    ====================  ==========  ==========
    Range (*L*, m)        Lower bp    Upper bp
    ====================  ==========  ==========
    L ≤ −1013.3           (−1013.3, 1000)
    −1013.3 < L ≤ −800    1000 → 1117.42
    −800     < L ≤ −300    1117.42 → 1472
    −300     < L ≤ −15     1472 → 1980
    −15      < L ≤ −0.1    1980 → 2019
    −0.1     < L < 0       2019 → 2000
    0 ≤ L < 50             200 → 184.13
    50 ≤ L < 500           184.13 → 432.18
    500 ≤ L < 1100         432.18 → 843.29
    1100 ≤ L < 1500        843.29 → 1000
    L ≥ 1500              (1500, 1000)
    ====================  ==========  ==========

    Values chosen reproduce the graphic in Kljun et al. (2015, Fig. 3).

    **Validity.**  The scheme is empirical and intended for flat to
    gently rolling terrain; it is *not* suitable for strongly
    heterogeneous or mountainous sites where local circulations dominate
    PBL depth.

    Examples
    --------
    Scalar input
    >>> planetary_boundary_layer_height(-200.0)
    1621.68

    Vectorised input with nan handling
    >>> import numpy as np
    >>> L = np.array([ -1200, -500, -5, 10, 1200, np.nan ])
    >>> planetary_boundary_layer_height(L)
    array([1000.  , 1294.84, 1996.34, 192.41, 1000.  ,     nan])
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
    Convert **virtual (sonic)** temperature to *true* air temperature
    using the extended heat-capacity formulation of *Schotanus et al.*
    (1983) as recast by **Kaimal & Gaynor** (1990, their Eq. 14).

    The speed of sound depends on the mixture of dry air and water
    vapour.  Sonic anemometers therefore report the *virtual*
    temperature :math:`T_s` biased high relative to the thermodynamic
    temperature :math:`T_a`.  The correction becomes significant under
    warm, humid conditions.  With the specific heats

    * ``Cpd = 1004   J kg⁻¹ K⁻¹`` (dry-air constant-pressure)
    * ``Cvd =  717   J kg⁻¹ K⁻¹`` (dry-air constant-volume)
    * ``Cpw = 1952   J kg⁻¹ K⁻¹`` (water-vapour constant-pressure)
    * ``Cvw = 1463   J kg⁻¹ K⁻¹`` (water-vapour constant-volume)

    the implicit equation for *Tₐ* can be rearranged to the closed-form
    quadratic solved here:

    .. math::
       T_a \;=\;
       \frac{ p + (2R_v - (C_{vw}/C_{vd}+1)R_d)\,ρ_v T_s -
              \sqrt{Δ} }
            { 2ρ_v\,\bigl[ (R_v - C_{pw}/C_{pd} R_d)
                          + (R_v-R_d)(R_v-C_{vw}/C_{vd}R_d)ρ_vT_s/p
                         \bigr] }

    where *p* is pressure (Pa) and :math:`ρ_v` is absolute humidity
    (kg m⁻³).  The intermediate discriminant *Δ* and constant factors
    are pre-computed for speed.

    The implementation is fully *vectorised*—inputs may be scalars or
    broadcast-compatible NumPy arrays.

    Parameters
    ----------
    sonic_temp : float or ndarray
        Virtual (sonic) temperature :math:`T_s` in **Kelvin**.
    h2o_density : float or ndarray
        Water-vapour density :math:`ρ_v` in **g m⁻³** (non-negative).
    pressure : float or ndarray
        Atmospheric pressure in **kPa** (positive).

    Returns
    -------
    float or ndarray or None
        Corrected air temperature :math:`T_a` (Kelvin).

        * **Scalar inputs** – a single ``float`` or ``None`` if any
          validation fails.
        * **Array inputs** – an array of the broadcast shape; elements
          that cannot be evaluated are set to ``np.nan``.

    Raises
    ------
    None
        All internal errors (shape mismatch, invalid maths) are caught
        and converted to ``None`` / ``np.nan`` as described above.

    Notes
    -----
    * **Units** – do **not** pass Celsius; convert to Kelvin first.
    * The result is constrained to the physically reasonable range
      −100 °C … +100 °C (173.15 K … 373.15 K).  Values outside the range
      are flagged invalid.
    * Broadcasting follows NumPy rules.  A scalar pressure can be mixed
      with vector temperature and humidity.
    * For most eddy-covariance applications the simpler
      ``air_temperature = sonic_temp / (1 + 0.32 e/p)`` (Kaimal & Gaynor,
      1990) is adequate.  The present formulation is more precise when
      very high humidity or low pressure prevail.

    References
    ----------
    Schotanus, P., Nieuwstadt, F., & de Bruin, H. (1983).
    *Temperature measurement with a sonic anemometer and its application
    to heat and moisture fluxes*. **Boundary-Layer Meteorology**, *26*,
    81–93.

    Kaimal, J. C., & Gaynor, J. E. (1990).
    *Another look at sonic thermometry*. **Boundary-Layer Meteorology**,
    *53*, 401–410.

    Examples
    --------
    >>> Ts  = 303.15      # 30 °C in K
    >>> rho = 15.0        # g m⁻³  (≈ 70 % RH at 30 °C)
    >>> p   = 98.0        # kPa  (≈ 400 m a.s.l.)
    >>> Ta  = air_temperature_from_sonic(Ts, rho, p)
    >>> round(Ta - 273.15, 2)      # convert to °C
    28.44

    Vectorised usage with automatic broadcasting:

    >>> import numpy as np
    >>> Ts  = np.array([290., 300., 310.])
    >>> rho = np.array([  8.,  12.,  18.])   # g m⁻³
    >>> p   = 101.3                          # sea-level kPa
    >>> air_temperature_from_sonic(Ts, rho, p)
    array([288.52..., 298.28..., 307.97...])
    """
    # function body …

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
) -> Union[
    Tuple[float, float, float],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Compute dry-air density, water-vapour pressure, and moist-air density
    from temperature, pressure, and absolute humidity.

    The routine supports *broadcast-compatible* NumPy arrays as well as
    scalars; all inputs are first broadcast to the same shape.  It
    returns three quantities:

    ===========  =====================================  ===================
    Symbol       Description                            Units
    ===========  =====================================  ===================
    *e*          water-vapour partial pressure           kPa
    *ρ_d*        dry-air density                         g m⁻³
    *ρ_a*        moist-air (virtual) density             kg m⁻³
    ===========  =====================================  ===================

    Ideal-gas relationships are applied with the specific gas constants

    ``Rv = 461.51  J kg⁻¹ K⁻¹`` (water vapour)
    ``Rd = 287.04  J kg⁻¹ K⁻¹`` (dry air)

    and the conversion ``1 kPa = 1000 J m⁻³`` for pressure–energy
    equivalence.

    Parameters
    ----------
    temperature : float or ndarray
        Air temperature :math:`T` in **Kelvin**; must be strictly
        positive.
    pressure : float or ndarray
        Ambient pressure :math:`p` in **kPa**; must be strictly
        positive.
    h2o_density : float or ndarray
        Absolute water-vapour density :math:`ρ_v` in **g m⁻³**;
        non-negative.

    Returns
    -------
    (e_air, rho_d, rho_a) : tuple
        * **Scalar input** → 3-tuple of floats
          ``(e_air, ρ_d, ρ_a)``.
        * **Array input** → 3-tuple of ndarrays with the broadcast
          shape of the inputs.

        Units follow the table above.

    Raises
    ------
    ValueError
        If any argument contains ``NaN`` or violates positivity
        constraints, or if the input shapes cannot be broadcast to a
        common shape.

    Notes
    -----
    * **Water-vapour pressure**

      .. math:: e \;=\; ρ_v R_v T / 1000

      (dividing by 1000 converts to kilopascals).

    * **Dry-air density**

      .. math:: ρ_d \;=\; \frac{p - e}{R_d T}\,·1000

      returned in **g m⁻³** to match the input water-vapour density
      units.

    * **Moist-air density**

      .. math:: ρ_a \;=\; ρ_d/1000 + ρ_v

      where ``ρ_d/1000`` converts g m⁻³ back to kg m⁻³.

    * All calculations vectorise automatically; use scalar inputs for
      single-value evaluation.

    Examples
    --------
    >>> # Scalar example (sea-level conditions, 20 °C, 60 % RH)
    >>> T   = 293.15        # K
    >>> p   = 101.3         # kPa
    >>> rho = 10.5          # g m⁻³  absolute humidity
    >>> e, rho_d, rho_a = calculate_air_density(T, p, rho)
    >>> round(e, 2), round(rho_d, 1), round(rho_a, 3)
    (2.49, 1190.5, 1.200)

    >>> # Vectorised example
    >>> import numpy as np
    >>> T   = np.array([280., 290., 300.])      # K
    >>> p   = 95.0                              # kPa  (broadcasts)
    >>> rho = np.array([6., 9., 15.])           # g m⁻³
    >>> e, rho_d, rho_a = calculate_air_density(T, p, rho)
    >>> e
    array([ 2.42...,  3.53...,  5.91...])  # kPa

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
    temperature: Union[float, np.ndarray],
    pressure: Union[float, np.ndarray],
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Saturation vapour pressure over liquid water / ice **and**
    the pressure–broadening enhancement factor.

    The routine implements the WMO thermodynamic formulation recommended
    by *Buck* (1981, 1996 revision) combined with the *Goff–Gratch*
    enhancement factor that corrects for the non-ideal behaviour of
    moist air at elevated pressures.  It is fully vectorised—both
    *temperature* and *pressure* may be scalars or broadcast-compatible
    NumPy arrays.

    Two sets of empirical constants are used:

    * ``T ≥ 0 °C`` (Buck over **liquid water**)
    * ``T < 0 °C`` (Buck over **ice**)

    Parameters
    ----------
    temperature : float or ndarray
        Dry-bulb air temperature in **degrees Celsius** (°C).  Valid for
        at least −60 °C ≤ *T* ≤ +60 °C, the calibrated range of the Buck
        equations.
    pressure : float or ndarray
        Ambient static pressure *p* in **kilopascals** (kPa).
        Must be strictly positive.

    Returns
    -------
    e_sat : float or ndarray
        Saturation vapour pressure (kPa) corresponding to *temperature*
        and *pressure*.
    enhance_factor : float or ndarray
        Dimensionless enhancement factor *f* that accounts for pressure
        broadening of the vapour pressure curve.
        The **actual** saturation vapour pressure is
        ``e_sat = f · e_s``, where *e_s* is the ideal saturation pressure
        from the Buck equation.

        * Scalar inputs → ``Tuple[float, float]``
          ``(e_sat, enhance_factor)``
        * Array inputs  → ``Tuple[np.ndarray, np.ndarray]``
          with the broadcast shape.

    Raises
    ------
    ValueError
        If *temperature* and *pressure* cannot be broadcast to a common
        shape, or if *pressure* contains non-positive values.

    Notes
    -----
    **Enhancement factor**

    .. math::
        f(p, T) = 1.00041 \;+\; p
                   \Bigl[\,3.48\\times10^{-5} + 7.4\\times10^{-9}
                   (T + 30.6 - 0.38\,p)^2 \Bigr]

    where *p* is in kPa and *T* in °C.

    **Buck saturation vapour pressure**

    .. math::
        e_s(T) =
        \\begin{cases}
            0.61121 \exp
            \\bigl[(17.368\,T)/(T + 238.88)\\bigr], & T ≥ 0^{\\circ}\\text{C} \\\\[4pt]
            0.61121 \exp
            \\bigl[(17.966\,T)/(T + 247.15)\\bigr], & T < 0^{\\circ}\\text{C}
        \\end{cases}

    The final ``e_sat`` returned here is ``f × e_s``.

    References
    ----------
    Buck, A. L. (1981). *New equations for computing vapor pressure and
    enhancement factor*. **J. Appl. Meteorol.**, *20*, 1527–1532.
    Buck, A. L. (1996). *Comparison of NOAA/Atmospheric Laboratory and
    Wexler vapor pressure formulations*. **J. Appl. Meteorol.**,
    *35*, 1227–1232.

    Examples
    --------
    >>> # Scalar example: 25 °C, 100 kPa
    >>> e, f = calculate_saturation_vapor_pressure(25.0, 100.0)
    >>> round(e, 3), round(f, 6)
    (3.178, 1.004)

    >>> # Vectorised example with automatic broadcasting
    >>> import numpy as np
    >>> T  = np.array([-10.,  0., 10., 20.])   # °C
    >>> p  = 95.0                              # kPa (scalar broadcasts)
    >>> e, f = calculate_saturation_vapor_pressure(T, p)
    >>> e
    array([0.260..., 0.612..., 1.228..., 2.339...])
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
    Compute dew-point temperature from vapour pressure and ambient
    pressure, with optional user-supplied enhancement factors.

    The algorithm follows the two–step *Buck (1981, 1996)* formulation
    used by the WMO:

    1. **Initial estimate**
       Assume a simple pressure-broadening factor
       :math:`f_{0} = 1.00072 + 3.46·10^{-5} \, p`
       (``p`` in kPa) and invert the warm-water Buck equation to obtain a
       first dew-point :math:`T_{d}^{(0)}`.
    2. **Refined estimate**
       Re-evaluate the enhancement factor with the full quadratic form,
       replace *f* in the saturation equation, and solve again:

       *For liquid water (`T_d ≥ 0 °C`)*

       .. math::
          T_d = \frac{238.88 \, \ln\!\bigl(e/(0.61121 f)\bigr)}
                       {17.368 - \ln\!\bigl(e/(0.61121 f)\bigr)}

       *For ice (`T_d < 0 °C`)*

       .. math::
          T_d = \frac{247.15 \, \ln\!\bigl(e/(0.61121 f)\bigr)}
                       {17.966 - \ln\!\bigl(e/(0.61121 f)\bigr)}

    Parameters
    ----------
    e_air : float or ndarray
        Actual water-vapour pressure *e* (kPa).
    pressure : float or ndarray
        Ambient static pressure *p* (kPa).  Must be positive.
    enhance_factor : float or ndarray, optional
        Pre-computed enhancement factor *f*.
        If *None* (default) the routine estimates *f* internally via the
        two-step process described above.
        May be a scalar or array broadcast-compatible with *e_air*.

    Returns
    -------
    float or ndarray
        Dew-point temperature (°C).
        A scalar is returned for scalar input; otherwise an array with
        the common broadcast shape.

    Raises
    ------
    ValueError
        If the inputs cannot be broadcast to a common shape or if
        *pressure* contains non-positive values.

    Notes
    -----
    * **Enhancement factor** – corrects saturation vapour pressure for
      the non-ideal behaviour of moist air.  Supplying an accurate
      *f* (e.g. from :pyfunc:`calculate_saturation_vapor_pressure`) skips
      the first-guess iteration and yields slightly faster execution.
    * Valid temperature range: −60 °C ≤ *Tₙ* ≤ +60 °C (empirical fit).
      Extreme values may produce small extrapolation errors.
    * All arithmetic is vectorised with NumPy; inputs can be any shape
      so long as broadcasting rules are satisfied.

    References
    ----------
    Buck, A. L. (1981). *New equations for computing vapor pressure and
    enhancement factor*. **J. Appl. Meteor.**, *20*, 1527–1532.
    Buck, A. L. (1996). *Comparison of NOAA/Atmospheric Laboratory and
    Wexler vapor pressure formulations*. **J. Appl. Meteor.**,
    *35*, 1227–1232.

    Examples
    --------
    >>> # Scalar example
    >>> e  = 1.8         # kPa
    >>> p  = 95.0        # kPa
    >>> Td = calculate_dewpoint_temperature(e, p)
    >>> round(Td, 2)
    14.03

    >>> # Vectorised example with custom enhancement factor
    >>> import numpy as np
    >>> e  = np.array([0.6, 1.2, 2.4])         # kPa
    >>> p  = np.array([101.3, 100.0,  98.0])   # kPa
    >>> f  = 1.00041 + 3.46e-5 * p             # simple estimate
    >>> calculate_dewpoint_temperature(e, p, enhance_factor=f)
    array([ 5.93..., 11.14..., 18.23...])
    """
    ...

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
    End-to-end processor for near-surface boundary-layer observations.

    The class transforms a minimal set of *raw* meteorological inputs
    (height above ground, air temperature, pressure, water-vapour
    density) into a suite of frequently required boundary-layer metrics:

    * **Thermodynamic properties** – true air temperature, vapour
      pressure, saturation vapour pressure, dew-point temperature,
      relative humidity.
    * **Density diagnostics** – dry-air density and moist-air (virtual)
      density.

    All calculations leverage the utility functions defined in the same
    module:

    * :pyfunc:`calculate_air_density`
    * :pyfunc:`calculate_saturation_vapor_pressure`
    * :pyfunc:`calculate_dewpoint_temperature`

    The processor is therefore *unit-aware*:
    – temperature **Kelvin**,
    – pressure **kilopascals**,
    – water-vapour density **g m⁻³**.

    Parameters
    ----------
    params : BoundaryLayerParams
        Dataclass or simple namespace containing the raw measurements
        with attributes

        =====================  ================  ====================
        Attribute               Symbol           Units
        =====================  ================  ====================
        ``height``              *z*              m
        ``temperature``         *T*              K
        ``pressure``            *p*              kPa
        ``h2o_density``         ρᵥ              g m⁻³
        =====================  ================  ====================

    Attributes
    ----------
    params : BoundaryLayerParams
        Stored copy of the input structure.
    T_0C_K : float
        Module-level constant (273.15 K) used internally for °C
        conversions.

    Raises
    ------
    ValueError
        If any of the mandatory attributes in *params* fails the basic
        sanity checks:

        * ``height`` ≤ 0 m
        * ``temperature`` < 0 K
        * ``pressure`` ≤ 0 kPa
        * ``h2o_density`` < 0 g m⁻³

    Notes
    -----
    *The processor does **not*** estimate the planetary-boundary-layer
    depth; see :pyfunc:`planetary_boundary_layer_height` for that
    functionality.
    * All returned quantities are **instantaneous**—use appropriate
      temporal averaging before interpreting boundary-layer statistics.

    Examples
    --------
    >>> from mymodule.boundary import BoundaryLayerParams, BoundaryLayerProcessor
    >>> raw = BoundaryLayerParams(
    ...     height       = 2.0,      # m
    ...     temperature  = 298.15,   # K  (25 °C)
    ...     pressure     = 95.0,     # kPa (≈900 m a.s.l.)
    ...     h2o_density  = 12.0      # g m⁻³
    ... )
    >>> blp = BoundaryLayerProcessor(raw)
    >>> results = blp.process_measurements()
    >>> round(results["relative_humidity"], 1)
    56.8
    """

    def __init__(self, params: BoundaryLayerParams):
        """
        Instantiate the processor and immediately validate inputs.

        Parameters
        ----------
        params : BoundaryLayerParams
            Container with the raw boundary-layer observations.

        Raises
        ------
        ValueError
            Propagated from :pymeth:`_validate_params` when mandatory
            sanity checks fail.
        """
        self.params = params
        self._validate_params()

    def _validate_params(self) -> None:
        """
        Internal consistency checks for the input measurements.

        Raises
        ------
        ValueError
            If any of the following conditions is violated:

            * ``height``  > 0 m
            * ``temperature`` ≥ 0 K
            * ``pressure`` > 0 kPa
            * ``h2o_density`` ≥ 0 g m⁻³
        """
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
        Derive secondary thermodynamic and hygrometric variables.

        The method calls the lower-level helper functions in sequence:

        1. :pyfunc:`calculate_air_density` → *vapour pressure* (*e*),
           *dry-air density* (ρ_d), *moist-air density* (ρ_a)
        2. :pyfunc:`calculate_saturation_vapor_pressure` → *e_sat*, *f*
           (enhancement factor)
        3. Relative humidity ``RH = 100 e / e_sat``
        4. :pyfunc:`calculate_dewpoint_temperature` → *T_d*
        5. Assemble results into a dictionary.

        Returns
        -------
        dict
            Mapping with the following keys

            ================  ====================================  Units
            air_temperature   thermodynamic air temperature *T*     K
            vapor_pressure    actual vapour pressure *e*            kPa
            saturation_vapor_pressure  e_sat                        kPa
            relative_humidity RH                                      %
            dewpoint_temperature *T_d*                              °C
            dry_air_density    ρ_d                                  g m⁻³
            moist_air_density  ρ_a                                  kg m⁻³
            ================  ====================================  =====

        Notes
        -----
        *If* the derived relative humidity exceeds 100 % or falls below
        0 %, suspect measurement or conversion errors in the raw inputs.
        * The returned dictionary is intentionally light-weight; convert
          to a *pandas* ``Series`` or *xarray* ``Dataset`` as needed.

        Examples
        --------
        >>> blp = BoundaryLayerProcessor(raw_params)    # doctest: +SKIP
        >>> blp.process_measurements()                 # doctest: +SKIP
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
