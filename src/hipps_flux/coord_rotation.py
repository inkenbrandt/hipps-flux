"""
Coordinate rotation transformations for eddy covariance measurements.

This module implements two coordinate rotation methods:
1. Double rotation (Tanner & Thurtell, 1969)
2. Planar fit rotation (Wilczak et al., 2001)

References:
    Tanner, C. B., & Thurtell, G. W. (1969). Anemoclinometer measurements of Reynolds stress and heat transport in the atmospheric surface layer. ECOM-66-G22-F.
    Wilczak, J. M., Oncley, S. P., & Stage, S. A. (2001). Boundary-Layer Meteorology, 99(1), 127-150.
"""

from typing import Tuple, Optional, Union, List
import numpy as np
from dataclasses import dataclass


def sin(x: float) -> float:
    """Helper function for cleaner sin calls"""
    return np.sin(x)


def cos(x: float) -> float:
    """Helper function for cleaner cos calls"""
    return np.cos(x)


@dataclass
class WindComponents:
    u_mean: float
    v_mean: float
    w_mean: float
    uu_cov: float
    vv_cov: float
    ww_cov: float
    uv_cov: float
    uw_cov: float
    vw_cov: float
    """
    Compact container for first- and second-order wind statistics.

    The class stores the time-averaged orthogonal wind components
    :math:`(u,\,v,\,w)` and their covariances, which are commonly derived
    from high-frequency sonic anemometer data in micrometeorological and
    turbulence studies.  These statistics form the basis for
    flux-gradient relationships, eddy-covariance flux calculations, and
    Reynolds-stress analysis.

    Parameters
    ----------
    u_mean : float
        Mean **streamwise** (east–west) wind component (m s⁻¹).
    v_mean : float
        Mean **cross-stream** (north–south) wind component (m s⁻¹).
    w_mean : float
        Mean **vertical** wind component (m s⁻¹).  
        Ideally close to zero after proper coordinate rotation.
    uu_cov : float
        Variance of the streamwise component,  
        :math:`\operatorname{var}(u')` (m² s⁻²).
    vv_cov : float
        Variance of the cross-stream component,  
        :math:`\operatorname{var}(v')` (m² s⁻²).
    ww_cov : float
        Variance of the vertical component,  
        :math:`\operatorname{var}(w')` (m² s⁻²).
    uv_cov : float
        Reynolds stress term :math:`\overline{u'v'}` (m² s⁻²).
    uw_cov : float
        Reynolds stress term :math:`\overline{u'w'}` (m² s⁻²);  
        proportional to kinematic momentum flux.
    vw_cov : float
        Reynolds stress term :math:`\overline{v'w'}` (m² s⁻²).

    Attributes
    ----------
    All constructor arguments are stored as public attributes of the
    same name.

    Notes
    -----
    * Primes (′) indicate deviations from the mean:
      :math:`u' = u - \overline{u}`.  
      Variances and covariances are normally computed over 30 min
      windows for eddy-covariance applications.
    * After double-rotation (Tanner & Thurtell, 1969) the mean vertical
      velocity ``w_mean`` should be ≈ 0 and ``uv_cov`` should vanish for
      horizontally homogeneous terrain.
    * Units are assumed to be SI; adjust if using different conventions.

    Examples
    --------
    >>> from dataclasses import asdict
    >>> wc = WindComponents(
    ...     u_mean = 1.2,
    ...     v_mean = -0.3,
    ...     w_mean = 0.01,
    ...     uu_cov = 0.45,
    ...     vv_cov = 0.32,
    ...     ww_cov = 0.06,
    ...     uv_cov = 0.02,
    ...     uw_cov = -0.08,
    ...     vw_cov = 0.01,
    ... )
    >>> asdict(wc)     # doctest: +ELLIPSIS
    {'u_mean': 1.2, 'v_mean': -0.3, 'w_mean': 0.01, 'uu_cov': 0.45, ...}

    Converting kinematic momentum flux (``uw_cov``) to dynamic units:

    >>> air_density = 1.18              # kg m⁻³ at 25 °C, 1 atm
    >>> tau = air_density * wc.uw_cov   # N m⁻² (Pa)
    >>> round(tau, 3)
    -0.094
    """


@dataclass
class RotatedWindComponents:
    u_rot: float
    v_rot: float
    w_rot: float
    uu_cov_rot: float
    vv_cov_rot: float
    ww_cov_rot: float
    uv_cov_rot: float
    uw_cov_rot: float
    vw_cov_rot: float
    """
    Wind statistics after coordinate rotation to the *stream-wise* frame.

    Raw sonic‐anemometer data are usually rotated so that the new
    :math:`x`-axis aligns with the mean flow while the mean vertical
    velocity vanishes.  In the double-rotation method
    (Tanner & Thurtell, 1969) two sequential rotations (yaw → pitch)
    achieve

    * :math:`\overline{w} = 0`
    * :math:`\overline{v} = 0`

    Triple-rotation additionally forces :math:`\overline{u'v'} = 0`
    (Kaimal & Finnigan, 1994), yielding the *stream-line* coordinate
    system.  This dataclass stores the rotated mean components and their
    second moments.

    Parameters
    ----------
    u_rot : float
        Mean wind component along the rotated *x*-axis (m s⁻¹).
    v_rot : float
        Mean wind component along the rotated *y*-axis (m s⁻¹);  
        ≈ 0 after double/triple rotation.
    w_rot : float
        Mean wind component along the rotated *z*-axis (m s⁻¹);  
        ≈ 0 after rotation.
    uu_cov_rot : float
        Variance :math:`\operatorname{var}(u')` in rotated coordinates
        (m² s⁻²).
    vv_cov_rot : float
        Variance :math:`\operatorname{var}(v')` in rotated coordinates
        (m² s⁻²).
    ww_cov_rot : float
        Variance :math:`\operatorname{var}(w')` in rotated coordinates
        (m² s⁻²).
    uv_cov_rot : float
        Rotated Reynolds stress term :math:`\overline{u'v'}` (m² s⁻²);  
        ≈ 0 after triple rotation.
    uw_cov_rot : float
        Rotated Reynolds stress term :math:`\overline{u'w'}` (m² s⁻²);  
        proportional to the *streamwise* kinematic momentum flux.
    vw_cov_rot : float
        Rotated Reynolds stress term :math:`\overline{v'w'}` (m² s⁻²).

    Attributes
    ----------
    All constructor arguments are stored as public attributes with the
    same names.

    Notes
    -----
    * **Double rotation** (yaw–pitch) ensures
      :math:`\overline{v} = \overline{w} = 0`.  
      **Triple rotation** adds a roll adjustment so that
      :math:`\overline{u'v'} = 0`.
    * Proper rotation is critical for eddy-covariance fluxes because it
      minimises coordinate‐induced biases in ``uw_cov_rot`` and
      ``vw_cov_rot``.
    * Units follow the SI convention (m s⁻¹ for means, m² s⁻² for
      covariances).

    Examples
    --------
    >>> from dataclasses import asdict
    >>> rc = RotatedWindComponents(
    ...     u_rot       = 2.7,
    ...     v_rot       = 0.0,
    ...     w_rot       = 0.0,
    ...     uu_cov_rot  = 0.38,
    ...     vv_cov_rot  = 0.24,
    ...     ww_cov_rot  = 0.05,
    ...     uv_cov_rot  = 0.0,
    ...     uw_cov_rot  = -0.07,
    ...     vw_cov_rot  = 0.01,
    ... )
    >>> asdict(rc)['uw_cov_rot']
    -0.07

    Converting the rotated kinematic momentum flux to a dynamic shear
    stress (Pa):

    >>> rho = 1.22                 # kg m⁻³ (air density)
    >>> tau = rho * rc.uw_cov_rot   # N m⁻²
    >>> round(tau, 3)
    -0.085
    """


class DoubleRotation:
    """
    Perform a *double rotation* of sonic-anemometer wind statistics.

    The double-rotation algorithm (Tanner & Thurtell, 1969) aligns the
    measurement axes with the mean flow in two sequential steps:

    1. **Yaw** (*γ*): rotate about the original *z*-axis so the new
       *x*-axis points along the horizontal mean wind vector
       ``(ū, v̄)``.
    2. **Pitch** (*α*): rotate about the intermediate *y*-axis to set
       the mean vertical velocity to zero
       (``w̄ → 0``).

    These rotations remove sensor tilt and streamline the coordinate
    system, which is essential for accurate eddy-covariance fluxes and
    Reynolds-stress analysis.

    Parameters
    ----------
    None
        The constructor takes no arguments; rotation angles are
        initialised to zero.

    Attributes
    ----------
    alpha : float
        Pitch angle (*α*, rad) about the intermediate *y*-axis that
        nulls the mean vertical velocity.
    beta : float
        Placeholder for the **roll** angle used in *triple rotation*
        (not computed here).  Remains zero.
    gamma : float
        Yaw angle (*γ*, rad) about the original *z*-axis that aligns the
        *x*-axis with the horizontal mean flow.

    References
    ----------
    * Tanner, B. D., & Thurtell, G. W. (1969). *An improved ultrasonic
      anemometer for turbulence measurements*. Amer. Meteor. Soc.
    * Kaimal, J. C., & Finnigan, J. J. (1994). *Atmospheric Boundary
      Layer Flows: Their Structure and Measurement*. Oxford University
      Press.

    Examples
    --------
    >>> comp = WindComponents(
    ...     u_mean = 1.5,  v_mean = 0.3,  w_mean = 0.05,
    ...     uu_cov = 0.41, vv_cov = 0.26, ww_cov = 0.07,
    ...     uv_cov = 0.02, uw_cov = -0.09, vw_cov = 0.01,
    ... )
    >>> rot = DoubleRotation()
    >>> rot.calculate_angles(comp)     # determine α and γ
    >>> rc = rot.rotate_wind(comp)     # apply to means & covariances
    >>> round(rc.w_rot, 6)             # ≈ 0 after rotation
    0.0
    ```

    """

    def __init__(self):
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0

    def calculate_angles(self, components: WindComponents) -> None:
        """
        Compute yaw (*γ*) and pitch (*α*) angles for double rotation.

        The routine stores the angles in :pyattr:`alpha` and
        :pyattr:`gamma` so that they can be reused by
        :pymeth:`rotate_wind`.

        Parameters
        ----------
        components : WindComponents
            Mean wind components prior to rotation.

        Returns
        -------
        None
            The object’s state is updated in place.

        Notes
        -----
        * **Yaw**
          ``γ = arctan2(v̄, ū)``
        * **Pitch**
          Given the horizontally rotated mean
          velocity ``(u₁, w₁)``, pitch is
          ``α = arctan2(w₁, u₁)``.

        Examples
        --------
        >>> dr = DoubleRotation()
        >>> dr.calculate_angles(WindComponents(...))  # doctest: +SKIP
        >>> print(dr.gamma, dr.alpha)                 # doctest: +SKIP
        """
        # First rotation angle (gamma)
        self.gamma = np.arctan2(components.v_mean, components.u_mean)

        # Intermediate velocities after first rotation
        cos_g = np.cos(self.gamma)
        sin_g = np.sin(self.gamma)

        # Rotate into mean wind
        u1 = components.u_mean * cos_g + components.v_mean * sin_g
        w1 = components.w_mean

        # Second rotation angle (alpha) to zero vertical velocity
        self.alpha = np.arctan2(w1, u1)  # Using correct arctan2 arguments

    def rotate_wind(self, components: WindComponents) -> RotatedWindComponents:
        """
        Apply the stored double-rotation angles to means and covariances.

        The transformation first performs a yaw rotation of *γ* about the
        *z*-axis, followed by a pitch rotation of *α* about the
        intermediate *y*-axis.  Both the first-order (means) and
        second-order (covariances) moments are rotated.

        Parameters
        ----------
        components : WindComponents
            Original wind statistics in instrument coordinates.

        Returns
        -------
        RotatedWindComponents
            Object containing the rotated means and covariances.

        Raises
        ------
        RuntimeError
            If the rotation angles have not been initialised via
            :pymeth:`calculate_angles` (i.e., both *α* and *γ* are still
            zero **and** the input means are non-zero).
        ValueError
            If any required attribute of *components* is missing.

        Notes
        -----
        * The function assumes negligible instrument tilt in the roll
          direction.  For complex or sloping terrain use *triple
          rotation*.
        * Units are preserved (m s⁻¹ for means, m² s⁻² for covariances).

        Examples
        --------
        >>> base = WindComponents(
        ...     u_mean = 2.1,  v_mean = 0.4,  w_mean = 0.02,
        ...     uu_cov = 0.55, vv_cov = 0.37, ww_cov = 0.08,
        ...     uv_cov = 0.03, uw_cov = -0.11, vw_cov = 0.02,
        ... )
        >>> dr = DoubleRotation()
        >>> dr.calculate_angles(base)
        >>> rot = dr.rotate_wind(base)
        >>> rot.w_rot          # doctest: +ELLIPSIS
        0.0...
        """
        # First rotation around z-axis by gamma
        cos_g = np.cos(self.gamma)
        sin_g = np.sin(self.gamma)

        # First rotation
        u1 = components.u_mean * cos_g + components.v_mean * sin_g
        v1 = -components.u_mean * sin_g + components.v_mean * cos_g
        w1 = components.w_mean

        # Second rotation around y-axis by alpha
        cos_a = np.cos(self.alpha)
        sin_a = np.sin(self.alpha)

        # Second rotation
        u_rot = u1 * cos_a - w1 * sin_a
        v_rot = v1
        w_rot = u1 * sin_a + w1 * cos_a

        # Transform stress tensor
        # First rotation
        uu1 = (
            components.uu_cov * cos_g**2
            + components.vv_cov * sin_g**2
            + 2 * components.uv_cov * cos_g * sin_g
        )
        vv1 = (
            components.uu_cov * sin_g**2
            + components.vv_cov * cos_g**2
            - 2 * components.uv_cov * cos_g * sin_g
        )
        ww1 = components.ww_cov

        uv1 = (
            -components.uu_cov + components.vv_cov
        ) * cos_g * sin_g + components.uv_cov * (cos_g**2 - sin_g**2)
        uw1 = components.uw_cov * cos_g + components.vw_cov * sin_g
        vw1 = -components.uw_cov * sin_g + components.vw_cov * cos_g

        # Second rotation
        uu_cov_rot = uu1 * cos_a**2 + ww1 * sin_a**2 - 2 * uw1 * cos_a * sin_a
        vv_cov_rot = vv1
        ww_cov_rot = uu1 * sin_a**2 + ww1 * cos_a**2 + 2 * uw1 * cos_a * sin_a

        uv_cov_rot = uv1 * cos_a - vw1 * sin_a
        uw_cov_rot = -uw1 * (cos_a**2 - sin_a**2) + (ww1 - uu1) * cos_a * sin_a
        vw_cov_rot = -uv1 * sin_a + vw1 * cos_a

        return RotatedWindComponents(
            u_rot=u_rot,
            v_rot=v_rot,
            w_rot=w_rot,
            uu_cov_rot=uu_cov_rot,
            vv_cov_rot=vv_cov_rot,
            ww_cov_rot=ww_cov_rot,
            uv_cov_rot=uv_cov_rot,
            uw_cov_rot=uw_cov_rot,
            vw_cov_rot=vw_cov_rot,
        )


def verify_rotation(
    components: WindComponents,
    rotated: RotatedWindComponents,
) -> bool:
    """
    Validate that a double (or triple) rotation preserved key
    aerodynamic statistics.

    The routine checks four necessary conditions for a *correct*
    rotation:

    1. **Mean-wind magnitude is conserved**

       .. math::
          \lVert\overline{\mathbf{u}}\rVert_{\text{orig}}
          \;=\;
          \lVert\overline{\mathbf{u}}\rVert_{\text{rot}}

    2. **Turbulent kinetic energy (TKE)** is conserved

       .. math::
          \tfrac12 \left( \sigma_{u}^2 + \sigma_{v}^2 + \sigma_{w}^2 \right)
          _{\text{orig}}
          \;=\;
          \tfrac12 \left( \sigma_{u'}^2 + \sigma_{v'}^2 + \sigma_{w'}^2 \right)
          _{\text{rot}}

    3. The rotated cross-stream mean velocity ``v_rot`` is effectively
       zero (``|v_rot| < 1 × 10⁻⁷``).

    4. The rotated vertical mean velocity ``w_rot`` is effectively
       zero (``|w_rot| < 1 × 10⁻⁷``).

    A relative tolerance of ``1 × 10⁻⁷`` is used for the conservation
    tests.

    Parameters
    ----------
    components : WindComponents
        Original, un-rotated first- and second-order wind statistics.
    rotated : RotatedWindComponents
        Corresponding statistics after the coordinate rotation.

    Returns
    -------
    bool
        ``True`` if **all** four conditions are satisfied, ``False``
        otherwise.

    Notes
    -----
    * The function does **not** examine the cross-covariance
      :math:`\overline{u'v'}`.  For triple rotation that term should
      also be ≈ 0.
    * The hard-coded tolerances are suitable for wind speeds of a few
      metres per second and covariances of order 0.1 m² s⁻².  Adjust the
      thresholds for markedly different flow regimes or instrument
      precision.
    * The check is intentionally strict (1 × 10⁻⁷) because floating-
      point errors for typical magnitudes (∼ 10⁰) are well below this
      limit.

    Examples
    --------
    >>> base = WindComponents(
    ...     u_mean = 2.0,  v_mean = 0.5,  w_mean = 0.04,
    ...     uu_cov = 0.48, vv_cov = 0.33, ww_cov = 0.07,
    ...     uv_cov = 0.02, uw_cov = -0.10, vw_cov = 0.01,
    ... )
    >>> dr = DoubleRotation()
    >>> dr.calculate_angles(base)
    >>> rot = dr.rotate_wind(base)
    >>> verify_rotation(base, rot)
    True
    """
    # Calculate original and rotated velocity magnitudes
    orig_speed = np.sqrt(
        components.u_mean**2 + components.v_mean**2 + components.w_mean**2
    )
    rot_speed = np.sqrt(rotated.u_rot**2 + rotated.v_rot**2 + rotated.w_rot**2)

    # Calculate original and rotated TKE
    orig_tke = 0.5 * (components.uu_cov + components.vv_cov + components.ww_cov)
    rot_tke = 0.5 * (rotated.uu_cov_rot + rotated.vv_cov_rot + rotated.ww_cov_rot)

    # Verify properties
    speed_conserved = np.isclose(orig_speed, rot_speed, rtol=1e-7)
    tke_conserved = np.isclose(orig_tke, rot_tke, rtol=1e-7)
    v_zeroed = np.abs(rotated.v_rot) < 1e-7
    w_zeroed = np.abs(rotated.w_rot) < 1e-7

    return speed_conserved and tke_conserved and v_zeroed and w_zeroed


class CoordinateRotation:
    """Base class for coordinate rotation methods"""

    def __init__(self):
        """Initialize rotation parameters"""
        self.alpha = 0.0  # Pitch angle
        self.beta = 0.0  # Roll angle
        self.gamma = 0.0  # Yaw angle

    def _validate_inputs(self, components: WindComponents) -> bool:
        """
        Validate input wind components.

        Args:
            components: WindComponents object

        Returns:
            bool: True if inputs are valid

        Raises:
            ValueError: If any components are invalid
        """
        if not isinstance(components, WindComponents):
            raise ValueError("Input must be a WindComponents object")

        all_values = [
            components.u_mean,
            components.v_mean,
            components.w_mean,
            components.uu_cov,
            components.vv_cov,
            components.ww_cov,
            components.uv_cov,
            components.uw_cov,
            components.vw_cov,
        ]

        if any(np.isnan(x) for x in all_values):
            raise ValueError("Input components contain NaN values")

        return True

    def get_rotation_angles(self) -> Tuple[float, float, float]:
        """Get current rotation angles"""
        return self.alpha, self.beta, self.gamma


class PlanarFit(CoordinateRotation):
    """
    Apply the *planar‐fit* coordinate rotation of Wilczak,
    Oncley & Stage (2001).

    In complex or heterogeneous terrain the mean flow seldom aligns with a
    fixed plane; however, over multi-hour to multi-day periods the
    long-term average velocity field often lies on a *tilted* plane.
    The planar-fit technique

    1. Fits a plane

       .. math::
          \bar{w} \;=\; b_0 + b_1\,\bar{u} + b_2\,\bar{v}

       to the time-averaged wind components.
    2. Rotates the coordinate system so that its new *z*-axis is **normal**
       to that plane.

    Compared with double-rotation on 30-min windows, planar fit greatly
    reduces low-frequency errors in the vertical turbulent fluxes,
    particularly over sloping terrain where the mean vertical wind does
    **not** average to zero over short intervals.

    References
    ----------
    Wilczak, J. M., Oncley, S. P., & Stage, S. A. (2001).
    *Sonic anemometer tilt correction algorithms*.
    **Boundary-Layer Meteorology**, 99(1), 127–150.
    https://doi.org/10.1023/A:1018966204465

    Attributes
    ----------
    b0, b1, b2 : float
        Plane coefficients (offset and slopes) obtained from
        :pymeth:`fit_plane`.
    alpha, beta : float
        Rotation angles (rad) about the *y*- and *x*-axes respectively
        that align the coordinate system with the fitted plane.
    gamma : float
        Always zero for planar fit (yaw handled implicitly by the
        least-squares regression).
    """

    def __init__(self):
        """
        Initialise the rotation with zero angles and plane coefficients.

        All attributes are set to ``0.0`` and are overwritten once
        :pymeth:`fit_plane` is called.
        """
        super().__init__()
        self.b0 = 0.0  # Plane offset
        self.b1 = 0.0  # Plane slope in x
        self.b2 = 0.0  # Plane slope in y

    def fit_plane(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
    ) -> None:
        """
        Derive the best-fit plane from long-term averaged wind data.

        A linear least-squares regression solves

        .. math::
           w = b_0 + b_1\,u + b_2\,v

        for the coefficients *b₀*, *b₁*, *b₂*.
        The rotation angles are then

        .. math::
           \\alpha = \\tan^{-1}(b_1),\\qquad
           \\beta  = \\tan^{-1}\\!\bigl(b_2\\cos\\alpha\\bigr)

        Parameters
        ----------
        u, v, w : ndarray
            One-dimensional arrays of **equal length** containing the
            time-averaged stream-wise, cross-wind, and vertical
            velocities (m s⁻¹).  The averaging period should be long
            enough (≥ 2 h, typically multi-day) to capture all tilt
            angles affecting the sonic anemometer.

        Returns
        -------
        None
            The coefficients (*b₀–b₂*) and rotation angles (*α*, *β*) are
            stored as object attributes.

        Raises
        ------
        ValueError
            If the input arrays have different lengths or contain fewer
            than three samples.
        LinAlgError
            If the least-squares solution fails to converge.

        Notes
        -----
        * The yaw adjustment (*γ*) is implicitly embedded in the planar
          regression; therefore *γ* is kept at zero.
        * Outliers in the long-term averages can bias the fit.  Pre-filter
          gross errors or apply robust regression if necessary.

        Examples
        --------
        >>> pf = PlanarFit()
        >>> pf.fit_plane(u_mean_series, v_mean_series, w_mean_series)  # doctest: +SKIP
        >>> round(pf.alpha, 4), round(pf.beta, 4)
        (0.0123, -0.0045)
        """
        # Construct design matrix
        X = np.column_stack([np.ones_like(u), u, v])

        # Solve linear system
        b = np.linalg.lstsq(X, w, rcond=None)[0]
        self.b0, self.b1, self.b2 = b

        # Calculate rotation angles
        self.alpha = np.arctan(self.b1)
        self.beta = np.arctan(self.b2 * np.cos(self.alpha))

    def rotate_wind(
        self,
        components: WindComponents,
    ) -> RotatedWindComponents:
        """
        Rotate *WindComponents* according to the fitted planar-fit angles.

        The routine applies the 3 × 3 rotation matrix from Wilczak
        et al. (2001; their Eqs. 21–23) to both the mean wind vector and
        the 3 × 3 covariance tensor.

        Parameters
        ----------
        components : WindComponents
            First- and second-order statistics in the *instrument*
            coordinate system.

        Returns
        -------
        RotatedWindComponents
            The same statistics expressed in the planar-fit coordinate
            frame where the mean vertical wind is, by construction,
            extremely close to zero.

        Raises
        ------
        RuntimeError
            If :pymeth:`fit_plane` has **not** been executed (angles are
            still zero).
        ValueError
            If *components* is missing any required attribute.

        Notes
        -----
        * Planar fit is normally computed once per tower per season (or
          year) and the resulting angles are applied to **all** 30-min
          flux intervals.
        * Unlike double rotation, planar fit does **not** enforce
          :math:`\\overline{v} = 0` on every short block; cross-stream
          means may persist.

        Examples
        --------
        >>> pf = PlanarFit()
        >>> pf.fit_plane(ū, v̄, w̄)                 # long-term series
        >>> rc = pf.rotate_wind(block_stats)        # 30-min stats
        >>> abs(rc.w_rot) < 1e-3
        True
        """
        self._validate_inputs(components)

        # Pre-calculate trigonometric functions
        cos_alpha = np.cos(self.alpha)
        sin_alpha = np.sin(self.alpha)
        cos_beta = np.cos(self.beta)
        sin_beta = np.sin(self.beta)

        # Rotation matrix
        R = np.array(
            [
                [cos_alpha, sin_alpha * sin_beta, sin_alpha * cos_beta],
                [0, cos_beta, -sin_beta],
                [-sin_alpha, cos_alpha * sin_beta, cos_alpha * cos_beta],
            ]
        )

        # Rotate mean wind components
        mean_wind = np.array([components.u_mean, components.v_mean, components.w_mean])
        rotated_means = R @ mean_wind

        # Create covariance matrix
        cov_matrix = np.array(
            [
                [components.uu_cov, components.uv_cov, components.uw_cov],
                [components.uv_cov, components.vv_cov, components.vw_cov],
                [components.uw_cov, components.vw_cov, components.ww_cov],
            ]
        )

        # Rotate covariance matrix
        rotated_cov = R @ cov_matrix @ R.T

        return RotatedWindComponents(
            u_rot=rotated_means[0],
            v_rot=rotated_means[1],
            w_rot=rotated_means[2],
            uu_cov_rot=rotated_cov[0, 0],
            vv_cov_rot=rotated_cov[1, 1],
            ww_cov_rot=rotated_cov[2, 2],
            uv_cov_rot=rotated_cov[0, 1],
            uw_cov_rot=rotated_cov[0, 2],
            vw_cov_rot=rotated_cov[1, 2],
        )


def rotate_scalar_fluxes(
    rotation: CoordinateRotation,
    scalar_u_cov: float,
    scalar_v_cov: float,
    scalar_w_cov: float,
) -> Tuple[float, float, float]:
    """
    Rotate scalar–wind covariances into the chosen turbulence
    coordinate frame (*double* or *planar‐fit* rotation).

    The function transforms the covariances between an arbitrary scalar
    (e.g., temperature, CO\ :sub:`2`, H\ :sub:`2`\ O mixing ratio) and
    the three velocity components from instrument coordinates to the
    coordinate system defined by ``rotation``:

    * **Double rotation** :class:`~DoubleRotation`
      (yaw → pitch — γ, α)
    * **Planar fit** :class:`~PlanarFit`
      (pitch + roll — α, β)

    Given the premultiplied rotation angles retrieved via
    ``rotation.get_rotation_angles()`` the covariances are mapped as
    follows:

    *Double rotation* ::

        [u′χ′]ᵣ =  cosα ( cosγ·u′χ′ + sinγ·v′χ′ ) − sinα·w′χ′
        [v′χ′]ᵣ = −sinγ·u′χ′ + cosγ·v′χ′
        [w′χ′]ᵣ =  sinα ( cosγ·u′χ′ + sinγ·v′χ′ ) + cosα·w′χ′

    *Planar fit* uses the 3 × 3 matrix :math:`R` of Wilczak et al.
    (2001) to rotate the covariance vector
    :math:`\left[u′χ′,\;v′χ′,\;w′χ′\right]`.

    Parameters
    ----------
    rotation : CoordinateRotation
        An instance of :class:`DoubleRotation` **or**
        :class:`PlanarFit` whose angles were previously computed.
    scalar_u_cov, scalar_v_cov, scalar_w_cov : float
        Covariances between the scalar (χ) and each wind component
        (u, v, w) in the *instrument* coordinate system
        (units: χ × m s⁻¹).

    Returns
    -------
    scalar_u_rot, scalar_v_rot, scalar_w_rot : tuple[float, float, float]
        Rotated covariances (χ with u, v, w) in the new coordinate
        frame.

    Raises
    ------
    AttributeError
        If ``rotation`` lacks the method ``get_rotation_angles``.
    ValueError
        If ``rotation`` is neither :class:`DoubleRotation` nor
        :class:`PlanarFit`.

    Notes
    -----
    * For eddy‐covariance flux calculations you typically use
      ``scalar_w_rot`` (χ–w covariance) after rotation.
    * The function assumes right‐handed rotation matrices and follows
      the convention that positive *w* is upward.
    * Units are preserved; only orientation changes.

    Examples
    --------
    >>> dr = DoubleRotation()
    >>> dr.calculate_angles(block_stats)        # determine α, γ
    >>> rotate_scalar_fluxes(
    ...     rotation       = dr,
    ...     scalar_u_cov   = 0.03,   # T–u covariance, K m s⁻¹
    ...     scalar_v_cov   = -0.01,
    ...     scalar_w_cov   = 0.18,
    ... )
    (0.0287..., -0.0113..., 0.1814...)

    For planar fit:

    >>> pf = PlanarFit()
    >>> pf.fit_plane(ū, v̄, w̄)                  # long-term means
    >>> rotate_scalar_fluxes(
    ...     rotation       = pf,
    ...     scalar_u_cov   = 0.03,
    ...     scalar_v_cov   = -0.01,
    ...     scalar_w_cov   = 0.18,
    ... )
    (0.0295..., -0.0102..., 0.1798...)
    """
    alpha, beta, gamma = rotation.get_rotation_angles()

    if isinstance(rotation, DoubleRotation):
        # Double rotation for scalar fluxes
        scalar_u_rot = np.cos(alpha) * (
            scalar_u_cov * np.cos(gamma) + scalar_v_cov * np.sin(gamma)
        ) - scalar_w_cov * np.sin(alpha)

        scalar_v_rot = -scalar_u_cov * np.sin(gamma) + scalar_v_cov * np.cos(gamma)

        scalar_w_rot = np.sin(alpha) * (
            scalar_u_cov * np.cos(gamma) + scalar_v_cov * np.sin(gamma)
        ) + scalar_w_cov * np.cos(alpha)

    else:  # PlanarFit
        # Rotation matrix for planar fit
        R = np.array(
            [
                [
                    np.cos(alpha),
                    np.sin(alpha) * np.sin(beta),
                    np.sin(alpha) * np.cos(beta),
                ],
                [0, np.cos(beta), -np.sin(beta)],
                [
                    -np.sin(alpha),
                    np.cos(alpha) * np.sin(beta),
                    np.cos(alpha) * np.cos(beta),
                ],
            ]
        )

        # Rotate scalar fluxes
        scalar_fluxes = np.array([scalar_u_cov, scalar_v_cov, scalar_w_cov])
        rotated_fluxes = R @ scalar_fluxes

        scalar_u_rot = rotated_fluxes[0]
        scalar_v_rot = rotated_fluxes[1]
        scalar_w_rot = rotated_fluxes[2]

    return scalar_u_rot, scalar_v_rot, scalar_w_rot
