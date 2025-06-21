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


class DoubleRotation:
    def __init__(self):
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0

    def calculate_angles(self, components: WindComponents) -> None:
        """
        Calculate double rotation angles.
        First rotation (gamma) aligns x-axis with mean wind.
        Second rotation (alpha) zeros mean vertical velocity.
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
        Apply double rotation to wind components and fluxes.
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
        uu1 = components.uu_cov * cos_g ** 2 + components.vv_cov * sin_g ** 2 + 2 * components.uv_cov * cos_g * sin_g
        vv1 = components.uu_cov * sin_g ** 2 + components.vv_cov * cos_g ** 2 - 2 * components.uv_cov * cos_g * sin_g
        ww1 = components.ww_cov

        uv1 = (-components.uu_cov + components.vv_cov) * cos_g * sin_g + components.uv_cov * (cos_g ** 2 - sin_g ** 2)
        uw1 = components.uw_cov * cos_g + components.vw_cov * sin_g
        vw1 = -components.uw_cov * sin_g + components.vw_cov * cos_g

        # Second rotation
        uu_cov_rot = uu1 * cos_a ** 2 + ww1 * sin_a ** 2 - 2 * uw1 * cos_a * sin_a
        vv_cov_rot = vv1
        ww_cov_rot = uu1 * sin_a ** 2 + ww1 * cos_a ** 2 + 2 * uw1 * cos_a * sin_a

        uv_cov_rot = uv1 * cos_a - vw1 * sin_a
        uw_cov_rot = -uw1 * (cos_a ** 2 - sin_a ** 2) + (ww1 - uu1) * cos_a * sin_a
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
            vw_cov_rot=vw_cov_rot
        )


def verify_rotation(components: WindComponents, rotated: RotatedWindComponents) -> bool:
    """
    Verify the rotation satisfies key properties.
    """
    # Calculate original and rotated velocity magnitudes
    orig_speed = np.sqrt(components.u_mean ** 2 + components.v_mean ** 2 + components.w_mean ** 2)
    rot_speed = np.sqrt(rotated.u_rot ** 2 + rotated.v_rot ** 2 + rotated.w_rot ** 2)

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
        self.beta = 0.0   # Roll angle
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
            components.u_mean, components.v_mean, components.w_mean,
            components.uu_cov, components.vv_cov, components.ww_cov,
            components.uv_cov, components.uw_cov, components.vw_cov
        ]

        if any(np.isnan(x) for x in all_values):
            raise ValueError("Input components contain NaN values")

        return True

    def get_rotation_angles(self) -> Tuple[float, float, float]:
        """Get current rotation angles"""
        return self.alpha, self.beta, self.gamma


class PlanarFit(CoordinateRotation):
    """
    Implementation of Wilczak et al. (2001) planar fit method.

    Planar fit:
    1. Fits a plane to long-term averaged wind data
    2. Rotates coordinate system so z-axis is normal to this plane
    """

    def __init__(self):
        """Initialize planar fit parameters"""
        super().__init__()
        self.b0 = 0.0  # Plane offset
        self.b1 = 0.0  # Plane slope in x
        self.b2 = 0.0  # Plane slope in y

    def fit_plane(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> None:
        """
        Fit plane to wind velocity components.

        Args:
            u: Array of streamwise velocities
            v: Array of crosswind velocities
            w: Array of vertical velocities
        """
        # Construct design matrix
        X = np.column_stack([np.ones_like(u), u, v])

        # Solve linear system
        b = np.linalg.lstsq(X, w, rcond=None)[0]
        self.b0, self.b1, self.b2 = b

        # Calculate rotation angles
        self.alpha = np.arctan(self.b1)
        self.beta = np.arctan(self.b2 * np.cos(self.alpha))

    def rotate_wind(self, components: WindComponents) -> RotatedWindComponents:
        """
        Apply planar fit rotation to wind components.

        Args:
            components: WindComponents object with raw measurements

        Returns:
            RotatedWindComponents object with rotated values
        """
        self._validate_inputs(components)

        # Pre-calculate trigonometric functions
        cos_alpha = np.cos(self.alpha)
        sin_alpha = np.sin(self.alpha)
        cos_beta = np.cos(self.beta)
        sin_beta = np.sin(self.beta)

        # Rotation matrix
        R = np.array([
            [cos_alpha, sin_alpha*sin_beta, sin_alpha*cos_beta],
            [0, cos_beta, -sin_beta],
            [-sin_alpha, cos_alpha*sin_beta, cos_alpha*cos_beta]
        ])

        # Rotate mean wind components
        mean_wind = np.array([
            components.u_mean,
            components.v_mean,
            components.w_mean
        ])
        rotated_means = R @ mean_wind

        # Create covariance matrix
        cov_matrix = np.array([
            [components.uu_cov, components.uv_cov, components.uw_cov],
            [components.uv_cov, components.vv_cov, components.vw_cov],
            [components.uw_cov, components.vw_cov, components.ww_cov]
        ])

        # Rotate covariance matrix
        rotated_cov = R @ cov_matrix @ R.T

        return RotatedWindComponents(
            u_rot=rotated_means[0],
            v_rot=rotated_means[1],
            w_rot=rotated_means[2],
            uu_cov_rot=rotated_cov[0,0],
            vv_cov_rot=rotated_cov[1,1],
            ww_cov_rot=rotated_cov[2,2],
            uv_cov_rot=rotated_cov[0,1],
            uw_cov_rot=rotated_cov[0,2],
            vw_cov_rot=rotated_cov[1,2]
        )

def rotate_scalar_fluxes(
    rotation: CoordinateRotation,
    scalar_u_cov: float,
    scalar_v_cov: float,
    scalar_w_cov: float
) -> Tuple[float, float, float]:
    """
    Rotate scalar fluxes using pre-calculated rotation angles.

    Args:
        rotation: CoordinateRotation object with calculated angles
        scalar_u_cov: Covariance of scalar with u
        scalar_v_cov: Covariance of scalar with v
        scalar_w_cov: Covariance of scalar with w

    Returns:
        Tuple containing rotated scalar covariances (u, v, w)
    """
    alpha, beta, gamma = rotation.get_rotation_angles()

    if isinstance(rotation, DoubleRotation):
        # Double rotation for scalar fluxes
        scalar_u_rot = (np.cos(alpha) *
                       (scalar_u_cov * np.cos(gamma) +
                        scalar_v_cov * np.sin(gamma)) -
                       scalar_w_cov * np.sin(alpha))

        scalar_v_rot = (-scalar_u_cov * np.sin(gamma) +
                       scalar_v_cov * np.cos(gamma))

        scalar_w_rot = (np.sin(alpha) *
                       (scalar_u_cov * np.cos(gamma) +
                        scalar_v_cov * np.sin(gamma)) +
                       scalar_w_cov * np.cos(alpha))

    else:  # PlanarFit
        # Rotation matrix for planar fit
        R = np.array([
            [np.cos(alpha), np.sin(alpha)*np.sin(beta), np.sin(alpha)*np.cos(beta)],
            [0, np.cos(beta), -np.sin(beta)],
            [-np.sin(alpha), np.cos(alpha)*np.sin(beta), np.cos(alpha)*np.cos(beta)]
        ])

        # Rotate scalar fluxes
        scalar_fluxes = np.array([scalar_u_cov, scalar_v_cov, scalar_w_cov])
        rotated_fluxes = R @ scalar_fluxes

        scalar_u_rot = rotated_fluxes[0]
        scalar_v_rot = rotated_fluxes[1]
        scalar_w_rot = rotated_fluxes[2]

    return scalar_u_rot, scalar_v_rot, scalar_w_rot
