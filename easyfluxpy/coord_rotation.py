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

@dataclass
class WindComponents:
    """Container for wind velocity components and statistics"""
    u_mean: float  # Mean streamwise velocity
    v_mean: float  # Mean crosswind velocity
    w_mean: float  # Mean vertical velocity
    uu_cov: float  # u'u' covariance 
    vv_cov: float  # v'v' covariance
    ww_cov: float  # w'w' covariance
    uv_cov: float  # u'v' covariance
    uw_cov: float  # u'w' covariance
    vw_cov: float  # v'w' covariance

@dataclass
class RotatedWindComponents:
    """Container for rotated wind components and fluxes"""
    u_rot: float  # Rotated streamwise velocity
    v_rot: float  # Rotated crosswind velocity 
    w_rot: float  # Rotated vertical velocity
    uu_cov_rot: float  # Rotated u'u' covariance
    vv_cov_rot: float  # Rotated v'v' covariance  
    ww_cov_rot: float  # Rotated w'w' covariance
    uv_cov_rot: float  # Rotated u'v' covariance
    uw_cov_rot: float  # Rotated u'w' covariance
    vw_cov_rot: float  # Rotated v'w' covariance

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

class DoubleRotation(CoordinateRotation):
    """
    Implementation of Tanner & Thurtell (1969) double rotation method.
    
    Double rotation performs:
    1. First rotation around z-axis (gamma) to align x-axis with mean wind
    2. Second rotation around y-axis (alpha) to make mean vertical wind zero 
    """
    
    def calculate_angles(self, components: WindComponents) -> None:
        """
        Calculate rotation angles from wind components.
        
        Args:
            components: WindComponents object with raw measurements
        """
        self._validate_inputs(components)
        
        # Calculate gamma (rotation around z-axis)
        self.gamma = np.arctan2(components.v_mean, components.u_mean)
        
        # Calculate alpha (rotation around y-axis)
        u_plane = np.sqrt(components.u_mean**2 + components.v_mean**2)
        self.alpha = np.arctan2(components.w_mean, u_plane)
        
        # Beta is zero for double rotation
        self.beta = 0.0

    def rotate_wind(self, components: WindComponents) -> RotatedWindComponents:
        """
        Apply double rotation to wind components.
        
        Args:
            components: WindComponents object with raw measurements
            
        Returns:
            RotatedWindComponents object with rotated values
        """
        self._validate_inputs(components)
        
        # Pre-calculate trigonometric functions
        cos_gamma = np.cos(self.gamma)
        sin_gamma = np.sin(self.gamma)
        cos_alpha = np.cos(self.alpha)
        sin_alpha = np.sin(self.alpha)
        
        # Rotate mean wind components
        u_mean_r = (cos_alpha * 
                   (components.u_mean * cos_gamma + 
                    components.v_mean * sin_gamma) - 
                   components.w_mean * sin_alpha)
        v_mean_r = 0  # By definition for double rotation
        w_mean_r = (sin_alpha * 
                   (components.u_mean * cos_gamma + 
                    components.v_mean * sin_gamma) + 
                   components.w_mean * cos_alpha)
        
        # Intermediate calculations for covariances
        uw_tmp = (components.uu_cov * cos_gamma**2 + 
                 components.vv_cov * sin_gamma**2)
        vw_tmp = (components.uw_cov * cos_gamma + 
                 components.vw_cov * sin_gamma)
        
        # Rotate variances
        uu_cov_r = (cos_alpha**2 * uw_tmp +
                    components.ww_cov * sin_alpha**2 +
                    components.uv_cov * cos_alpha**2 * np.sin(2*self.gamma) -
                    np.sin(2*self.alpha) * vw_tmp)
        
        vv_cov_r = (components.uu_cov * sin_gamma**2 +
                    components.vv_cov * cos_gamma**2 -
                    components.uv_cov * np.sin(2*self.gamma))
        
        ww_cov_r = (sin_alpha**2 * uw_tmp +
                    components.ww_cov * cos_alpha**2 +
                    components.uv_cov * sin_alpha**2 * np.sin(2*self.gamma) +
                    np.sin(2*self.alpha) * vw_tmp)
        
        # Rotate covariances
        uv_cov_r = (-0.5 * (components.uu_cov - components.vv_cov) * 
                    cos_alpha * np.sin(2*self.gamma) +
                    components.uv_cov * cos_alpha * np.cos(2*self.gamma) +
                    sin_alpha * (components.uw_cov * sin_gamma - 
                                components.vw_cov * cos_gamma))
        
        uw_cov_r = (0.5 * np.sin(2*self.alpha) * 
                    (uw_tmp - components.ww_cov + 
                     components.uv_cov * np.sin(2*self.gamma)) +
                    np.cos(2*self.alpha) * vw_tmp)
        
        vw_cov_r = (-sin_alpha * 
                    (0.5 * (components.uu_cov - components.vv_cov) * 
                     np.sin(2*self.gamma) - 
                     components.uv_cov * np.cos(2*self.gamma)) -
                    cos_alpha * 
                    (components.uw_cov * sin_gamma - 
                     components.vw_cov * cos_gamma))
        
        return RotatedWindComponents(
            u_rot=u_mean_r, v_rot=v_mean_r, w_rot=w_mean_r,
            uu_cov_rot=uu_cov_r, vv_cov_rot=vv_cov_r, ww_cov_rot=ww_cov_r,
            uv_cov_rot=uv_cov_r, uw_cov_rot=uw_cov_r, vw_cov_rot=vw_cov_r
        )

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
