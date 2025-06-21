
"""
Footprint modeling for eddy covariance measurements.

This module implements two footprint models:
1. Kljun et al. (2004, 2015) parameterization for 1D and 2D footprints
2. Kormann and Meixner (2001) analytical model

References:
    Kljun et al. (2004) A Simple Parameterisation for Flux Footprint Predictions
    Kljun et al. (2015) A simple two-dimensional parameterisation for Flux Footprint Prediction (FFP)
    Kormann and Meixner (2001) An Analytical Footprint Model For Non-Neutral Stratification
"""

from typing import Tuple, Optional, Dict, Union, List
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.special import gamma
@dataclass
class FootprintParams:
    """Parameters for footprint calculations"""
    z_m: float  # Measurement height (m)
    z_0: float  # Roughness length (m)
    u_star: float  # Friction velocity (m/s)
    sigma_v: float  # Standard deviation of lateral velocity (m/s)
    h_abl: float  # Boundary layer height (m)
    L: float  # Obukhov length (m)
    wind_dir: float  # Wind direction (degrees)
    
    def __post_init__(self):
        """Validate parameters"""
        if self.z_m <= self.z_0:
            raise ValueError("Measurement height must be greater than roughness length")
        if self.u_star <= 0:
            raise ValueError("Friction velocity must be positive")
        if self.h_abl <= self.z_m:
            raise ValueError("ABL height must be greater than measurement height")
        if not 0 <= self.wind_dir <= 360:
            raise ValueError("Wind direction must be between 0 and 360 degrees")

class FootprintModel(ABC):
    """Abstract base class for footprint models"""
    
    def __init__(self, params: FootprintParams):
        """
        Initialize footprint model.
        
        Args:
            params: FootprintParams object containing required parameters
        """
        self.params = params
        self._validate_stability()
    
    @abstractmethod
    def calculate_footprint(self) -> Dict[str, np.ndarray]:
        """Calculate footprint contribution"""
        pass
    
    def _validate_stability(self) -> None:
        """Check stability conditions"""
        self.zL = self.params.z_m / self.params.L
        if abs(self.zL) > 1:
            raise ValueError("Model may not be valid for strongly stable/unstable conditions")

class KljunFootprint(FootprintModel):
    """
    Implementation of Kljun et al. (2004, 2015) footprint parameterization.
    
    This model provides:
    1. Peak contribution distance
    2. 80% footprint distance
    3. Cross-wind integrated footprint
    4. 2D footprint for given domain
    """
    
    def __init__(self, params: FootprintParams):
        """Initialize Kljun model"""
        super().__init__(params)
        self.scaled_length = None  # Will store characteristic length scale
        
    def _calculate_scaling(self) -> None:
        """Calculate scaling parameters following Kljun et al."""
        # Calculate characteristic length scale
        self.scaled_length = (
            self.params.u_star * self.params.h_abl /
            abs(self.params.L if self.params.L != 0 else 1e-6)
        )
        
        # Calculate scaled sigmas
        self.scaled_sigma_v = self.params.sigma_v / self.params.u_star
        
        # Calculate scaled measurement height
        self.scaled_z = self.params.z_m / self.params.h_abl
        
    def calculate_footprint(
        self,
        x_points: Optional[np.ndarray] = None,
        y_points: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate footprint function.
        
        Args:
            x_points: Optional array of x-coordinates (m)
            y_points: Optional array of y-coordinates (m)
            
        Returns:
            Dictionary containing:
            - x_peak: Distance of peak contribution
            - x_80: 80% footprint distance
            - f_ci: Cross-wind integrated footprint
            - f_2d: 2D footprint if x_points and y_points provided
        """
        self._calculate_scaling()
        
        # Calculate peak distance
        x_peak = self._calculate_peak_distance()
        
        # Calculate 80% distance
        x_80 = self._calculate_cumulative_distance(0.8)
        
        # Calculate cross-wind integrated footprint
        if x_points is None:
            x_points = np.linspace(0, x_80 * 1.5, 100)
        f_ci = self._crosswind_integrated_footprint(x_points)
        
        result = {
            'x_peak': x_peak,
            'x_80': x_80,
            'x': x_points,
            'f_ci': f_ci
        }
        
        # Calculate 2D footprint if y_points provided
        if y_points is not None:
            f_2d = self._calculate_2d_footprint(x_points, y_points)
            result.update({
                'y': y_points,
                'f_2d': f_2d
            })
            
        return result
    
    def _calculate_peak_distance(self) -> float:
        """Calculate distance of peak contribution"""
        a = 0.175 * (1 - np.exp(-self.scaled_length/50))
        b = 3.418 / np.log(self.params.z_m/self.params.z_0)
        x_peak = self.params.h_abl * a * b
        return x_peak
    
    def _calculate_cumulative_distance(self, fraction: float) -> float:
        """
        Calculate distance for given cumulative contribution.
        
        Args:
            fraction: Fraction of total contribution (0-1)
        """
        a = 0.175 * (1 - np.exp(-self.scaled_length/50))
        b = 3.418 / np.log(self.params.z_m/self.params.z_0)
        c = -np.log(1 - fraction)
        return self.params.h_abl * a * b * c
    
    def _crosswind_integrated_footprint(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cross-wind integrated footprint.
        
        Args:
            x: Array of distances (m)
            
        Returns:
            Array of footprint values
        """
        x_scaled = x / self.params.h_abl
        a = 0.175 * (1 - np.exp(-self.scaled_length/50))
        b = 3.418 / np.log(self.params.z_m/self.params.z_0)
        
        # Calculate footprint following Kljun et al. (2015)
        f = (1 / (a * b * self.params.h_abl) * 
             np.exp(-x_scaled / (a * b)))
        
        return f
    
    def _calculate_2d_footprint(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Calculate 2D footprint distribution.
        
        Args:
            x: Array of x-coordinates (m)
            y: Array of y-coordinates (m)
            
        Returns:
            2D array of footprint values
        """
        # Calculate cross-wind integrated footprint
        f_ci = self._crosswind_integrated_footprint(x)
        
        # Create 2D grid
        X, Y = np.meshgrid(x, y)
        
        # Calculate cross-wind dispersion
        sigma_y = self.scaled_sigma_v * x
        
        # Calculate 2D distribution
        f_2d = np.zeros_like(X)
        for i in range(len(y)):
            f_2d[i,:] = (f_ci / (np.sqrt(2*np.pi) * sigma_y) * 
                        np.exp(-0.5 * (Y[i,:]/sigma_y)**2))
        
        return f_2d


class KormannMeixnerFootprint:
    """Implementation of Kormann and Meixner (2001) analytical footprint model."""

    def __init__(self, params: "FootprintParams"):
        """Initialize footprint model."""
        self.params = params
        self.r = None  # Shape factor
        self.u = None  # Power law wind profile exponent
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate stability condition."""
        self.zL = self.params.z_m / self.params.L if self.params.L != 0 else 0.0001
        if abs(self.zL) > 2:
            raise ValueError(f"Model may not be valid for |z/L| > 2, got {abs(self.zL):.2f}")

    def _calculate_parameters(self) -> None:
        """Calculate model parameters based on stability."""
        # Get measurement height normalized by roughness length
        zeta = max(self.params.z_m / self.params.z_0, 2.0)  # Ensure minimum ratio

        if self.params.L < 0:  # Unstable conditions
            # Calculate stability parameter with bounds
            zL = max(min(-self.zL, 2), 0.01)  # Limit range for stability functions

            # Modified power law exponent for unstable conditions
            phi_m = (1 - 16 * zL) ** (-0.25)
            self.u = max(0.09 * np.log(zeta) * phi_m, 0.1)  # Ensure positive

            # Shape factor for unstable conditions
            self.r = max(2.0 + self.u - (1 - 24 * zL) / (1 - 16 * zL), 1.5)

        else:  # Stable and neutral conditions
            # Calculate stability parameter with bounds
            zL = min(max(self.zL, 0), 1)  # Limit range for stability functions

            # Modified power law exponent for stable conditions
            phi_m = 1 + 5 * zL
            self.u = max(0.09 * np.log(zeta) * phi_m, 0.1)  # Ensure positive

            # Shape factor for stable conditions
            self.r = max(2.0 + self.u - 1 / (1 + 5 * zL), 1.5)

    def calculate_footprint(
            self,
            x_points: np.ndarray,
            y_points: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Calculate footprint function."""
        self._calculate_parameters()

        # Calculate characteristic length and velocity scales
        U = max(self.params.u_star * np.log(self.params.z_m / self.params.z_0) / 0.4, 0.1)
        x_scale = max(self.params.u_star * self.params.z_m / (2 * 0.4 * U), 0.1)

        # Calculate peak distance with guaranteed positive value
        x_peak = x_scale * max(self.r - 1, 0.1)  # Ensure positive peak distance

        if self.params.L < 0:  # Unstable conditions
            stability_factor = (1 - min(16 * abs(self.zL), 0.9)) ** (-0.25)
            x_peak *= max(stability_factor, 0.1)
        else:  # Stable conditions
            stability_factor = (1 + min(5 * self.zL, 2)) ** 0.25
            x_peak *= max(stability_factor, 0.1)

        # Calculate cross-wind integrated footprint
        xi = x_points / x_scale

        if self.params.L < 0:  # Unstable conditions
            f_ci = (1 / (x_scale * gamma(self.r)) *
                    (xi / (self.r)) ** (self.r - 1) *
                    np.exp(-xi / (self.r)))
        else:  # Stable conditions
            f_ci = (1 / (x_scale * gamma(self.r)) *
                    (xi / (self.r)) ** (self.r - 1) *
                    np.exp(-xi / (self.r)))

        # Ensure non-negative values
        f_ci = np.maximum(f_ci, 0)

        result = {
            'x_peak': float(x_peak),  # Ensure Python float
            'x': x_points,
            'f_ci': f_ci
        }

        # Calculate 2D footprint if y_points provided
        if y_points is not None:
            # Calculate dispersion parameter
            if self.params.L < 0:
                sigma_v_scale = 1.3  # Increased dispersion for unstable conditions
            else:
                sigma_v_scale = 0.7  # Decreased dispersion for stable conditions

            # Calculate crosswind dispersion with minimum value
            sigma_y = np.maximum(
                sigma_v_scale * self.params.sigma_v * x_points / U *
                (1 + 1.0 / x_scale) ** (-0.25),
                0.1  # Minimum dispersion
            )

            # Create 2D grid
            X, Y = np.meshgrid(x_points, y_points)

            # Calculate 2D distribution
            f_2d = np.zeros_like(X)
            for i in range(len(y_points)):
                f_2d[i, :] = (f_ci / (np.sqrt(2 * np.pi) * sigma_y) *
                              np.exp(-0.5 * (Y[i, :] / sigma_y) ** 2))

            # Ensure non-negative values
            f_2d = np.maximum(f_2d, 0)

            result.update({
                'y': y_points,
                'f_2d': f_2d
            })

        return result


def calculate_footprint_climatology(
        model: str,
        params_list: list,
        domain: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Calculate footprint climatology from multiple observations.

    Args:
        model: Footprint model to use ('kljun' or 'kormann-meixner')
        params_list: List of FootprintParams for each observation
        domain: Dictionary with 'x' and 'y' arrays defining domain

    Returns:
        Dictionary containing:
        - f_clim: Climatological footprint
        - x: x-coordinates
        - y: y-coordinates
        - contrib_90: 90% contribution contour
    """
    # Validate inputs
    if not params_list:
        raise ValueError("Empty parameters list")
    if 'x' not in domain or 'y' not in domain:
        raise ValueError("Domain must contain 'x' and 'y' arrays")

    # Initialize footprint model for each observation
    if model.lower() == 'kljun':
        ModelClass = KljunFootprint
    elif model.lower() == 'kormann-meixner':
        ModelClass = KormannMeixnerFootprint
    else:
        raise ValueError(f"Unknown footprint model: {model}")

    # Calculate individual footprints
    footprints = []
    x_points = domain['x']
    y_points = domain['y']

    for params in params_list:
        try:
            fp_model = ModelClass(params)
            result = fp_model.calculate_footprint(x_points, y_points)

            if 'f_2d' in result:
                # Normalize individual footprint
                f_2d = result['f_2d']
                dx = x_points[1] - x_points[0]
                dy = y_points[1] - y_points[0]
                integral = np.trapz(np.trapz(f_2d, x_points, axis=1), y_points)

                if integral > 0:
                    f_2d_norm = f_2d / integral
                    footprints.append(f_2d_norm)

        except (ValueError, RuntimeWarning) as e:
            print(f"Warning: Skipping invalid parameter set: {e}")
            continue

    if not footprints:
        raise ValueError("No valid footprints calculated")

    # Calculate average footprint
    f_clim = np.mean(footprints, axis=0)

    # Normalize climatological footprint
    dx = x_points[1] - x_points[0]
    dy = y_points[1] - y_points[0]
    integral = np.trapz(np.trapz(f_clim, x_points, axis=1), y_points)

    if integral > 0:
        f_clim = f_clim / integral
    else:
        raise ValueError("Zero or negative integral in climatological footprint")

    # Calculate 90% contribution contour
    f_sort = np.sort(f_clim.flatten())[::-1]  # Sort in descending order
    cumsum = np.cumsum(f_sort)
    cumsum = cumsum / cumsum[-1]  # Normalize
    threshold = f_sort[np.searchsorted(cumsum, 0.9)]
    contrib_90 = f_clim >= threshold

    return {
        'f_clim': f_clim,
        'x': x_points,
        'y': y_points,
        'contrib_90': contrib_90
    }
