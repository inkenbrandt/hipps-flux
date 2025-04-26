```python
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

from typing import Tuple, Optional, Dict, Union
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

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

class KormannMeixnerFootprint(FootprintModel):
    """
    Implementation of Kormann and Meixner (2001) analytical footprint model.
    
    This model provides:
    1. Analytical solution for cross-wind integrated footprint
    2. 2D footprint distribution
    """
    
    def __init__(self, params: FootprintParams):
        """Initialize Kormann-Meixner model"""
        super().__init__(params)
        self.r = None  # Shape factor
        self.u = None  # Power law wind profile exponent
        
    def _calculate_parameters(self) -> None:
        """Calculate model parameters"""
        # Calculate power law exponents
        if self.zL > 0:  # Stable
            self.u = 0.09 * np.log(self.params.z_m/self.params.z_0) * self.zL
            self.r = 2 + self.u - 1/(1 + 5*self.zL)
        else:  # Unstable
            self.u = 0.09 * np.log(self.params.z_m/self.params.z_0) / (1 - 16*self.zL)**0.25
            self.r = 2 + self.u - (1 - 24*self.zL)/(1 - 16*self.zL)
    
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
            - f_ci: Cross-wind integrated footprint
            - f_2d: 2D footprint if y_points provided
        """
        self._calculate_parameters()
        
        # Calculate characteristic length and velocity
        U = self.params.u_star * np.log(self.params.z_m/self.params.z_0) / 0.4
        x_scale = self.params.u_star * self.params.z_m / (2 * 0.4 * U)
        
        # Calculate peak distance
        x_peak = x_scale * (self.r - 1)
        
        # Setup distance array if not provided
        if x_points is None:
            x_points = np.linspace(0, x_peak * 5, 100)
            
        # Calculate cross-wind integrated footprint
        f_ci = self._crosswind_integrated_footprint(x_points, x_scale)
        
        result = {
            'x_peak': x_peak,
            'x': x_points,
            'f_ci': f_ci
        }
        
        # Calculate 2D footprint if y_points provided
        if y_points is not None:
            f_2d = self._calculate_2d_footprint(x_points, y_points, x_scale)
            result.update({
                'y': y_points,
                'f_2d': f_2d
            })
            
        return result
    
    def _crosswind_integrated_footprint(
        self,
        x: np.ndarray,
        x_scale: float
    ) -> np.ndarray:
        """Calculate cross-wind integrated footprint"""
        # Calculate footprint following Kormann and Meixner (2001)
        xi = x/x_scale
        f = (1/(x_scale * gamma(self.r)) * 
             (xi/self.r)**(self.r-1) * 
             np.exp(-xi/self.r))
        return f
    
    def _calculate_2d_footprint(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_scale: float
    ) -> np.ndarray:
        """Calculate 2D footprint distribution"""
        # Calculate cross-wind integrated footprint
        f_ci = self._crosswind_integrated_footprint(x, x_scale)
        
        # Create 2D grid
        X, Y = np.meshgrid(x, y)
        
        # Calculate dispersion parameter
        sigma_y = 0.3 * x * (1 + 0.28/x_scale)**(-1/4)
        
        # Calculate 2D distribution
        f_2d = np.zeros_like(X)
        for i in range(len(y)):
            f_2d[i,:] = (f_ci / (np.sqrt(2*np.pi) * sigma_y) * 
                        np.exp(-0.5 * (Y[i,:]/sigma_y)**2))
        
        return f_2d

def calculate_footprint_climatology(
    model: str,
    params_list: List[FootprintParams],
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
    # Initialize footprint model for each observation
    if model.lower() == 'kljun':
        ModelClass = KljunFootprint
    elif model.lower() == 'kormann-meixner':
        ModelClass = KormannMeixnerFootprint
    else:
        raise ValueError(f"Unknown footprint model: {model}")
        
    # Calculate individual footprints
    footprints = []
    for params in params_list:
        fp_model = ModelClass(params)
        result = fp_model.calculate_footprint(domain['x'], domain['y'])
        footprints.append(result['f_2d'])
        
    # Calculate average footprint
    f_clim = np.mean(footprints, axis=0)
    
    # Calculate 90% contribution contour
    f_sort = np.sort(f_clim.flatten())[::-1]
    cumsum = np.cumsum(f_sort)
    cumsum /= cumsum[-1]
    threshold = f_sort[np.searchsorted(cumsum, 0.9)]
    contrib_90 = f_clim >= threshold
    
    return {
        'f_clim': f_clim,
        'x': domain['x'],
        'y': domain['y'],
        'contrib_90': contrib_90
    }
```
