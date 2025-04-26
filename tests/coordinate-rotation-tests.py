import numpy as np
import pytest
from eddy_covariance.coord_rotation import (
    WindComponents, RotatedWindComponents, 
    CoordinateRotation, DoubleRotation, PlanarFit,
    rotate_scalar_fluxes
)

# Test data setup fixtures
@pytest.fixture
def wind_components():
    """Create sample wind components for testing"""
    return WindComponents(
        u_mean=3.0,
        v_mean=1.0, 
        w_mean=0.1,
        uu_cov=0.5,
        vv_cov=0.3,
        ww_cov=0.2,
        uv_cov=0.1,
        uw_cov=-0.15,
        vw_cov=-0.05
    )

@pytest.fixture
def planar_fit_data():
    """Create sample data for planar fit testing"""
    # Generate synthetic wind data
    n_samples = 1000
    u = np.random.normal(5.0, 1.0, n_samples)
    v = np.random.normal(0.0, 1.0, n_samples)
    w = 0.1*u + 0.05*v + np.random.normal(0.0, 0.1, n_samples)
    return u, v, w

class TestWindComponents:
    """Tests for WindComponents data class"""
    
    def test_wind_components_creation(self):
        """Test creation of WindComponents object"""
        components = WindComponents(
            u_mean=1.0, v_mean=0.5, w_mean=0.1,
            uu_cov=0.2, vv_cov=0.15, ww_cov=0.1,
            uv_cov=0.05, uw_cov=-0.1, vw_cov=-0.05
        )
        assert components.u_mean == 1.0
        assert components.v_mean == 0.5
        assert components.w_mean == 0.1
        assert components.uu_cov == 0.2
        assert components.vv_cov == 0.15
        assert components.ww_cov == 0.1
        assert components.uv_cov == 0.05
        assert components.uw_cov == -0.1
        assert components.vw_cov == -0.05

class TestDoubleRotation:
    """Tests for double rotation implementation"""
    
    def test_double_rotation_initialization(self, wind_components):
        """Test initialization of double rotation"""
        rotation = DoubleRotation()
        assert rotation.alpha == 0.0
        assert rotation.beta == 0.0
        assert rotation.gamma == 0.0
        
    def test_calculate_angles(self, wind_components):
        """Test calculation of rotation angles"""
        rotation = DoubleRotation()
        rotation.calculate_angles(wind_components)
        
        # Check gamma calculation (around z-axis)
        expected_gamma = np.arctan2(wind_components.v_mean, 
                                  wind_components.u_mean)
        assert np.isclose(rotation.gamma, expected_gamma)
        
        # Check alpha calculation (around y-axis)
        u_plane = np.sqrt(wind_components.u_mean**2 + 
                         wind_components.v_mean**2)
        expected_alpha = np.arctan2(wind_components.w_mean, u_plane)
        assert np.isclose(rotation.alpha, expected_alpha)
        
        # Beta should remain zero for double rotation
        assert rotation.beta == 0.0
        
    def test_rotate_wind(self, wind_components):
        """Test wind component rotation"""
        rotation = DoubleRotation()
        rotation.calculate_angles(wind_components)
        rotated = rotation.rotate_wind(wind_components)
        
        # Check rotated means
        assert np.isclose(rotated.v_rot, 0.0, atol=1e-10)  # v should be zero
        assert np.isclose(rotated.w_rot, 0.0, atol=1e-10)  # w should be zero
        
        # Check conservation of total velocity
        original_speed = np.sqrt(wind_components.u_mean**2 + 
                               wind_components.v_mean**2 + 
                               wind_components.w_mean**2)
        rotated_speed = np.sqrt(rotated.u_rot**2 + 
                              rotated.v_rot**2 + 
                              rotated.w_rot**2)
        assert np.isclose(original_speed, rotated_speed)

class TestPlanarFit:
    """Tests for planar fit implementation"""
    
    def test_planar_fit_initialization(self):
        """Test initialization of planar fit"""
        pf = PlanarFit()
        assert pf.alpha == 0.0
        assert pf.beta == 0.0
        assert pf.gamma == 0.0
        assert pf.b0 == 0.0
        assert pf.b1 == 0.0
        assert pf.b2 == 0.0
        
    def test_fit_plane(self, planar_fit_data):
        """Test plane fitting to wind data"""
        u, v, w = planar_fit_data
        pf = PlanarFit()
        pf.fit_plane(u, v, w)
        
        # Check that fitted coefficients are close to true values
        assert np.isclose(pf.b1, 0.1, atol=0.02)  # u coefficient
        assert np.isclose(pf.b2, 0.05, atol=0.02) # v coefficient
        
        # Check rotation angles calculation
        assert np.isclose(pf.alpha, np.arctan(pf.b1), atol=1e-10)
        assert np.isclose(pf.beta, 
                         np.arctan(pf.b2 * np.cos(pf.alpha)), 
                         atol=1e-10)
        
    def test_rotate_wind(self, wind_components, planar_fit_data):
        """Test wind rotation using planar fit"""
        u, v, w = planar_fit_data
        pf = PlanarFit()
        pf.fit_plane(u, v, w)
        rotated = pf.rotate_wind(wind_components)
        
        # Check mean vertical wind is reduced
        assert abs(rotated.w_rot) < abs(wind_components.w_mean)
        
        # Check conservation of total velocity
        original_speed = np.sqrt(wind_components.u_mean**2 + 
                               wind_components.v_mean**2 + 
                               wind_components.w_mean**2)
        rotated_speed = np.sqrt(rotated.u_rot**2 + 
                              rotated.v_rot**2 + 
                              rotated.w_rot**2)
        assert np.isclose(original_speed, rotated_speed)

def test_rotate_scalar_fluxes(wind_components):
    """Test rotation of scalar fluxes"""
    # Create sample scalar fluxes
    scalar_u = 0.2
    scalar_v = 0.1
    scalar_w = -0.15
    
    # Test with double rotation
    dr = DoubleRotation()
    dr.calculate_angles(wind_components)
    
    rotated_fluxes = rotate_scalar_fluxes(
        dr, scalar_u, scalar_v, scalar_w
    )
    
    # Check conservation of total scalar flux
    original_flux = np.sqrt(scalar_u**2 + scalar_v**2 + scalar_w**2)
    rotated_flux = np.sqrt(sum(x**2 for x in rotated_fluxes))
    assert np.isclose(original_flux, rotated_flux)
    
    # Test with planar fit
    pf = PlanarFit()
    u = np.random.normal(5.0, 1.0, 1000)
    v = np.random.normal(0.0, 1.0, 1000)
    w = 0.1*u + 0.05*v + np.random.normal(0.0, 0.1, 1000)
    pf.fit_plane(u, v, w)
    
    rotated_fluxes = rotate_scalar_fluxes(
        pf, scalar_u, scalar_v, scalar_w
    )
    
    # Check conservation with planar fit rotation
    rotated_flux = np.sqrt(sum(x**2 for x in rotated_fluxes))
    assert np.isclose(original_flux, rotated_flux)

class TestErrorHandling:
    """Tests for error handling"""
    
    def test_invalid_wind_components(self):
        """Test handling of invalid wind components"""
        with pytest.raises(ValueError):
            WindComponents(
                u_mean=np.nan,
                v_mean=1.0,
                w_mean=0.1,
                uu_cov=0.5,
                vv_cov=0.3,
                ww_cov=0.2,
                uv_cov=0.1,
                uw_cov=-0.15,
                vw_cov=-0.05
            )
            
    def test_planar_fit_insufficient_data(self):
        """Test handling of insufficient data for planar fit"""
        pf = PlanarFit()
        with pytest.raises(ValueError):
            pf.fit_plane(
                np.array([1.0]),  # Single point
                np.array([0.5]),
                np.array([0.1])
            )
            
    def test_invalid_rotation_angles(self):
        """Test handling of invalid rotation angles"""
        dr = DoubleRotation()
        dr.alpha = np.pi  # Invalid rotation angle
        with pytest.raises(ValueError):
            dr.rotate_wind(wind_components)
