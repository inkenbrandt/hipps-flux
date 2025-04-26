import pytest
import numpy as np
from easyfluxpy.footprint import (
    FootprintParams,
    KljunFootprint,
    KormannMeixnerFootprint,
    calculate_footprint_climatology
)

class TestFootprintParams:
    """Test suite for FootprintParams validation"""
    
    def test_valid_params(self):
        """Test creation with valid parameters"""
        params = FootprintParams(
            z_m=2.0,        # Measurement height
            z_0=0.15,       # Roughness length
            u_star=0.5,     # Friction velocity
            sigma_v=0.3,    # Standard deviation of lateral velocity
            h_abl=1000.0,   # Boundary layer height
            L=-50.0,        # Obukhov length
            wind_dir=180.0  # Wind direction
        )
        assert params.z_m == 2.0
        assert params.z_0 == 0.15
        assert params.u_star == 0.5
        assert params.sigma_v == 0.3
        assert params.h_abl == 1000.0
        assert params.L == -50.0
        assert params.wind_dir == 180.0
        
    def test_invalid_measurement_height(self):
        """Test validation of measurement height"""
        with pytest.raises(ValueError, match="Measurement height must be greater than roughness length"):
            FootprintParams(
                z_m=0.1,      # Less than roughness length
                z_0=0.15,
                u_star=0.5,
                sigma_v=0.3,
                h_abl=1000.0,
                L=-50.0,
                wind_dir=180.0
            )
            
    def test_invalid_friction_velocity(self):
        """Test validation of friction velocity"""
        with pytest.raises(ValueError, match="Friction velocity must be positive"):
            FootprintParams(
                z_m=2.0,
                z_0=0.15,
                u_star=0.0,   # Zero friction velocity
                sigma_v=0.3,
                h_abl=1000.0,
                L=-50.0,
                wind_dir=180.0
            )
            
    def test_invalid_abl_height(self):
        """Test validation of boundary layer height"""
        with pytest.raises(ValueError, match="ABL height must be greater than measurement height"):
            FootprintParams(
                z_m=2.0,
                z_0=0.15,
                u_star=0.5,
                sigma_v=0.3,
                h_abl=1.0,    # Less than measurement height
                L=-50.0,
                wind_dir=180.0
            )
            
    def test_invalid_wind_direction(self):
        """Test validation of wind direction range"""
        with pytest.raises(ValueError, match="Wind direction must be between 0 and 360 degrees"):
            FootprintParams(
                z_m=2.0,
                z_0=0.15,
                u_star=0.5,
                sigma_v=0.3,
                h_abl=1000.0,
                L=-50.0,
                wind_dir=370.0  # Outside valid range
            )

class TestKljunFootprint:
    """Test suite for KljunFootprint model"""
    
    @pytest.fixture
    def stable_params(self):
        """Fixture for stable conditions"""
        return FootprintParams(
            z_m=2.0,
            z_0=0.15,
            u_star=0.5,
            sigma_v=0.3,
            h_abl=1000.0,
            L=50.0,    # Stable
            wind_dir=180.0
        )
    
    @pytest.fixture
    def unstable_params(self):
        """Fixture for unstable conditions"""
        return FootprintParams(
            z_m=2.0,
            z_0=0.15,
            u_star=0.5,
            sigma_v=0.3,
            h_abl=1000.0,
            L=-50.0,   # Unstable
            wind_dir=180.0
        )
        
    def test_model_initialization(self, stable_params):
        """Test model initialization and stability validation"""
        model = KljunFootprint(stable_params)
        assert model.params == stable_params
        assert hasattr(model, 'scaled_length')
        assert model.scaled_length is None
        
    def test_peak_distance_calculation(self, stable_params, unstable_params):
        """Test calculation of peak contribution distance"""
        stable_model = KljunFootprint(stable_params)
        unstable_model = KljunFootprint(unstable_params)
        
        stable_result = stable_model.calculate_footprint()
        unstable_result = unstable_model.calculate_footprint()
        
        # Peak should be further in stable conditions
        assert stable_result['x_peak'] > unstable_result['x_peak']
        assert all(x > 0 for x in [stable_result['x_peak'], unstable_result['x_peak']])
        
    def test_crosswind_integrated_footprint(self, stable_params):
        """Test cross-wind integrated footprint calculation"""
        model = KljunFootprint(stable_params)
        x_points = np.linspace(0, 100, 50)
        
        result = model.calculate_footprint(x_points=x_points)
        
        assert 'f_ci' in result
        assert len(result['f_ci']) == len(x_points)
        assert all(f >= 0 for f in result['f_ci'])  # Non-negative values
        assert abs(np.trapz(result['f_ci'], x_points) - 1.0) < 0.1  # Approximately integrates to 1
        
    def test_2d_footprint_calculation(self, stable_params):
        """Test 2D footprint calculation"""
        model = KljunFootprint(stable_params)
        x_points = np.linspace(0, 100, 20)
        y_points = np.linspace(-50, 50, 20)
        
        result = model.calculate_footprint(x_points=x_points, y_points=y_points)
        
        assert 'f_2d' in result
        assert result['f_2d'].shape == (len(y_points), len(x_points))
        assert all(f >= 0 for f in result['f_2d'].flatten())  # Non-negative values
        # Total integral should be approximately 1
        assert abs(np.trapz(np.trapz(result['f_2d'], x_points), y_points) - 1.0) < 0.1

class TestKormannMeixnerFootprint:
    """Test suite for KormannMeixnerFootprint model"""
    
    @pytest.fixture
    def default_params(self):
        """Fixture for default parameters"""
        return FootprintParams(
            z_m=2.0,
            z_0=0.15,
            u_star=0.5,
            sigma_v=0.3,
            h_abl=1000.0,
            L=-50.0,
            wind_dir=180.0
        )
        
    def test_model_initialization(self, default_params):
        """Test model initialization"""
        model = KormannMeixnerFootprint(default_params)
        assert model.params == default_params
        assert hasattr(model, 'r')  # Shape factor
        assert hasattr(model, 'u')  # Power law wind profile exponent
        
    def test_parameter_calculation(self, default_params):
        """Test calculation of model parameters"""
        model = KormannMeixnerFootprint(default_params)
        model._calculate_parameters()
        
        assert model.r is not None
        assert model.u is not None
        assert model.r > 0  # Shape factor should be positive
        
    def test_footprint_calculation(self, default_params):
        """Test basic footprint calculation"""
        model = KormannMeixnerFootpoint(default_params)
        x_points = np.linspace(0, 100, 50)
        
        result = model.calculate_footprint(x_points=x_points)
        
        assert 'x_peak' in result
        assert 'f_ci' in result
        assert len(result['f_ci']) == len(x_points)
        assert all(f >= 0 for f in result['f_ci'])
        
    def test_stability_dependence(self):
        """Test footprint dependence on stability"""
        # Create models with different stability conditions
        stable_params = FootprintParams(
            z_m=2.0, z_0=0.15, u_star=0.5, sigma_v=0.3,
            h_abl=1000.0, L=50.0, wind_dir=180.0
        )
        unstable_params = FootprintParams(
            z_m=2.0, z_0=0.15, u_star=0.5, sigma_v=0.3,
            h_abl=1000.0, L=-50.0, wind_dir=180.0
        )
        
        stable_model = KormannMeixnerFootprint(stable_params)
        unstable_model = KormannMeixnerFootprint(unstable_params)
        
        x_points = np.linspace(0, 100, 50)
        stable_result = stable_model.calculate_footprint(x_points=x_points)
        unstable_result = unstable_model.calculate_footprint(x_points=x_points)
        
        # Check that peak location changes with stability
        assert stable_result['x_peak'] != unstable_result['x_peak']
        # Footprint should extend further in stable conditions
        assert np.sum(stable_result['f_ci'] > 1e-3) > np.sum(unstable_result['f_ci'] > 1e-3)

def test_footprint_climatology():
    """Test footprint climatology calculation"""
    # Create sample parameters list
    params_list = [
        FootprintParams(
            z_m=2.0, z_0=0.15, u_star=0.5, sigma_v=0.3,
            h_abl=1000.0, L=-50.0, wind_dir=180.0
        ),
        FootprintParams(
            z_m=2.0, z_0=0.15, u_star=0.4, sigma_v=0.25,
            h_abl=1000.0, L=30.0, wind_dir=200.0
        )
    ]
    
    # Define domain
    domain = {
        'x': np.linspace(0, 100, 20),
        'y': np.linspace(-50, 50, 20)
    }
    
    # Calculate climatology using both models
    for model_name in ['kljun', 'kormann-meixner']:
        result = calculate_footprint_climatology(model_name, params_list, domain)
        
        assert 'f_clim' in result
        assert 'contrib_90' in result
        assert result['f_clim'].shape == (len(domain['y']), len(domain['x']))
        assert result['contrib_90'].shape == result['f_clim'].shape
        assert result['contrib_90'].dtype == bool
        
        # Check that climatological footprint integrates approximately to 1
        integral = np.trapz(np.trapz(result['f_clim'], domain['x']), domain['y'])
        assert abs(integral - 1.0) < 0.1
        
        # Check that 90% contribution area is reasonable
        assert 0.89 < np.sum(result['f_clim'][result['contrib_90']]) < 0.91

def test_invalid_model_name():
    """Test error handling for invalid model name"""
    with pytest.raises(ValueError, match="Unknown footprint model"):
        calculate_footprint_climatology(
            'invalid_model',
            [FootprintParams(z_m=2.0, z_0=0.15, u_star=0.5, sigma_v=0.3,
                           h_abl=1000.0, L=-50.0, wind_dir=180.0)],
            {'x': np.array([0, 1]), 'y': np.array([0, 1])}
        )
