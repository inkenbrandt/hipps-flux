import pytest
import numpy as np
from easyfluxpy.boundary_layer import (
    calculate_air_temperature,
    planetary_boundary_layer_height,
    calculate_air_density,
    calculate_saturation_vapor_pressure,
    calculate_dewpoint_temperature,
    BoundaryLayerParams,
    BoundaryLayerProcessor
)

# Test Data Constants
STANDARD_TEMPERATURE = 293.15  # 20°C in K
STANDARD_PRESSURE = 101.325    # kPa
STANDARD_H2O_DENSITY = 10.0    # g/m³

class TestCalculateAirTemperature:
    """Tests for air temperature calculation function"""
    
    def test_valid_inputs(self):
        """Test with typical valid inputs"""
        sonic_temp = 293.15  # 20°C
        h2o_density = 10.0   # g/m³
        pressure = 101.325   # kPa
        
        result = calculate_air_temperature(sonic_temp, h2o_density, pressure)
        
        assert result is not None
        assert isinstance(result, float)
        assert 273.15 < result < 323.15  # Between 0°C and 50°C
        
    def test_invalid_inputs(self):
        """Test with invalid inputs"""
        test_cases = [
            (np.nan, 10.0, 101.325),      # NaN temperature
            (293.15, -1.0, 101.325),      # Negative water vapor
            (293.15, 10.0, -101.325),     # Negative pressure
            (0.0, 10.0, 101.325),         # Zero temperature
            (293.15, 10.0, 0.0),          # Zero pressure
        ]
        
        for sonic_temp, h2o_density, pressure in test_cases:
            result = calculate_air_temperature(sonic_temp, h2o_density, pressure)
            assert result is None

class TestPlanetaryBoundaryLayerHeight:
    """Tests for planetary boundary layer height calculation"""
    
    def test_unstable_conditions(self):
        """Test PBL height calculation for unstable conditions"""
        test_cases = [
            (-2000, 1000.0),    # Very unstable
            (-800, 1200.0),     # Moderately unstable
            (-300, 1500.0),     # Slightly unstable
            (-15, 2000.0),      # Near neutral unstable
        ]
        
        for obukhov, expected in test_cases:
            result = planetary_boundary_layer_height(obukhov)
            assert abs(result - expected) < 1.0
            
    def test_stable_conditions(self):
        """Test PBL height calculation for stable conditions"""
        test_cases = [
            (1500, 1000.0),    # Very stable
            (1100, 800.0),     # Moderately stable
            (500, 250.0),      # Slightly stable
            (50, 200.0),       # Near neutral stable
        ]
        
        for obukhov, expected in test_cases:
            result = planetary_boundary_layer_height(obukhov)
            assert abs(result - expected) < 1.0
            
    def test_invalid_inputs(self):
        """Test with invalid inputs"""
        assert planetary_boundary_layer_height(np.nan) is None
        assert planetary_boundary_layer_height("invalid") is None

class TestCalculateAirDensity:
    """Tests for air density calculations"""
    
    def test_standard_conditions(self):
        """Test density calculations at standard conditions"""
        temperature = 293.15  # 20°C
        pressure = 101.325   # kPa
        h2o_density = 10.0   # g/m³
        
        e_air, rho_d, rho_a = calculate_air_density(temperature, pressure, h2o_density)
        
        # Check vapor pressure is reasonable
        assert 0 < e_air < pressure
        
        # Check dry air density is reasonable (1.1-1.3 kg/m³)
        assert 1100 < rho_d < 1300
        
        # Check moist air density is reasonable (1.1-1.3 kg/m³)
        assert 1.1 < rho_a < 1.3
        
    def test_temperature_dependence(self):
        """Test density variation with temperature"""
        temps = [273.15, 293.15, 313.15]  # 0°C, 20°C, 40°C
        
        densities = []
        for temp in temps:
            _, _, rho_a = calculate_air_density(temp, STANDARD_PRESSURE, STANDARD_H2O_DENSITY)
            densities.append(rho_a)
            
        # Check density decreases with increasing temperature
        assert densities[0] > densities[1] > densities[2]

class TestCalculateSaturationVaporPressure:
    """Tests for saturation vapor pressure calculations"""
    
    def test_above_freezing(self):
        """Test vapor pressure calculation above freezing"""
        temperature = 20.0    # °C
        pressure = 101.325   # kPa
        
        e_sat, enhance_factor = calculate_saturation_vapor_pressure(temperature, pressure)
        
        # Check reasonable vapor pressure range
        assert 2.0 < e_sat < 3.0
        
        # Check enhancement factor is close to 1
        assert 0.99 < enhance_factor < 1.01
        
    def test_below_freezing(self):
        """Test vapor pressure calculation below freezing"""
        temperature = -10.0   # °C
        pressure = 101.325   # kPa
        
        e_sat, enhance_factor = calculate_saturation_vapor_pressure(temperature, pressure)
        
        # Check reasonable vapor pressure range
        assert 0.2 < e_sat < 0.3
        
        # Check enhancement factor is close to 1
        assert 0.99 < enhance_factor < 1.01
        
    def test_temperature_dependence(self):
        """Test vapor pressure increases with temperature"""
        temps = [-10.0, 0.0, 10.0, 20.0]
        pressures = []
        
        for temp in temps:
            e_sat, _ = calculate_saturation_vapor_pressure(temp, STANDARD_PRESSURE)
            pressures.append(e_sat)
            
        # Check pressure increases with temperature
        assert all(p1 < p2 for p1, p2 in zip(pressures, pressures[1:]))

class TestCalculateDewpointTemperature:
    """Tests for dewpoint temperature calculations"""
    
    def test_standard_conditions(self):
        """Test dewpoint calculation at standard conditions"""
        e_air = 1.5          # kPa
        pressure = 101.325   # kPa
        
        t_dp = calculate_dewpoint_temperature(e_air, pressure)
        
        # Check reasonable dewpoint range
        assert -20.0 < t_dp < 30.0
        
    def test_vapor_pressure_dependence(self):
        """Test dewpoint increases with vapor pressure"""
        vapor_pressures = [0.5, 1.0, 1.5, 2.0]
        dewpoints = []
        
        for e_air in vapor_pressures:
            t_dp = calculate_dewpoint_temperature(e_air, STANDARD_PRESSURE)
            dewpoints.append(t_dp)
            
        # Check dewpoint increases with vapor pressure
        assert all(t1 < t2 for t1, t2 in zip(dewpoints, dewpoints[1:]))

class TestBoundaryLayerProcessor:
    """Tests for BoundaryLayerProcessor class"""
    
    @pytest.fixture
    def standard_params(self):
        """Fixture providing standard boundary layer parameters"""
        return BoundaryLayerParams(
            height=1000.0,
            temperature=STANDARD_TEMPERATURE,
            pressure=STANDARD_PRESSURE,
            h2o_density=STANDARD_H2O_DENSITY
        )
    
    def test_initialization(self, standard_params):
        """Test processor initialization"""
        processor = BoundaryLayerProcessor(standard_params)
        assert processor.params == standard_params
        
    def test_invalid_initialization(self):
        """Test initialization with invalid parameters"""
        invalid_params = [
            BoundaryLayerParams(-1000.0, STANDARD_TEMPERATURE, STANDARD_PRESSURE, STANDARD_H2O_DENSITY),
            BoundaryLayerParams(1000.0, -STANDARD_TEMPERATURE, STANDARD_PRESSURE, STANDARD_H2O_DENSITY),
            BoundaryLayerParams(1000.0, STANDARD_TEMPERATURE, -STANDARD_PRESSURE, STANDARD_H2O_DENSITY),
            BoundaryLayerParams(1000.0, STANDARD_TEMPERATURE, STANDARD_PRESSURE, -STANDARD_H2O_DENSITY),
        ]
        
        for params in invalid_params:
            with pytest.raises(ValueError):
                BoundaryLayerProcessor(params)
                
    def test_process_measurements(self, standard_params):
        """Test measurement processing"""
        processor = BoundaryLayerProcessor(standard_params)
        results = processor.process_measurements()
        
        # Check all expected keys are present
        expected_keys = {
            'air_temperature',
            'vapor_pressure',
            'saturation_vapor_pressure',
            'relative_humidity',
            'dewpoint_temperature',
            'dry_air_density',
            'moist_air_density'
        }
        assert set(results.keys()) == expected_keys
        
        # Check values are in reasonable ranges
        assert 273.15 < results['air_temperature'] < 323.15
        assert 0 < results['vapor_pressure'] < results['saturation_vapor_pressure']
        assert 0 < results['relative_humidity'] < 100
        assert results['dewpoint_temperature'] < (results['air_temperature'] - 273.15)
        assert 1.0 < results['moist_air_density'] < 1.5
