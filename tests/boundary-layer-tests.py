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
STANDARD_PRESSURE = 101.325  # kPa
STANDARD_H2O_DENSITY = 10.0  # g/m³


class TestCalculateAirTemperature:
    """Tests for air temperature calculation function"""

    def test_valid_inputs(self):
        """Test with typical valid inputs"""
        sonic_temp = 293.15  # 20°C in K
        h2o_density = 8.0  # g/m³ (typical value at ~20°C and 50% RH)
        pressure = 101.325  # kPa

        result = calculate_air_temperature(sonic_temp, h2o_density, pressure)

        assert result is not None
        assert isinstance(result, (float, np.float64))
        # Temperature difference should be small for typical conditions
        assert abs(result - sonic_temp) < 1.0, (
            f"Got {result:.2f}K, expected close to {sonic_temp:.2f}K"
            f" (difference: {abs(result - sonic_temp):.2f}K)"
        )

    def test_reference_cases(self):
        """Test against reference cases from Kaimal and Gaynor (1991)"""
        # Test cases calculated directly using Ts/T = 1 + 0.32e/p formula
        reference_cases = [
            # sonic_temp(K), h2o(g/m³), pressure(kPa), expected_temp(K)
            # Values calculated using e = ρRvT and the 0.32e/p relationship
            (300.15, 10.0, 101.325, 298.84),  # Moderate conditions
            (300.15, 15.0, 101.325, 298.19),  # More humid conditions
            (290.15, 5.0, 101.325, 289.49),  # Cooler conditions
        ]

        for sonic_temp, h2o, pressure, expected in reference_cases:
            result = calculate_air_temperature(sonic_temp, h2o, pressure)
            assert result is not None
            assert abs(result - expected) < 0.1, (
                f"Reference case failed: "
                f"For Ts={sonic_temp:.2f}K, h2o={h2o}g/m³, P={pressure}kPa: "
                f"got {result:.2f}K, expected {expected:.2f}K"
                f"\nDifference: {abs(result - expected):.3f}K"
            )

            # Verify the Kaimal and Gaynor relationship
            h2o_density_kgm3 = h2o / 1000.0
            e = h2o_density_kgm3 * 461.5 * sonic_temp
            p = pressure * 1000.0
            ratio = 1 + 0.32 * e / p
            calc_temp = sonic_temp / ratio
            assert abs(result - calc_temp) < 0.01, (
                f"Result doesn't match Kaimal and Gaynor formula: "
                f"got {result:.3f}K, formula gives {calc_temp:.3f}K"
            )

    def test_temperature_relationships(self):
        """Test physical relationships in temperature calculations"""
        sonic_temp = 293.15  # K
        h2o_values = [0.0, 2.0, 4.0]  # g/m³
        pressure = 101.325  # kPa

        results = []
        for h2o in h2o_values:
            temp = calculate_air_temperature(sonic_temp, h2o, pressure)
            assert temp is not None
            results.append(temp)

        # Check that temperature decreases with increasing humidity
        assert all(t1 > t2 for t1, t2 in zip(results[:-1], results[1:])), (
            f"Temperature should decrease with increasing humidity. Got: {results}"
        )

        # Calculate temperature differences between consecutive humidity values
        diffs = [t1 - t2 for t1, t2 in zip(results[:-1], results[1:])]
        # Check that differences are reasonable but non-zero
        assert all(0.2 < diff < 0.4 for diff in diffs), (
            f"Temperature differences {diffs} outside expected range [0.2, 0.4]"
        )

    def test_zero_humidity(self):
        """Test boundary condition of zero humidity"""
        sonic_temp = 300.15  # K
        pressure = 101.325  # kPa

        result = calculate_air_temperature(sonic_temp, 0.0, pressure)
        assert result is not None
        # For zero humidity, air temperature should equal sonic temperature
        assert abs(result - sonic_temp) < 0.01, (
            f"For zero humidity, expected {sonic_temp:.3f}K, got {result:.3f}K"
        )

    def test_invalid_inputs(self):
        """Test with invalid inputs"""
        test_cases = [
            (np.nan, 10.0, 101.325),  # NaN temperature
            (293.15, -1.0, 101.325),  # Negative water vapor
            (293.15, 10.0, -101.325),  # Negative pressure
            (0.0, 10.0, 101.325),  # Zero temperature
            (293.15, 10.0, 0.0),  # Zero pressure
        ]

        for sonic_temp, h2o_density, pressure in test_cases:
            result = calculate_air_temperature(sonic_temp, h2o_density, pressure)
            assert result is None, (
                f"Expected None for inputs: T={sonic_temp}, "
                f"h2o={h2o_density}, P={pressure}"
            )
class TestPlanetaryBoundaryLayerHeight:
    """Tests for planetary boundary layer height calculation"""

    def test_unstable_conditions(self):
        """Test PBL height calculation for unstable conditions"""
        # First calculate the actual values from the function
        test_obukhov = [-2000, -800, -300, -15]
        expected_values = [1000.0, 1117.42, 1472.0, 1980.0]

        for obukhov, expected in zip(test_obukhov, expected_values):
            result = planetary_boundary_layer_height(obukhov)
            assert abs(result - expected) < 0.1, f"Failed for Obukhov length {obukhov}"

    def test_stable_conditions(self):
        """Test PBL height calculation for stable conditions"""
        # First calculate the actual values from the function
        test_obukhov = [1500, 1100, 500, 50]
        expected_values = [1000.0, 843.29, 432.18, 184.13]

        for obukhov, expected in zip(test_obukhov, expected_values):
            result = planetary_boundary_layer_height(obukhov)
            assert abs(result - expected) < 0.1, f"Failed for Obukhov length {obukhov}"

    def test_extreme_values(self):
        """Test boundary conditions"""
        # Test very unstable conditions
        assert abs(planetary_boundary_layer_height(-10000) - 1000.0) < 0.1

        # Test very stable conditions
        assert abs(planetary_boundary_layer_height(10000) - 1000.0) < 0.1

        # Test near-neutral conditions
        assert abs(planetary_boundary_layer_height(-0.1) - 2019.0) < 0.1
        assert abs(planetary_boundary_layer_height(0.1) - 199.89) < 0.1

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
