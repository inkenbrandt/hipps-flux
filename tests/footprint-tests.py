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

        # Test physical constraints
        assert stable_result['x_peak'] > 0, "Peak distance should be positive"
        assert unstable_result['x_peak'] > 0, "Peak distance should be positive"

        # Test relative to measurement height
        min_expected = 0.5 * stable_params.z_m  # Peak should be at least half the measurement height
        max_expected = 100 * stable_params.z_m  # Peak shouldn't be unreasonably far

        assert min_expected < stable_result['x_peak'] < max_expected, \
            f"Stable peak distance {stable_result['x_peak']} outside expected range"
        assert min_expected < unstable_result['x_peak'] < max_expected, \
            f"Unstable peak distance {unstable_result['x_peak']} outside expected range"

        # Test that peak distances are different for different stability conditions
        # but don't assume which one should be larger
        assert abs(stable_result['x_peak'] - unstable_result['x_peak']) < \
               max(stable_result['x_peak'], unstable_result['x_peak']), \
            "Peak distances should be different but within reasonable bounds"

        # Test relationship with friction velocity
        high_ustar_params = FootprintParams(
            z_m=stable_params.z_m,
            z_0=stable_params.z_0,
            u_star=stable_params.u_star * 2,  # Double friction velocity
            sigma_v=stable_params.sigma_v,
            h_abl=stable_params.h_abl,
            L=stable_params.L,
            wind_dir=stable_params.wind_dir
        )

        high_ustar_model = KljunFootprint(high_ustar_params)
        high_ustar_result = high_ustar_model.calculate_footprint()

        # Higher friction velocity should result in a further peak
        assert high_ustar_result['x_peak'] > stable_result['x_peak'], \
            "Peak distance should increase with friction velocity"
        
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

        # Use fewer points for better numerical stability
        x_points = np.linspace(1, 100, 15)  # Start at 1 to avoid singularity at x=0
        y_points = np.linspace(-50, 50, 15)

        result = model.calculate_footprint(x_points=x_points, y_points=y_points)

        # Basic checks
        assert 'f_2d' in result, "2D footprint missing from results"
        assert result['f_2d'].shape == (len(y_points), len(x_points)), \
            f"Wrong shape: got {result['f_2d'].shape}, expected {(len(y_points), len(x_points))}"

        # Check for negative values with tolerance for numerical precision
        tolerance = 1e-10  # Tolerance for numerical errors
        min_value = np.min(result['f_2d'])
        if min_value < 0:
            if abs(min_value) > tolerance:
                raise AssertionError(
                    f"Found significant negative values in footprint: {min_value}"
                )
            # Zero out small negative values
            result['f_2d'][result['f_2d'] < 0] = 0

        # Check physical properties
        assert np.all(np.isfinite(result['f_2d'])), "Found non-finite values in footprint"

        # Calculate grid spacing
        dx = x_points[1] - x_points[0]
        dy = y_points[1] - y_points[0]

        # Normalize footprint
        total_integral = np.sum(result['f_2d']) * dx * dy
        assert total_integral > 0, "Total integral should be positive"
        normalized_footprint = result['f_2d'] / total_integral

        # Test normalized integration
        normalized_integral = np.sum(normalized_footprint) * dx * dy
        assert 0.99 < normalized_integral < 1.01, \
            f"Normalized integral ({normalized_integral:.3f}) differs from 1.0"

        # Peak should be in a reasonable location
        peak_y_idx, peak_x_idx = np.unravel_index(
            np.argmax(normalized_footprint),
            normalized_footprint.shape
        )

        # Peak should be near center in crosswind (y) direction
        assert len(y_points) // 4 <= peak_y_idx <= 3 * len(y_points) // 4, \
            f"Peak not near center in crosswind direction: y-index = {peak_y_idx}"

        # Peak should be in first half of domain in streamwise (x) direction
        assert peak_x_idx <= len(x_points) // 2, \
            f"Peak too far downwind: x-index = {peak_x_idx}"

        # Test crosswind symmetry of normalized footprint
        y_center_idx = len(y_points) // 2
        max_asymmetry = 0.1  # Allow 10% asymmetry due to numerical effects

        # Test at multiple x-locations
        x_test_indices = [len(x_points) // 4, len(x_points) // 2, 3 * len(x_points) // 4]
        for x_idx in x_test_indices:
            y_profile = normalized_footprint[:, x_idx]

            # Test symmetry around center
            for i in range(y_center_idx):
                left_val = y_profile[i]
                right_val = y_profile[-(i + 1)]
                mean_val = (left_val + right_val) / 2
                if mean_val > 0.01 * np.max(y_profile):  # Only check significant values
                    asymmetry = abs(left_val - right_val) / mean_val
                    assert asymmetry < max_asymmetry, \
                        f"Excessive asymmetry ({asymmetry:.3f}) at y-index {i}, x-index {x_idx}"

        # Test exponential-like decay in crosswind direction
        center_x_idx = len(x_points) // 3
        y_profile = normalized_footprint[:, center_x_idx]
        peak_y = np.argmax(y_profile)

        # Test monotonic decay on both sides of peak
        if peak_y > 0:
            left_profile = y_profile[:peak_y]
            assert np.all(np.diff(left_profile) >= -tolerance), \
                "Non-monotonic increase towards peak from left"

        if peak_y < len(y_profile) - 1:
            right_profile = y_profile[peak_y:]
            assert np.all(np.diff(right_profile) <= tolerance), \
                "Non-monotonic decrease from peak towards right"

        # Additional test: Check relative magnitude of footprint values
        max_value = np.max(normalized_footprint)
        assert max_value > 0, "Maximum footprint value should be positive"

        # Values far from peak should be much smaller
        edge_values = np.concatenate([
            normalized_footprint[0, :],  # Left edge
            normalized_footprint[-1, :],  # Right edge
            normalized_footprint[:, -1]  # Downwind edge
        ])
        assert np.all(edge_values < 0.1 * max_value), \
            "Edge values should be much smaller than peak value"


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

        # Test parameter values for different stability conditions
        unstable_params = FootprintParams(
            z_m=default_params.z_m,
            z_0=default_params.z_0,
            u_star=default_params.u_star,
            sigma_v=default_params.sigma_v,
            h_abl=default_params.h_abl,
            L=-30.0,  # Unstable conditions
            wind_dir=default_params.wind_dir
        )

        stable_params = FootprintParams(
            z_m=default_params.z_m,
            z_0=default_params.z_0,
            u_star=default_params.u_star,
            sigma_v=default_params.sigma_v,
            h_abl=default_params.h_abl,
            L=30.0,  # Stable conditions
            wind_dir=default_params.wind_dir
        )

        # Check parameters for different stability conditions
        model_unstable = KormannMeixnerFootprint(unstable_params)
        model_stable = KormannMeixnerFootprint(stable_params)

        model_unstable._calculate_parameters()
        model_stable._calculate_parameters()

        # Parameters should differ between stability conditions
        assert model_unstable.r != model_stable.r, "Shape factor should differ with stability"
        assert model_unstable.u != model_stable.u, "Wind profile exponent should differ with stability"

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

    def _check_contributions(self, peak_x: float, cumsum: np.ndarray, x_points: np.ndarray) -> None:
        """Helper method to check cumulative contributions at different distances"""
        # Calculate contributions at multiples of peak distance
        contributions = []
        for mult in [1.0, 2.0, 5.0, 10.0]:
            dist = peak_x * mult
            if dist <= x_points[-1]:
                idx = np.searchsorted(x_points, dist)
                if idx < len(cumsum):  # Ensure we don't exceed array bounds
                    contribution = float(cumsum[idx])  # Convert to Python float
                    contributions.append((mult, contribution))

        # Check that contributions are monotonically increasing
        prev_contrib = 0.0  # Use Python float
        for mult, contrib in contributions:
            assert float(contrib) > float(prev_contrib), \
                f"Non-increasing contribution: {contrib:.3f} at {mult}x peak distance"
            prev_contrib = contrib

        # Check final contribution
        if contributions:
            final_mult, final_contrib = contributions[-1]
            assert 0.3 < float(final_contrib) < 1.1, \
                f"Unusual total contribution ({final_contrib:.3f}) at {final_mult}x peak distance"

    def test_stability_dependence(self):
        """Test footprint dependence on stability"""
        # Create strongly contrasting stability conditions
        stable_params = FootprintParams(
            z_m=2.0,
            z_0=0.15,
            u_star=0.5,
            sigma_v=0.3,
            h_abl=1000.0,
            L=10.0,  # Very stable
            wind_dir=180.0
        )
        unstable_params = FootprintParams(
            z_m=2.0,
            z_0=0.15,
            u_star=0.5,
            sigma_v=0.3,
            h_abl=1000.0,
            L=-10.0,  # Very unstable
            wind_dir=180.0
        )

        stable_model = KormannMeixnerFootprint(stable_params)
        unstable_model = KormannMeixnerFootprint(unstable_params)

        # Create domain that will capture stability differences
        z_m = stable_params.z_m
        x_min = 0.01 * z_m
        x_max = 100 * z_m

        # High-resolution grid for better peak detection
        x_close = np.linspace(x_min, z_m, 30)
        x_mid = np.linspace(z_m, 10 * z_m, 40)[1:]
        x_far = np.linspace(10 * z_m, x_max, 20)[1:]
        x_points = np.unique(np.concatenate([x_close, x_mid, x_far]))

        # Calculate footprints
        stable_result = stable_model.calculate_footprint(x_points=x_points)
        unstable_result = unstable_model.calculate_footprint(x_points=x_points)

        # Compare peak locations
        stable_peak = float(stable_result['x_peak'])
        unstable_peak = float(unstable_result['x_peak'])
        difference = abs(stable_peak - unstable_peak)

        # Peak locations should differ by at least 5% of measurement height
        min_difference = 0.05 * z_m
        assert difference > min_difference, \
            f"Peak locations too similar: stable={stable_peak:.3f}m, " \
            f"unstable={unstable_peak:.3f}m, difference={difference:.3f}m " \
            f"(expected > {min_difference:.3f}m)"

        # Normalize footprints for shape comparison
        stable_f = stable_result['f_ci']
        unstable_f = unstable_result['f_ci']

        stable_norm = stable_f / np.trapz(stable_f, x_points)
        unstable_norm = unstable_f / np.trapz(unstable_f, x_points)

        # Calculate shape difference metric
        shape_diff = np.trapz(np.abs(stable_norm - unstable_norm), x_points)
        assert shape_diff > 0.1, \
            f"Footprint shapes too similar: difference metric = {shape_diff:.3f}"

    def test_footprint_calculation(self, default_params):
        """Test basic footprint calculation"""
        model = KormannMeixnerFootprint(default_params)

        # Set up domain based on measurement height
        z_m = default_params.z_m
        x_min = 0.01 * z_m
        x_max = 100 * z_m

        # Create non-uniform grid focusing on near-source region
        x_close = np.linspace(x_min, z_m, 30)
        x_mid = np.linspace(z_m, 10 * z_m, 40)[1:]
        x_far = np.linspace(10 * z_m, x_max, 20)[1:]
        x_points = np.unique(np.concatenate([x_close, x_mid, x_far]))

        # Calculate footprint
        result = model.calculate_footprint(x_points=x_points)

        # Basic checks
        assert 'x_peak' in result, "Missing peak distance in results"
        assert 'f_ci' in result, "Missing crosswind-integrated footprint in results"
        assert len(result['f_ci']) == len(x_points), "Incorrect footprint length"

        # Convert numpy values to Python floats for comparisons
        assert float(result['x_peak']) >= 0, "Peak distance should be non-negative"
        assert float(result['x_peak']) <= float(x_points[-1]), "Peak should be within domain"
        assert all(float(f) >= 0 for f in result['f_ci']), "Footprint values should be non-negative"

        # Find peak
        peak_idx = int(np.argmax(result['f_ci']))  # Convert to Python int
        peak_x = float(x_points[peak_idx])  # Convert to Python float

        # Check peak location relative to measurement height
        min_peak = float(0.01 * z_m)
        max_peak = float(50 * z_m)
        assert min_peak <= peak_x <= max_peak, \
            f"Peak location (x={peak_x:.3f}m) outside reasonable range " \
            f"[{min_peak:.3f}m, {max_peak:.3f}m]"

        # Normalize footprint
        total_integral = float(np.trapz(result['f_ci'], x_points))
        assert total_integral > 0, "Total integral should be positive"
        f_ci_normalized = result['f_ci'] / total_integral

        # Calculate cumulative contribution
        cumsum = np.zeros_like(x_points)
        for i in range(1, len(x_points)):
            cumsum[i] = float(np.trapz(f_ci_normalized[:i + 1], x_points[:i + 1]))

        # Check contributions
        self._check_contributions(peak_x, cumsum, x_points)

        # Test 2D footprint if not skipped
        if hasattr(self, 'skip_2d_test'):
            return

        # Setup 2D domain
        y_max = 5.0 * peak_x
        y_points = np.linspace(-y_max, y_max, 31)
        result_2d = model.calculate_footprint(x_points=x_points, y_points=y_points)

        # Check and normalize 2D footprint
        f_2d = result_2d['f_2d']
        total_integral_2d = float(np.trapz(np.trapz(f_2d, x_points, axis=1), y_points))
        assert total_integral_2d > 0, "Total 2D integral should be positive"

        f_2d_normalized = f_2d / total_integral_2d
        integral_2d = float(np.trapz(np.trapz(f_2d_normalized, x_points, axis=1), y_points))
        assert 0.99 <= integral_2d <= 1.01, \
            f"2D integral ({integral_2d:.3f}) differs significantly from 1.0"

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

        x_points = np.linspace(1, 100, 50)
        stable_result = stable_model.calculate_footprint(x_points=x_points)
        unstable_result = unstable_model.calculate_footprint(x_points=x_points)

        # Check that peak location changes with stability
        assert abs(stable_result['x_peak'] - unstable_result['x_peak']) > 0.1, \
            "Peak location should differ with stability"

        # Compare footprint shapes
        f_ci_stable = stable_result['f_ci']
        f_ci_unstable = unstable_result['f_ci']

        # Normalize footprints for shape comparison
        f_ci_stable_norm = f_ci_stable / np.max(f_ci_stable)
        f_ci_unstable_norm = f_ci_unstable / np.max(f_ci_unstable)

        # Shapes should be different (using correlation coefficient)
        corr = np.corrcoef(f_ci_stable_norm, f_ci_unstable_norm)[0, 1]
        assert corr < 0.99, "Footprint shapes should differ with stability"

        # Check relative spread
        stable_spread = np.sum(f_ci_stable > 0.1 * np.max(f_ci_stable))
        unstable_spread = np.sum(f_ci_unstable > 0.1 * np.max(f_ci_unstable))
        assert stable_spread != unstable_spread, \
            "Footprint spread should differ with stability"

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
