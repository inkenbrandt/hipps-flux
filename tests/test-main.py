import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from easyfluxpy.main import EddyCovarianceProcessor
from easyfluxpy.freq_factor import SensorSpecs
from easyfluxpy import boundary_layer


# Patch at module level for all tests
@pytest.fixture(autouse=True)
def mock_calculate_air_temperature():
    with patch('easyfluxpy.main.calculate_air_temperature') as mock:
        mock.return_value = 293.15
        yield mock


@pytest.fixture
def mock_config():
    """Basic configuration for testing"""
    return {
        'irgason': True,
        'output_interval': 30,
        'latitude': 41.766,
        'longitude': -111.855,
        'altitude': 1356.0,
        'measurement_height': 2.0,
        'surface_type': 'GRASS',
        'canopy_height': 0.5
    }


@pytest.fixture
def mock_measurements():
    """Generate mock measurement data with scalar met values"""
    n_samples = 1800
    np.random.seed(42)

    sonic_data = {
        'u': np.random.normal(2.0, 0.5, n_samples),
        'v': np.random.normal(0.0, 0.5, n_samples),
        'w': np.random.normal(0.0, 0.3, n_samples),
        'ts': np.random.normal(20.0, 1.0, n_samples)
    }

    irga_data = {
        'co2': np.random.normal(400.0, 10.0, n_samples),
        'h2o': np.random.normal(10.0, 1.0, n_samples),
        'press': np.full(n_samples, 101.3)
    }

    # Important: Use scalar values for met data
    met_data = {
        'temperature': float(np.mean(sonic_data['ts'])),
        'pressure': float(np.mean(irga_data['press'])),
        'rh': 50.0
    }

    return {
        'sonic': sonic_data,
        'irga': irga_data,
        'met': met_data
    }


class TestEddyCovarianceProcessor:
    """Test suite for EddyCovarianceProcessor class"""

    def test_initialization(self, mock_config):
        """Test processor initialization"""
        processor = EddyCovarianceProcessor(mock_config)

        assert processor.irgason == True
        assert processor.output_interval == 30
        assert processor.use_planar_fit == False
        assert processor.scan_count == 0

        assert processor.site['latitude'] == pytest.approx(41.766)
        assert processor.site['longitude'] == pytest.approx(-111.855)
        assert processor.site['measurement_height'] == pytest.approx(2.0)
        assert processor.site['surface_type'] == 'GRASS'

    def test_process_measurements(self, mock_config, mock_measurements):
        """Test measurement processing"""
        processor = EddyCovarianceProcessor(mock_config)

        results = processor.process_measurements(
            mock_measurements['sonic'],
            mock_measurements['irga'],
            mock_measurements['met']
        )

        assert 'fluxes' in results
        assert 'turbulence' in results
        assert 'meteorology' in results
        assert 'rotated_components' in results
        assert 'quality' in results

        fluxes = results['fluxes']
        for key in ['tau', 'H', 'LE', 'Fc']:
            assert key in fluxes
            assert np.isfinite(fluxes[key])

        turb = results['turbulence']
        assert turb['u_star'] > 0
        assert turb['TKE'] > 0

    def test_coordinate_rotation(self, mock_config, mock_measurements):
        """Test coordinate rotation results"""
        processor = EddyCovarianceProcessor(mock_config)
        results = processor.process_measurements(
            mock_measurements['sonic'],
            mock_measurements['irga'],
            mock_measurements['met']
        )

        rotated = results['rotated_components']
        assert abs(rotated.v_rot) < 1e-10
        assert abs(rotated.w_rot) < 1e-10

    def test_quality_control(self, mock_config, mock_measurements):
        """Test quality control results"""
        processor = EddyCovarianceProcessor(mock_config)
        results = processor.process_measurements(
            mock_measurements['sonic'],
            mock_measurements['irga'],
            mock_measurements['met']
        )

        qc = results['quality']
        assert 'overall_flag' in qc
        assert isinstance(qc['overall_flag'], int)
        assert 1 <= qc['overall_flag'] <= 9


@patch('easyfluxpy.main.EddyCovarianceProcessor')
def test_main(mock_processor):
    """Test main function"""
    from easyfluxpy.main import main

    processor_instance = Mock()
    mock_processor.return_value = processor_instance

    main()

    mock_processor.assert_called_once()
    processor_instance.run_processing_loop.assert_called_once()


class TestEdgeConditions:
    """Test edge cases and error conditions"""

    def test_missing_data(self, mock_config, mock_measurements):
        """Test handling of missing data"""
        processor = EddyCovarianceProcessor(mock_config)

        measurements = mock_measurements.copy()
        measurements['sonic']['u'][:100] = np.nan

        results = processor.process_measurements(
            measurements['sonic'],
            measurements['irga'],
            measurements['met']
        )

        assert all(key in results for key in ['fluxes', 'turbulence', 'meteorology'])

    def test_invalid_config(self, mock_config):
        """Test handling of invalid configuration"""
        invalid_config = mock_config.copy()
        invalid_config['measurement_height'] = -1.0

        with pytest.raises(ValueError):
            processor = EddyCovarianceProcessor(invalid_config)

    def test_extreme_values(self, mock_config, mock_measurements):
        """Test handling of extreme but possible values"""
        processor = EddyCovarianceProcessor(mock_config)

        measurements = mock_measurements.copy()
        measurements['sonic']['u'] = 10 * measurements['sonic']['u']
        measurements['irga']['co2'] = 2 * measurements['irga']['co2']

        results = processor.process_measurements(
            measurements['sonic'],
            measurements['irga'],
            measurements['met']
        )

        assert results['quality']['overall_flag'] > 5


if __name__ == '__main__':
    pytest.main(['-v'])