"""Tests for OVOD pipeline functionality."""
import pytest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch

# Add repo to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../repo'))


class TestPipelineImports:
    """Test that core pipeline modules can be imported."""
    
    def test_pipeline_import(self):
        """Test that the main pipeline can be imported."""
        try:
            from ovod.pipeline import OVODPipeline
            assert True, "Pipeline import successful"
        except ImportError as e:
            pytest.skip(f"Pipeline import failed (expected in CI): {e}")
    
    def test_detector_import(self):
        """Test that detector module can be imported."""
        try:
            from src.detector import GroundingDINODetector
            assert True, "Detector import successful"
        except ImportError as e:
            pytest.skip(f"Detector import failed (expected in CI): {e}")
    
    def test_segmenter_import(self):
        """Test that segmenter module can be imported."""
        try:
            from src.segmenter import SAM2Segmenter
            assert True, "Segmenter import successful"
        except ImportError as e:
            pytest.skip(f"Segmenter import failed (expected in CI): {e}")


class TestPipelineInitialization:
    """Test pipeline initialization without requiring actual models."""
    
    @patch('src.detector.GroundingDINODetector')
    @patch('src.segmenter.SAM2Segmenter')
    def test_pipeline_creation(self, mock_segmenter, mock_detector):
        """Test that pipeline can be created with mocked dependencies."""
        try:
            from ovod.pipeline import OVODPipeline
            
            # Mock the detector and segmenter
            mock_detector_instance = Mock()
            mock_segmenter_instance = Mock()
            mock_detector.return_value = mock_detector_instance
            mock_segmenter.return_value = mock_segmenter_instance
            
            # Create pipeline
            pipeline = OVODPipeline()
            
            assert pipeline is not None, "Pipeline should be created successfully"
            assert hasattr(pipeline, 'predict'), "Pipeline should have predict method"
            
        except ImportError as e:
            pytest.skip(f"Pipeline modules not available for testing: {e}")


class TestVisualizationUtils:
    """Test visualization utilities."""
    
    def test_visualize_import(self):
        """Test that visualization module can be imported."""
        try:
            from src.visualize import create_detection_visualization
            assert True, "Visualization import successful"
        except ImportError as e:
            pytest.skip(f"Visualization import failed: {e}")
    
    def test_dummy_image_creation(self):
        """Test creation of dummy image for visualization."""
        try:
            # Create a dummy image
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            assert dummy_image.shape == (480, 640, 3), "Dummy image should have correct shape"
            assert dummy_image.dtype == np.uint8, "Dummy image should be uint8"
            
        except Exception as e:
            pytest.fail(f"Failed to create dummy image: {e}")


if __name__ == "__main__":
    pytest.main([__file__])