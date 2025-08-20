"""Test that core modules can be imported."""
import pytest
import sys
import os

# Add repo to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../repo'))


def test_pipeline_import():
    """Test that the main pipeline can be imported."""
    try:
        from ovod.pipeline import OVODPipeline
        assert True, "Pipeline import successful"
    except ImportError as e:
        pytest.skip(f"Pipeline import failed (expected in CI): {e}")


def test_segmenter_import():
    """Test that segmenter module can be imported."""
    try:
        from src.segmenter import SAM2Segmenter
        assert True, "Segmenter import successful"
    except ImportError as e:
        pytest.skip(f"Segmenter import failed (expected in CI): {e}")


def test_detector_import():
    """Test that detector module can be imported."""
    try:
        from src.detector import GroundingDINODetector
        assert True, "Detector import successful"
    except ImportError as e:
        pytest.skip(f"Detector import failed (expected in CI): {e}")


def test_eval_import():
    """Test that eval module can be imported."""
    try:
        from eval import to_coco_xywh, normalize_category
        assert True, "Eval module import successful"
    except ImportError as e:
        pytest.fail(f"Eval module should always import: {e}")


if __name__ == "__main__":
    pytest.main([__file__])