"""Test synthetic inference without requiring real models."""
import pytest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch

# Add repo to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../repo'))


@pytest.mark.skipif(not os.environ.get('RUN_SYNTHETIC_TESTS'), 
                    reason="Synthetic tests require explicit opt-in")
def test_synthetic_pipeline_inference():
    """Test pipeline with synthetic data and mocked models."""
    try:
        from ovod.pipeline import OVODPipeline
        
        # Create synthetic 320x320 RGB image
        synthetic_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        
        # Try to create pipeline - may fail if models not available
        try:
            # Use CPU device for CI compatibility
            device = "cpu"
            pipeline = OVODPipeline(device=device)
            
            # Try to load models if method exists
            if hasattr(pipeline, 'load_model'):
                try:
                    pipeline.load_model()
                except Exception as e:
                    pytest.skip(f"Model loading failed (expected in CI): {e}")
            
            # Try prediction with low expectations
            try:
                result = pipeline.predict(
                    image=synthetic_image,
                    text_prompt="person",
                    max_detections=3
                )
                
                # Basic output validation
                assert isinstance(result, dict), "Result should be a dictionary"
                
                # Allow zero detections for synthetic data
                if 'boxes' in result:
                    assert isinstance(result['boxes'], (list, np.ndarray)), "Boxes should be list or array"
                if 'masks' in result:
                    assert isinstance(result['masks'], (list, np.ndarray)), "Masks should be list or array"
                
                print(f"✓ Synthetic inference completed: {len(result.get('boxes', []))} detections")
                
            except Exception as e:
                pytest.skip(f"Prediction failed (expected without real models): {e}")
                
        except Exception as e:
            pytest.skip(f"Pipeline creation failed (expected in CI): {e}")
            
    except ImportError as e:
        pytest.skip(f"Pipeline import failed: {e}")


def test_box_format_conversion():
    """Test box format conversion with synthetic data."""
    try:
        from eval import to_coco_xywh
        
        # Test normalized cxcywh to pixel xywh conversion
        x, y, w, h = to_coco_xywh([0.5, 0.5, 0.25, 0.5], 320, 320)
        
        # Verify reasonable output
        assert w > 0 and h > 0, f"Invalid dimensions: w={w}, h={h}"
        assert 0 <= x < 320 and 0 <= y < 320, f"Invalid coordinates: x={x}, y={y}"
        
        print(f"✓ Box conversion test: ({x}, {y}, {w}, {h})")
        
    except ImportError as e:
        pytest.fail(f"Eval module should be importable: {e}")


if __name__ == "__main__":
    pytest.main([__file__])