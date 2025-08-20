"""Tests for evaluation functionality."""
import pytest
import sys
import os

# Add repo to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../repo'))

from eval import to_coco_xywh, normalize_category


class TestCocoBoxConversion:
    """Test COCO box format conversion."""
    
    def test_normalized_cxcywh_to_pixel_xywh(self):
        """Test conversion from normalized center format to pixel xywh."""
        # Normalized center format: cx=0.5, cy=0.5, w=0.25, h=0.5
        x, y, w, h = to_coco_xywh([0.5, 0.5, 0.25, 0.5], 640, 480)
        
        # Expected: x = (0.5 * 640) - (0.25 * 640)/2 = 320 - 80 = 240
        # Expected: y = (0.5 * 480) - (0.5 * 480)/2 = 240 - 120 = 120
        # Expected: w = 0.25 * 640 = 160
        # Expected: h = 0.5 * 480 = 240
        
        assert abs(x - 240) < 1e-6, f"Expected x=240, got {x}"
        assert abs(y - 120) < 1e-6, f"Expected y=120, got {y}"
        assert abs(w - 160) < 1e-6, f"Expected w=160, got {w}"
        assert abs(h - 240) < 1e-6, f"Expected h=240, got {h}"
    
    def test_pixel_xyxy_to_pixel_xywh(self):
        """Test conversion from pixel xyxy to pixel xywh."""
        # Pixel xyxy format: x1=100, y1=50, x2=300, y2=250
        x, y, w, h = to_coco_xywh([100, 50, 300, 250], 640, 480)
        
        # Expected: x=100, y=50, w=200, h=200 (already in correct format)
        assert x == 100, f"Expected x=100, got {x}"
        assert y == 50, f"Expected y=50, got {y}"
        assert w == 200, f"Expected w=200, got {w}"
        assert h == 200, f"Expected h=200, got {h}"
    
    def test_edge_cases(self):
        """Test edge cases for box conversion."""
        # Test with zero area box
        x, y, w, h = to_coco_xywh([0.5, 0.5, 0, 0], 640, 480)
        assert w == 0 and h == 0, "Zero area box should remain zero"
        
        # Test with full image box
        x, y, w, h = to_coco_xywh([0.5, 0.5, 1.0, 1.0], 640, 480)
        assert w == 640 and h == 480, "Full image box should match image dimensions"


class TestCategoryNormalization:
    """Test category name normalization."""
    
    def test_basic_aliases(self):
        """Test basic category aliases."""
        assert normalize_category("car bike") == "bicycle"
        assert normalize_category("motorbike") == "motorcycle" 
        assert normalize_category("aeroplane") == "airplane"
        assert normalize_category("tv") == "tv"
    
    def test_no_change_cases(self):
        """Test categories that should not change."""
        assert normalize_category("person") == "person"
        assert normalize_category("dog") == "dog"
        assert normalize_category("unknown_category") == "unknown_category"


if __name__ == "__main__":
    pytest.main([__file__])