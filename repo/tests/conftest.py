"""
Pytest configuration and shared fixtures
"""
import pytest
import numpy as np
import torch
import cv2
from PIL import Image
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple RGB image with some patterns
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add colored rectangles to simulate objects
    cv2.rectangle(image, (50, 50), (200, 200), (255, 0, 0), -1)  # Red rectangle
    cv2.rectangle(image, (300, 100), (500, 300), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(image, (100, 350), (300, 450), (0, 0, 255), -1)  # Blue rectangle
    
    return image

@pytest.fixture
def sample_rgb_image():
    """Create sample RGB image as PIL Image"""
    image_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(image_array)

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_pipeline_results():
    """Mock pipeline results for testing"""
    return {
        "boxes": np.array([[50, 50, 200, 200], [300, 100, 500, 300]]),
        "labels": ["person", "car"],
        "scores": np.array([0.9, 0.8]),
        "masks": [
            np.random.choice([True, False], (480, 640)),
            np.random.choice([True, False], (480, 640))
        ],
        "timings": {
            "total_ms": 150.0,
            "detection_ms": 100.0,
            "segmentation_ms": 45.0,
            "nms_ms": 5.0
        },
        "image_shape": (480, 640),
        "prompt": "person, car",
        "grounding_prompt": "person . car .",
        "object_list": ["person", "car"]
    }

@pytest.fixture
def device():
    """Get appropriate device for testing"""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="session")
def skip_gpu_tests():
    """Skip GPU tests if CUDA not available"""
    return not torch.cuda.is_available()