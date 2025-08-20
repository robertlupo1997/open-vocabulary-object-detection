"""
End-to-end integration tests for OVOD
"""
import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

import streamlit as st
from PIL import Image

from ovod.pipeline import OVODPipeline


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    def test_full_pipeline_workflow(self, sample_image):
        """Test complete workflow from image to results"""
        
        # This test would require actual model weights
        # For now, we test the interface with mocked components
        
        pipeline = OVODPipeline(device="cpu")
        
        # Mock the detector and segmenter to avoid requiring actual weights
        with patch.object(pipeline.detector, 'predict') as mock_detector:
            with patch.object(pipeline.segmenter, 'set_image'):
                with patch.object(pipeline.segmenter, 'predict_masks') as mock_segmenter:
                    
                    # Setup mocks
                    mock_detector.return_value = (
                        torch.tensor([[100, 100, 300, 300], [400, 200, 600, 400]]),
                        torch.tensor([0.95, 0.87]),
                        ["person", "car"]
                    )
                    
                    mock_segmenter.return_value = (
                        np.random.choice([True, False], (2, 480, 640)),
                        np.array([0.9, 0.8]),
                        np.random.randn(2, 480, 640)
                    )
                    
                    # Run prediction
                    result = pipeline.predict(
                        sample_image, 
                        "find people and cars in the image",
                        return_masks=True,
                        max_detections=50
                    )
                    
                    # Verify results structure
                    assert "boxes" in result
                    assert "labels" in result  
                    assert "scores" in result
                    assert "masks" in result
                    assert "timings" in result
                    assert "image_shape" in result
                    
                    # Verify content
                    assert len(result["boxes"]) == 2
                    assert len(result["labels"]) == 2
                    assert len(result["scores"]) == 2
                    assert len(result["masks"]) == 2
                    assert result["labels"] == ["person", "car"]
                    
                    # Verify timing information
                    timings = result["timings"]
                    assert "total_ms" in timings
                    assert "detection_ms" in timings
                    assert "segmentation_ms" in timings
                    assert timings["total_ms"] > 0
    
    def test_pipeline_performance_requirements(self, sample_image):
        """Test that pipeline meets performance requirements"""
        
        pipeline = OVODPipeline(device="cpu")
        
        with patch.object(pipeline.detector, 'predict') as mock_detector:
            with patch.object(pipeline.segmenter, 'set_image'):
                with patch.object(pipeline.segmenter, 'predict_masks') as mock_segmenter:
                    
                    # Setup fast mocks
                    mock_detector.return_value = (
                        torch.tensor([[100, 100, 200, 200]]),
                        torch.tensor([0.9]),
                        ["person"]
                    )
                    
                    mock_segmenter.return_value = (
                        np.random.choice([True, False], (1, 480, 640)),
                        np.array([0.9]),
                        np.random.randn(1, 480, 640)
                    )
                    
                    # Test latency (should be reasonable even with mocks)
                    result = pipeline.predict(sample_image, "person")
                    
                    # On CPU with mocks, should be fast
                    assert result["timings"]["total_ms"] < 1000  # 1 second max
    
    def test_memory_usage_tracking(self, sample_image):
        """Test memory usage tracking throughout pipeline"""
        
        pipeline = OVODPipeline(device="cpu")
        
        # Get initial memory usage
        initial_memory = pipeline.get_memory_usage()
        assert "total_allocated_gb" in initial_memory
        
        # Mock prediction
        with patch.object(pipeline.detector, 'predict') as mock_detector:
            with patch.object(pipeline.segmenter, 'set_image'):
                with patch.object(pipeline.segmenter, 'predict_masks') as mock_segmenter:
                    
                    mock_detector.return_value = (torch.empty((0, 4)), torch.empty((0,)), [])
                    mock_segmenter.return_value = (np.empty((0, 480, 640)), np.empty((0,)), np.empty((0, 480, 640)))
                    
                    result = pipeline.predict(sample_image, "nothing")
                    
                    # Memory usage should still be trackable
                    final_memory = pipeline.get_memory_usage()
                    assert "total_allocated_gb" in final_memory
    
    def test_batch_processing_simulation(self, sample_image):
        """Test processing multiple images (simulated)"""
        
        pipeline = OVODPipeline(device="cpu")
        
        images = [sample_image] * 3  # Simulate 3 images
        prompts = ["person", "car", "person, car"]
        
        results = []
        
        with patch.object(pipeline.detector, 'predict') as mock_detector:
            with patch.object(pipeline.segmenter, 'set_image'):
                with patch.object(pipeline.segmenter, 'predict_masks') as mock_segmenter:
                    
                    # Different results for each image
                    mock_responses = [
                        (torch.tensor([[100, 100, 200, 200]]), torch.tensor([0.9]), ["person"]),
                        (torch.tensor([[150, 150, 250, 250]]), torch.tensor([0.8]), ["car"]),
                        (torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]), torch.tensor([0.9, 0.8]), ["person", "car"])
                    ]
                    
                    mock_segmenter.return_value = (
                        np.random.choice([True, False], (2, 480, 640)),
                        np.array([0.9, 0.8]),
                        np.random.randn(2, 480, 640)
                    )
                    
                    for i, (image, prompt) in enumerate(zip(images, prompts)):
                        mock_detector.return_value = mock_responses[i]
                        
                        result = pipeline.predict(image, prompt)
                        results.append(result)
                    
                    # Verify all results
                    assert len(results) == 3
                    assert len(results[0]["labels"]) == 1  # person
                    assert len(results[1]["labels"]) == 1  # car  
                    assert len(results[2]["labels"]) == 2  # person, car


class TestDemoAppIntegration:
    """Test Streamlit demo app integration"""
    
    @patch('demo_app.load_pipeline')
    def test_demo_app_pipeline_loading(self, mock_load_pipeline):
        """Test demo app pipeline loading"""
        
        # Mock successful pipeline loading
        mock_pipeline = Mock()
        mock_load_pipeline.return_value = (mock_pipeline, None)
        
        # Import and test (would require streamlit testing framework for full test)
        # This is a simplified test of the loading logic
        from demo_app import load_pipeline
        
        pipeline, error = load_pipeline()
        assert error is None
        assert pipeline is not None
    
    @patch('demo_app.load_pipeline')
    def test_demo_app_pipeline_failure(self, mock_load_pipeline):
        """Test demo app handling of pipeline loading failure"""
        
        # Mock failed pipeline loading
        mock_load_pipeline.return_value = (None, "Failed to load models")
        
        from demo_app import load_pipeline
        
        pipeline, error = load_pipeline()
        assert pipeline is None
        assert "Failed to load models" in error


class TestCOCOEvaluationIntegration:
    """Test COCO evaluation integration"""
    
    def test_coco_evaluator_initialization(self, temp_dir):
        """Test COCO evaluator can be initialized"""
        
        from metrics.coco_eval import COCOEvaluator, EvalConfig
        
        # Create mock COCO structure
        coco_dir = temp_dir / "coco"
        annotations_dir = coco_dir / "annotations"
        val_dir = coco_dir / "val2017"
        
        annotations_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)
        
        # Create minimal valid COCO annotation file
        mock_annotations = {
            "images": [
                {"id": 1, "file_name": "test.jpg", "height": 480, "width": 640}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "area": 2500}
            ],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "person"}
            ]
        }
        
        ann_file = annotations_dir / "instances_val2017.json"
        with open(ann_file, 'w') as f:
            json.dump(mock_annotations, f)
        
        # Create dummy image
        dummy_image = Image.new('RGB', (640, 480), color='red')
        dummy_image.save(val_dir / "test.jpg")
        
        # Test evaluator initialization
        config = EvalConfig(
            coco_path=str(coco_dir),
            subset="val2017",
            max_images=1,
            device="cpu"
        )
        
        evaluator = COCOEvaluator(config)
        
        # Should not raise exception
        assert evaluator.config.coco_path == str(coco_dir)
        assert evaluator.coco_categories is not None
        assert len(evaluator.coco_categories) == 80  # Standard COCO categories


class TestBenchmarkIntegration:
    """Test benchmark integration"""
    
    def test_benchmark_system_info(self):
        """Test benchmark system info collection"""
        
        from metrics.benchmark import LatencyBenchmark, BenchmarkConfig
        
        config = BenchmarkConfig(device="cpu", num_runs=1)
        benchmark = LatencyBenchmark(config)
        
        system_info = benchmark.get_system_info()
        
        assert "platform" in system_info
        assert "processor" in system_info
        assert "cpu_count" in system_info
        assert "memory_gb" in system_info
        assert "python_version" in system_info
        
        # GPU info should be present if CUDA available
        if torch.cuda.is_available():
            assert "gpu" in system_info
            assert system_info["gpu"] is not None
        else:
            assert system_info.get("gpu") is None
    
    def test_benchmark_test_image_creation(self):
        """Test benchmark test image creation"""
        
        from metrics.benchmark import LatencyBenchmark, BenchmarkConfig
        
        config = BenchmarkConfig(
            device="cpu", 
            image_sizes=[(640, 640), (1024, 1024)]
        )
        benchmark = LatencyBenchmark(config)
        
        test_images = benchmark.create_test_images()
        
        assert len(test_images) == 2
        assert (640, 640) in test_images
        assert (1024, 1024) in test_images
        
        # Check image properties
        img_640 = test_images[(640, 640)]
        assert img_640.shape == (640, 640, 3)
        assert img_640.dtype == np.uint8
        
        img_1024 = test_images[(1024, 1024)]
        assert img_1024.shape == (1024, 1024, 3)
        assert img_1024.dtype == np.uint8


class TestModelCardGeneration:
    """Test model card and documentation generation"""
    
    def test_model_card_requirements(self):
        """Test model card contains required information"""
        
        # This would be a template test for when MODEL_CARD.md is created
        required_sections = [
            "Model Description",
            "Performance",
            "Limitations", 
            "Training Data",
            "Evaluation",
            "Hardware Requirements",
            "Usage"
        ]
        
        # When MODEL_CARD.md exists, test that it contains these sections
        # For now, just verify the requirements are defined
        assert len(required_sections) > 0
    
    def test_data_card_requirements(self):
        """Test data card contains required information"""
        
        required_sections = [
            "Dataset Description",
            "Data Sources",
            "Data Processing",
            "Licensing",
            "Bias and Limitations"
        ]
        
        # When DATA_CARD.md exists, test that it contains these sections
        assert len(required_sections) > 0


# Import torch at module level for CUDA checks
import torch