"""
Integration tests for OVOD pipeline
"""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from ovod.pipeline import OVODPipeline


class TestOVODPipeline:
    """Test OVOD pipeline integration"""
    
    def test_pipeline_initialization_cpu(self):
        """Test pipeline initialization on CPU"""
        pipeline = OVODPipeline(device="cpu")
        
        assert pipeline.device == "cpu"
        assert pipeline.box_threshold == 0.35
        assert pipeline.text_threshold == 0.25
        assert pipeline.nms_threshold == 0.5
        assert pipeline.detector is not None
        assert pipeline.segmenter is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pipeline_initialization_cuda(self):
        """Test pipeline initialization on CUDA"""
        pipeline = OVODPipeline(device="cuda")
        
        assert pipeline.device == "cuda"
        assert pipeline.detector is not None
        assert pipeline.segmenter is not None
    
    def test_pipeline_custom_thresholds(self):
        """Test pipeline with custom thresholds"""
        pipeline = OVODPipeline(
            device="cpu",
            box_threshold=0.5,
            text_threshold=0.3,
            nms_threshold=0.6
        )
        
        assert pipeline.box_threshold == 0.5
        assert pipeline.text_threshold == 0.3
        assert pipeline.nms_threshold == 0.6
    
    def test_predict_empty_result(self, sample_image):
        """Test prediction with no detections"""
        pipeline = OVODPipeline(device="cpu")
        
        # Mock detector to return empty results
        with patch.object(pipeline, '_detect_objects') as mock_detect:
            mock_detect.return_value = (
                torch.empty((0, 4)),  # boxes
                torch.empty((0,)),    # scores
                []                    # phrases
            )
            
            result = pipeline.predict(sample_image, "nonexistent_object")
            
            assert len(result["boxes"]) == 0
            assert len(result["labels"]) == 0
            assert len(result["scores"]) == 0
            assert len(result["masks"]) == 0
            assert "timings" in result
            assert result["image_shape"] == sample_image.shape[:2]
    
    def test_predict_with_detections(self, sample_image):
        """Test prediction with mock detections"""
        pipeline = OVODPipeline(device="cpu")
        
        # Mock detector to return some detections
        mock_boxes = torch.tensor([[50, 50, 200, 200], [300, 100, 500, 300]])
        mock_scores = torch.tensor([0.9, 0.8])
        mock_phrases = ["person", "car"]
        
        with patch.object(pipeline, '_detect_objects') as mock_detect:
            mock_detect.return_value = (mock_boxes, mock_scores, mock_phrases)
            
            with patch.object(pipeline, '_generate_masks') as mock_segment:
                mock_masks = [
                    np.random.choice([True, False], sample_image.shape[:2]),
                    np.random.choice([True, False], sample_image.shape[:2])
                ]
                mock_segment.return_value = mock_masks
                
                result = pipeline.predict(sample_image, "person, car")
                
                assert len(result["boxes"]) == 2
                assert len(result["labels"]) == 2
                assert len(result["scores"]) == 2
                assert len(result["masks"]) == 2
                assert result["labels"] == ["person", "car"]
                assert "timings" in result
    
    def test_predict_invalid_prompt(self, sample_image):
        """Test prediction with invalid prompt"""
        pipeline = OVODPipeline(device="cpu")
        
        result = pipeline.predict(sample_image, "")
        
        assert "error" in result
        assert "Invalid prompt" in result["error"]
    
    def test_predict_without_masks(self, sample_image):
        """Test prediction without mask generation"""
        pipeline = OVODPipeline(device="cpu")
        
        mock_boxes = torch.tensor([[50, 50, 200, 200]])
        mock_scores = torch.tensor([0.9])
        mock_phrases = ["person"]
        
        with patch.object(pipeline, '_detect_objects') as mock_detect:
            mock_detect.return_value = (mock_boxes, mock_scores, mock_phrases)
            
            result = pipeline.predict(sample_image, "person", return_masks=False)
            
            assert len(result["boxes"]) == 1
            assert len(result["masks"]) == 0
            assert result["timings"]["segmentation_ms"] == 0.0
    
    def test_predict_with_max_detections(self, sample_image):
        """Test prediction with max detections limit"""
        pipeline = OVODPipeline(device="cpu")
        
        # Mock many detections
        num_detections = 10
        mock_boxes = torch.rand((num_detections, 4)) * 100
        mock_scores = torch.rand(num_detections)
        mock_phrases = [f"object_{i}" for i in range(num_detections)]
        
        with patch.object(pipeline, '_detect_objects') as mock_detect:
            mock_detect.return_value = (mock_boxes, mock_scores, mock_phrases)
            
            result = pipeline.predict(sample_image, "many objects", max_detections=5)
            
            assert len(result["boxes"]) <= 5
            assert len(result["labels"]) <= 5
            assert len(result["scores"]) <= 5
    
    def test_get_memory_usage(self):
        """Test memory usage reporting"""
        pipeline = OVODPipeline(device="cpu")
        
        with patch.object(pipeline.detector, 'get_memory_usage') as mock_det_mem:
            with patch.object(pipeline.segmenter, 'get_memory_usage') as mock_seg_mem:
                mock_det_mem.return_value = {"allocated_gb": 1.0, "reserved_gb": 1.5}
                mock_seg_mem.return_value = {"allocated_gb": 0.5, "reserved_gb": 0.8}
                
                memory = pipeline.get_memory_usage()
                
                assert memory["detector_allocated_gb"] == 1.0
                assert memory["segmenter_allocated_gb"] == 0.5
                assert memory["total_allocated_gb"] == 1.5
                assert memory["total_reserved_gb"] == 2.3
    
    def test_clear_cache(self):
        """Test cache clearing"""
        pipeline = OVODPipeline(device="cpu")
        
        # Should not raise any errors
        pipeline.clear_cache()
    
    def test_benchmark_latency(self, sample_image):
        """Test latency benchmarking"""
        pipeline = OVODPipeline(device="cpu")
        
        with patch.object(pipeline, 'predict') as mock_predict:
            # Mock timing results
            mock_predict.return_value = {
                "boxes": np.array([[50, 50, 200, 200]]),
                "labels": ["person"],
                "scores": np.array([0.9]),
                "timings": {"total_ms": 100.0}
            }
            
            results = pipeline.benchmark_latency(
                image_sizes=[(640, 640)],
                prompts=["person"],
                num_runs=3
            )
            
            assert "image_sizes" in results
            assert "prompts" in results
            assert "num_runs" in results
            assert "results" in results
            assert len(results["results"]) == 1  # 1 size x 1 prompt
            
            result = results["results"][0]
            assert "mean_ms" in result
            assert "std_ms" in result
            assert "times" in result
            assert len(result["times"]) == 3


class TestPipelineErrorHandling:
    """Test error handling in pipeline"""
    
    def test_prediction_exception_handling(self, sample_image):
        """Test that prediction exceptions are handled gracefully"""
        pipeline = OVODPipeline(device="cpu")
        
        with patch.object(pipeline, '_detect_objects') as mock_detect:
            mock_detect.side_effect = Exception("Detector failed")
            
            result = pipeline.predict(sample_image, "person")
            
            assert "error" in result
            assert "Pipeline error" in result["error"]
            assert "Detector failed" in result["error"]
    
    def test_invalid_image_shape(self):
        """Test handling of invalid image shapes"""
        pipeline = OVODPipeline(device="cpu")
        
        # Test with 2D image (missing channel dimension)
        invalid_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        with pytest.raises((IndexError, ValueError)):
            pipeline.predict(invalid_image, "person")
    
    def test_empty_image(self):
        """Test handling of empty images"""
        pipeline = OVODPipeline(device="cpu")
        
        empty_image = np.empty((0, 0, 3), dtype=np.uint8)
        
        # Should handle gracefully or raise appropriate error
        result = pipeline.predict(empty_image, "person")
        # Behavior may vary - either error or empty result is acceptable
        assert "error" in result or len(result["boxes"]) == 0


class TestPipelineComponents:
    """Test individual pipeline components"""
    
    def test_detect_objects_method(self, sample_image):
        """Test object detection method"""
        pipeline = OVODPipeline(device="cpu")
        
        with patch.object(pipeline.detector, 'predict') as mock_predict:
            mock_predict.return_value = (
                torch.tensor([[50, 50, 200, 200]]),
                torch.tensor([0.9]),
                ["person"]
            )
            
            boxes, scores, phrases = pipeline._detect_objects(sample_image, "person .")
            
            assert len(boxes) == 1
            assert len(scores) == 1
            assert len(phrases) == 1
            assert phrases[0] == "person"
    
    def test_generate_masks_method(self, sample_image):
        """Test mask generation method"""
        pipeline = OVODPipeline(device="cpu")
        
        boxes = np.array([[50, 50, 200, 200], [300, 100, 500, 300]])
        
        with patch.object(pipeline.segmenter, 'set_image') as mock_set_image:
            with patch.object(pipeline.segmenter, 'predict_masks') as mock_predict:
                mock_masks = np.random.choice([True, False], (2, 480, 640))
                mock_scores = np.array([0.9, 0.8])
                mock_logits = np.random.randn(2, 480, 640)
                
                mock_predict.return_value = (mock_masks, mock_scores, mock_logits)
                
                masks = pipeline._generate_masks(sample_image, boxes)
                
                assert len(masks) == 2
                mock_set_image.assert_called_once_with(sample_image)
                mock_predict.assert_called_once_with(boxes)
    
    def test_generate_masks_empty_boxes(self, sample_image):
        """Test mask generation with empty boxes"""
        pipeline = OVODPipeline(device="cpu")
        
        empty_boxes = np.empty((0, 4))
        masks = pipeline._generate_masks(sample_image, empty_boxes)
        
        assert len(masks) == 0
    
    def test_empty_result_method(self):
        """Test empty result generation"""
        pipeline = OVODPipeline(device="cpu")
        
        result = pipeline._empty_result(480, 640, "Test error")
        
        assert len(result["boxes"]) == 0
        assert len(result["labels"]) == 0
        assert len(result["scores"]) == 0
        assert len(result["masks"]) == 0
        assert result["image_shape"] == (480, 640)
        assert result["error"] == "Test error"