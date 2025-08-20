"""
Unit tests for NMS utilities
"""
import pytest
import torch
import numpy as np

from src.nms import (
    box_iou, nms, batched_nms, soft_nms, filter_detections
)


class TestBoxIoU:
    """Test box IoU calculation"""
    
    def test_identical_boxes(self):
        """Test IoU of identical boxes should be 1.0"""
        boxes1 = torch.tensor([[0, 0, 10, 10]])
        boxes2 = torch.tensor([[0, 0, 10, 10]])
        
        iou = box_iou(boxes1, boxes2)
        
        assert iou.shape == (1, 1)
        assert torch.allclose(iou, torch.tensor([[1.0]]))
    
    def test_non_overlapping_boxes(self):
        """Test IoU of non-overlapping boxes should be 0.0"""
        boxes1 = torch.tensor([[0, 0, 5, 5]])
        boxes2 = torch.tensor([[10, 10, 15, 15]])
        
        iou = box_iou(boxes1, boxes2)
        
        assert iou.shape == (1, 1)
        assert torch.allclose(iou, torch.tensor([[0.0]]))
    
    def test_half_overlapping_boxes(self):
        """Test IoU of half-overlapping boxes"""
        boxes1 = torch.tensor([[0, 0, 10, 10]])
        boxes2 = torch.tensor([[5, 0, 15, 10]])
        
        iou = box_iou(boxes1, boxes2)
        
        # Intersection: 5x10=50, Union: 100+100-50=150, IoU: 50/150=1/3
        expected_iou = 1.0 / 3.0
        assert iou.shape == (1, 1)
        assert torch.allclose(iou, torch.tensor([[expected_iou]]), atol=1e-6)
    
    def test_multiple_boxes(self):
        """Test IoU calculation with multiple boxes"""
        boxes1 = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]])
        boxes2 = torch.tensor([[5, 0, 15, 10], [25, 25, 35, 35]])
        
        iou = box_iou(boxes1, boxes2)
        
        assert iou.shape == (2, 2)
        # First box with first: 1/3, with second: 0
        # Second box with first: 0, with second: 1/3
        expected = torch.tensor([[1/3, 0], [0, 1/3]])
        assert torch.allclose(iou, expected, atol=1e-6)


class TestNMS:
    """Test Non-Maximum Suppression"""
    
    def test_empty_input(self):
        """Test NMS with empty input"""
        boxes = torch.empty((0, 4))
        scores = torch.empty((0,))
        
        keep = nms(boxes, scores)
        
        assert keep.shape == (0,)
    
    def test_single_box(self):
        """Test NMS with single box"""
        boxes = torch.tensor([[0, 0, 10, 10]], dtype=torch.float)
        scores = torch.tensor([0.9])
        
        keep = nms(boxes, scores)
        
        assert keep.tolist() == [0]
    
    def test_non_overlapping_boxes(self):
        """Test NMS with non-overlapping boxes (all should be kept)"""
        boxes = torch.tensor([
            [0, 0, 5, 5],
            [10, 10, 15, 15],
            [20, 20, 25, 25]
        ], dtype=torch.float)
        scores = torch.tensor([0.9, 0.8, 0.7])
        
        keep = nms(boxes, scores, iou_threshold=0.5)
        
        assert sorted(keep.tolist()) == [0, 1, 2]
    
    def test_overlapping_boxes(self):
        """Test NMS with overlapping boxes"""
        boxes = torch.tensor([
            [0, 0, 10, 10],    # High score
            [5, 0, 15, 10],    # Overlaps with first, lower score
            [20, 20, 30, 30]   # Non-overlapping
        ], dtype=torch.float)
        scores = torch.tensor([0.9, 0.8, 0.7])
        
        keep = nms(boxes, scores, iou_threshold=0.3)
        
        # Should keep highest scoring box and non-overlapping box
        assert sorted(keep.tolist()) == [0, 2]
    
    def test_score_ordering(self):
        """Test that NMS respects score ordering"""
        boxes = torch.tensor([
            [0, 0, 10, 10],
            [5, 0, 15, 10]  # Same overlap but different scores
        ], dtype=torch.float)
        
        # Test with first box having higher score
        scores1 = torch.tensor([0.9, 0.8])
        keep1 = nms(boxes, scores1, iou_threshold=0.3)
        assert keep1.tolist() == [0]
        
        # Test with second box having higher score
        scores2 = torch.tensor([0.8, 0.9])
        keep2 = nms(boxes, scores2, iou_threshold=0.3)
        assert keep2.tolist() == [1]


class TestBatchedNMS:
    """Test batched NMS for multiple classes"""
    
    def test_same_class_suppression(self):
        """Test that boxes of same class suppress each other"""
        boxes = torch.tensor([
            [0, 0, 10, 10],
            [5, 0, 15, 10]
        ], dtype=torch.float)
        scores = torch.tensor([0.9, 0.8])
        class_ids = torch.tensor([0, 0])  # Same class
        
        keep = batched_nms(boxes, scores, class_ids, iou_threshold=0.3)
        
        assert keep.tolist() == [0]  # Higher scoring box kept
    
    def test_different_class_no_suppression(self):
        """Test that boxes of different classes don't suppress each other"""
        boxes = torch.tensor([
            [0, 0, 10, 10],
            [5, 0, 15, 10]  # Overlapping but different class
        ], dtype=torch.float)
        scores = torch.tensor([0.9, 0.8])
        class_ids = torch.tensor([0, 1])  # Different classes
        
        keep = batched_nms(boxes, scores, class_ids, iou_threshold=0.3)
        
        assert sorted(keep.tolist()) == [0, 1]  # Both kept


class TestSoftNMS:
    """Test Soft NMS implementation"""
    
    def test_soft_nms_preserves_boxes(self):
        """Test that soft NMS preserves more boxes than hard NMS"""
        boxes = torch.tensor([
            [0, 0, 10, 10],
            [5, 0, 15, 10],
            [20, 20, 30, 30]
        ], dtype=torch.float)
        scores = torch.tensor([0.9, 0.8, 0.7])
        
        # Hard NMS
        hard_keep = nms(boxes, scores, iou_threshold=0.3)
        
        # Soft NMS
        soft_boxes, soft_scores = soft_nms(
            boxes, scores, 
            iou_threshold=0.3, 
            score_threshold=0.1
        )
        
        # Soft NMS should preserve more boxes (or at least same number)
        assert len(soft_boxes) >= len(hard_keep)
        assert len(soft_scores) == len(soft_boxes)
    
    def test_soft_nms_score_decay(self):
        """Test that soft NMS decays overlapping scores"""
        boxes = torch.tensor([
            [0, 0, 10, 10],
            [5, 0, 15, 10]  # Overlapping
        ], dtype=torch.float)
        scores = torch.tensor([0.9, 0.8])
        
        soft_boxes, soft_scores = soft_nms(boxes, scores, sigma=0.5)
        
        # First box score should remain same (highest)
        assert torch.allclose(soft_scores[0], scores[0])
        
        # Second box score should be decayed
        assert soft_scores[1] < scores[1]


class TestFilterDetections:
    """Test detection filtering function"""
    
    def test_score_filtering(self):
        """Test filtering by score threshold"""
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
        scores = np.array([0.9, 0.2])
        labels = ["person", "car"]
        
        filtered_boxes, filtered_scores, filtered_labels = filter_detections(
            boxes, scores, labels, score_threshold=0.5
        )
        
        assert len(filtered_boxes) == 1
        assert len(filtered_scores) == 1
        assert len(filtered_labels) == 1
        assert filtered_labels[0] == "person"
    
    def test_max_detections_limit(self):
        """Test max detections limit"""
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]])
        scores = np.array([0.9, 0.8, 0.7])
        labels = ["person", "car", "dog"]
        
        filtered_boxes, filtered_scores, filtered_labels = filter_detections(
            boxes, scores, labels, max_detections=2
        )
        
        assert len(filtered_boxes) == 2
        assert len(filtered_scores) == 2
        assert len(filtered_labels) == 2
    
    def test_empty_input(self):
        """Test filtering with empty input"""
        boxes = np.empty((0, 4))
        scores = np.empty((0,))
        labels = []
        
        filtered_boxes, filtered_scores, filtered_labels = filter_detections(
            boxes, scores, labels
        )
        
        assert len(filtered_boxes) == 0
        assert len(filtered_scores) == 0
        assert len(filtered_labels) == 0
    
    def test_nms_filtering(self):
        """Test NMS filtering in detection pipeline"""
        # Create overlapping boxes with different scores
        boxes = np.array([
            [0, 0, 10, 10],
            [5, 0, 15, 10],  # Overlapping with first
            [20, 20, 30, 30]  # Non-overlapping
        ])
        scores = np.array([0.9, 0.8, 0.7])
        labels = ["person", "person", "car"]
        
        filtered_boxes, filtered_scores, filtered_labels = filter_detections(
            boxes, scores, labels, nms_threshold=0.3
        )
        
        # Should keep highest scoring overlapping box and non-overlapping box
        assert len(filtered_boxes) == 2
        assert "person" in filtered_labels
        assert "car" in filtered_labels