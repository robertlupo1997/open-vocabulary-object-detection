"""
Non-Maximum Suppression utilities for object detection
"""
import torch
import numpy as np
from typing import Union, Tuple


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format
        
    Returns:
        iou: (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Find intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom

    wh = (rb - lt).clamp(min=0)  # width-height
    inter = wh[:, :, 0] * wh[:, :, 1]  # intersection area

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Non-Maximum Suppression
    
    Args:
        boxes: (N, 4) in xyxy format
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        keep: indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    
    # Sort by scores in descending order
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        # Keep the box with highest score
        i = order[0]
        keep.append(i)
        
        if order.numel() == 1:
            break
            
        # Compute IoU with remaining boxes
        iou = box_iou(boxes[i:i+1], boxes[order[1:]])[0]
        
        # Keep boxes with IoU below threshold
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_nms(boxes: torch.Tensor, 
                scores: torch.Tensor, 
                class_ids: torch.Tensor,
                iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Batched NMS for multiple classes
    
    Args:
        boxes: (N, 4) in xyxy format
        scores: (N,) confidence scores
        class_ids: (N,) class indices
        iou_threshold: IoU threshold for suppression
        
    Returns:
        keep: indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    
    # Offset boxes by class_id to prevent cross-class suppression
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def soft_nms(boxes: torch.Tensor, 
             scores: torch.Tensor,
             iou_threshold: float = 0.5,
             sigma: float = 0.5,
             score_threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft NMS implementation
    
    Args:
        boxes: (N, 4) in xyxy format
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression
        sigma: Gaussian parameter for score decay
        score_threshold: Final score threshold
        
    Returns:
        keep_boxes: filtered boxes
        keep_scores: filtered scores
    """
    if boxes.numel() == 0:
        return boxes, scores
    
    boxes = boxes.clone()
    scores = scores.clone()
    
    keep_mask = torch.ones_like(scores, dtype=torch.bool)
    
    for i in range(len(boxes)):
        if not keep_mask[i]:
            continue
            
        # Find boxes that haven't been processed
        remaining_mask = keep_mask.clone()
        remaining_mask[i] = False
        
        if not remaining_mask.any():
            continue
            
        # Compute IoU with remaining boxes
        iou = box_iou(boxes[i:i+1], boxes[remaining_mask])[0]
        
        # Apply soft suppression
        decay = torch.exp(-(iou ** 2) / sigma)
        scores[remaining_mask] *= decay
        
        # Remove boxes below threshold
        keep_mask = scores >= score_threshold
    
    return boxes[keep_mask], scores[keep_mask]


def filter_detections(boxes: np.ndarray,
                     scores: np.ndarray, 
                     labels: list,
                     score_threshold: float = 0.3,
                     nms_threshold: float = 0.5,
                     max_detections: int = 100) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Filter detections with score threshold and NMS
    
    Args:
        boxes: (N, 4) numpy array in xyxy format
        scores: (N,) numpy array of confidence scores
        labels: List of N label strings
        score_threshold: Minimum score to keep
        nms_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections to return
        
    Returns:
        filtered_boxes: (M, 4) filtered boxes
        filtered_scores: (M,) filtered scores  
        filtered_labels: List of M filtered labels
    """
    if len(boxes) == 0:
        return boxes, scores, labels
    
    # Score filtering
    score_mask = scores >= score_threshold
    boxes = boxes[score_mask]
    scores = scores[score_mask]
    labels = [labels[i] for i in range(len(labels)) if score_mask[i]]
    
    if len(boxes) == 0:
        return boxes, scores, labels
    
    # Convert to torch for NMS
    boxes_torch = torch.from_numpy(boxes).float()
    scores_torch = torch.from_numpy(scores).float()
    
    # Apply NMS
    keep_indices = nms(boxes_torch, scores_torch, nms_threshold)
    
    # Limit max detections
    if len(keep_indices) > max_detections:
        keep_indices = keep_indices[:max_detections]
    
    # Filter results
    keep_indices = keep_indices.numpy()
    filtered_boxes = boxes[keep_indices]
    filtered_scores = scores[keep_indices] 
    filtered_labels = [labels[i] for i in keep_indices]
    
    return filtered_boxes, filtered_scores, filtered_labels