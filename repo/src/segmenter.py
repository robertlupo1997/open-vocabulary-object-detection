"""
SAM 2 wrapper for mask generation from bounding boxes
"""
import torch
import numpy as np
from typing import List, Optional, Tuple
import sys
import os

# Multiple paths to find SAM2
sam2_paths = [
    os.path.join(os.path.dirname(__file__), '../../sam2'),
    os.path.join(os.path.dirname(__file__), '../../../sam2'),
    '/mnt/e/DEV-PROJECT/01-open-vocabulary-object-finder/sam2'
]

SAM2_AVAILABLE = False
for path in sam2_paths:
    if path not in sys.path:
        sys.path.append(path)
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        SAM2_AVAILABLE = True
        print(f"✅ SAM2 loaded from: {path}")
        break
    except ImportError:
        continue

if not SAM2_AVAILABLE:
    print("⚠️ SAM2 not available - trying pip installed version")
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        SAM2_AVAILABLE = True
        print("✅ SAM2 loaded from pip installation")
    except ImportError:
        pass


class SAM2Segmenter:
    """SAM 2 segmenter for generating masks from bounding boxes"""
    
    def __init__(self,
                 model_cfg: str = "sam2_hiera_s.yaml", 
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda"):
        
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 not available. Install from submodule.")
            
        self.device = torch.device(device)
        self.predictor = None
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_model()
    
    def _load_model(self):
        """Load SAM 2 model"""
        sam2_model = build_sam2(self.model_cfg, self.checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print(f"✅ SAM 2 loaded on {self.device}")
    
    def set_image(self, image: np.ndarray):
        """Set image for mask prediction"""
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        self.predictor.set_image(image)
    
    def predict_masks(self, 
                     boxes: np.ndarray,
                     multimask_output: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate masks from bounding boxes
        
        Args:
            boxes: (N, 4) in xyxy format
            multimask_output: Whether to return multiple masks per box
            
        Returns:
            masks: (N, H, W) binary masks
            scores: (N,) mask quality scores
            logits: (N, H, W) mask logits
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        if len(boxes) == 0:
            h, w = self.predictor.get_image_shape()
            return np.empty((0, h, w), dtype=bool), np.empty((0,)), np.empty((0, h, w))
        
        # Convert boxes to input format for SAM2
        input_boxes = torch.tensor(boxes, device=self.device)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=multimask_output,
        )
        
        return masks, scores, logits
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9
            }
        return {"allocated_gb": 0, "reserved_gb": 0}