"""
Grounding DINO wrapper for text-conditioned object detection
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
import sys
import os
from PIL import Image

# Add GroundingDINO to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../GroundingDINO'))

try:
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from groundingdino.util import box_ops
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False


class GroundingDINODetector:
    """Grounding DINO detector for open-vocabulary object detection"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda"):
        
        if not DINO_AVAILABLE:
            raise ImportError("GroundingDINO not available. Install from submodule.")
            
        self.device = torch.device(device)
        self.model = None
        
        # Set default config path if not provided
        if config_path is None:
            # From repo/src/detector.py -> ../GroundingDINO/...
            repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            gdino_dir = os.path.join(os.path.dirname(repo_dir), "GroundingDINO")
            config_path = os.path.join(gdino_dir, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        
        self.config_path = config_path
        
        # Set default checkpoint path if not provided
        if checkpoint_path is None:
            weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")
            checkpoint_path = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
            
        self.checkpoint_path = checkpoint_path
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_model()
    
    def _load_model(self):
        """Load Grounding DINO model"""
        args = SLConfig.fromfile(self.config_path)
        args.device = self.device
        model = build_model(args)
        
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        
        model.eval()
        model = model.to(self.device)
        self.model = model
        print(f"✅ Grounding DINO loaded on {self.device}")
    
    def predict(self, 
                image: torch.Tensor,
                text_prompt: str,
                box_threshold: float = 0.35,
                text_threshold: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Detect objects using text prompt
        
        Returns:
            boxes: (N, 4) in xyxy format
            scores: (N,) confidence scores  
            phrases: List of detected phrases
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        with torch.no_grad():
            # Import required functions and transforms
            from groundingdino.util.inference import predict
            import groundingdino.datasets.transforms as T
            
            # Convert tensor to PIL and apply transforms
            if len(image.shape) == 4:
                image = image.squeeze(0)
            if image.shape[0] == 3:  # CHW format
                image = image.permute(1, 2, 0)
            image_pil = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
            
            # Apply Grounding DINO transforms
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_transformed, _ = transform(image_pil, None)
            
            # Run inference
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
            
        return boxes, logits, phrases
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9
            }
        return {"allocated_gb": 0, "reserved_gb": 0}