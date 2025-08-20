"""
OVOD Pipeline: Open-Vocabulary Object Detection combining Grounding DINO + SAM 2
"""
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import cv2
from PIL import Image

from src.detector import GroundingDINODetector
from src.segmenter import SAM2Segmenter  
from src.nms import filter_detections
from src.prompts import prompt_processor


class OVODPipeline:
    """
    Complete pipeline for open-vocabulary object detection and segmentation
    
    Combines:
    - Grounding DINO for text-conditioned detection (boxes)
    - SAM 2 for precise segmentation (masks)
    - NMS for post-processing
    """
    
    def __init__(self, 
                 device: str = "cuda",
                 dino_config: Optional[str] = None,
                 dino_checkpoint: Optional[str] = None,
                 sam2_config: str = "sam2_hiera_s.yaml",
                 sam2_checkpoint: Optional[str] = None,
                 box_threshold: float = 0.35,
                 text_threshold: float = 0.25,
                 nms_threshold: float = 0.5):
        
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        
        # Initialize components
        detector_kwargs = {"device": device}
        if dino_config:
            detector_kwargs["config_path"] = dino_config
        if dino_checkpoint:
            detector_kwargs["checkpoint_path"] = dino_checkpoint
            
        self.detector = GroundingDINODetector(**detector_kwargs)
        
        self.segmenter = SAM2Segmenter(
            model_cfg=sam2_config,
            checkpoint_path=sam2_checkpoint,
            device=device
        )
        
        print(f"✅ OVOD Pipeline initialized on {device}")
        self._loaded = False
    
    def load_model(self):
        """Load the underlying models"""
        try:
            # Load detector model 
            if hasattr(self.detector, 'load_model'):
                self.detector.load_model()
            
            # Load segmenter model
            if hasattr(self.segmenter, 'load_model'):
                self.segmenter.load_model()
                
            self._loaded = True
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"⚠️  Model loading warning: {e}")
            self._loaded = False
        
    def predict(self, 
                image: np.ndarray, 
                prompt: str,
                return_masks: bool = True,
                max_detections: int = 100,
                **kwargs) -> Dict[str, Any]:
        """
        Run complete detection and segmentation pipeline
        
        Args:
            image: Input image as numpy array (H, W, 3)
            prompt: Text description of objects to find
            return_masks: Whether to generate segmentation masks
            max_detections: Maximum number of detections
            
        Returns:
            results: Dictionary with boxes, labels, scores, masks, timings
        """
        # Self-healing: ensure models are loaded
        if hasattr(self, "load_model") and getattr(self, "_loaded", False) is False:
            self.load_model()
            
        start_time = time.perf_counter()
        h, w = image.shape[:2]
        
        # Validate prompt
        is_valid, message = prompt_processor.validate_prompt(prompt)
        if not is_valid:
            return self._empty_result(h, w, f"Invalid prompt: {message}")
        
        timings = {}
        
        try:
            # 1. Process prompt
            t0 = time.perf_counter()
            grounding_prompt, object_list = prompt_processor.parse_detection_prompt(prompt)
            timings["prompt_processing_ms"] = (time.perf_counter() - t0) * 1000
            
            # 2. Object detection with Grounding DINO
            t0 = time.perf_counter()
            boxes, scores, phrases = self._detect_objects(image, grounding_prompt)
            timings["detection_ms"] = (time.perf_counter() - t0) * 1000
            
            # 3. Post-process detections
            t0 = time.perf_counter()
            if len(boxes) > 0:
                boxes_np = boxes.cpu().numpy()
                scores_np = scores.cpu().numpy()
                
                # Apply NMS and filtering
                boxes_np, scores_np, phrases = filter_detections(
                    boxes_np, scores_np, phrases,
                    score_threshold=self.box_threshold,
                    nms_threshold=self.nms_threshold,
                    max_detections=max_detections
                )
            else:
                boxes_np, scores_np = np.empty((0, 4)), np.empty((0,))
            timings["nms_ms"] = (time.perf_counter() - t0) * 1000
            
            # 4. Generate masks with SAM 2
            masks = []
            if return_masks and len(boxes_np) > 0:
                t0 = time.perf_counter()
                masks = self._generate_masks(image, boxes_np)
                timings["segmentation_ms"] = (time.perf_counter() - t0) * 1000
            else:
                timings["segmentation_ms"] = 0.0
            
            # 5. Prepare results
            timings["total_ms"] = (time.perf_counter() - start_time) * 1000
            
            return {
                "boxes": boxes_np,
                "labels": phrases,
                "scores": scores_np,
                "masks": masks,
                "timings": timings,
                "image_shape": (h, w),
                "prompt": prompt,
                "grounding_prompt": grounding_prompt,
                "object_list": object_list
            }
            
        except Exception as e:
            return self._empty_result(h, w, f"Pipeline error: {str(e)}")
    
    def _detect_objects(self, image: np.ndarray, grounding_prompt: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Run Grounding DINO detection"""
        # Normalize image to [0,1] range and convert to tensor
        if image.dtype == np.uint8:
            image_normalized = image.astype(np.float32) / 255.0
        else:
            image_normalized = image.astype(np.float32)
            
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float().to(self.device)
        
        # Run detection
        boxes, scores, phrases = self.detector.predict(
            image_tensor, 
            grounding_prompt,
            self.box_threshold,
            self.text_threshold
        )
        
        return boxes, scores, phrases
    
    def _generate_masks(self, image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        """Generate segmentation masks using SAM 2"""
        if len(boxes) == 0:
            return []
        
        # Set image for SAM 2
        self.segmenter.set_image(image)
        
        # Generate masks for all boxes
        masks, mask_scores, logits = self.segmenter.predict_masks(boxes)
        
        # Convert to list of individual masks
        mask_list = []
        for i in range(len(masks)):
            mask_list.append(masks[i])
        
        return mask_list
    
    def _empty_result(self, h: int, w: int, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty result with error message"""
        result = {
            "boxes": np.empty((0, 4)),
            "labels": [],
            "scores": np.empty((0,)),
            "masks": [],
            "timings": {"total_ms": 0.0},
            "image_shape": (h, w),
            "prompt": "",
            "grounding_prompt": "",
            "object_list": []
        }
        
        if error:
            result["error"] = error
            
        return result
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage from both models"""
        detector_mem = self.detector.get_memory_usage()
        segmenter_mem = self.segmenter.get_memory_usage()
        
        return {
            "detector_allocated_gb": detector_mem.get("allocated_gb", 0),
            "detector_reserved_gb": detector_mem.get("reserved_gb", 0),
            "segmenter_allocated_gb": segmenter_mem.get("allocated_gb", 0),
            "segmenter_reserved_gb": segmenter_mem.get("reserved_gb", 0),
            "total_allocated_gb": detector_mem.get("allocated_gb", 0) + segmenter_mem.get("allocated_gb", 0),
            "total_reserved_gb": detector_mem.get("reserved_gb", 0) + segmenter_mem.get("reserved_gb", 0)
        }
        
    def clear_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def benchmark_latency(self, 
                         image_sizes: List[Tuple[int, int]] = [(640, 640), (1024, 1024)],
                         prompts: List[str] = ["person", "car", "dog"],
                         num_runs: int = 5) -> Dict[str, Any]:
        """
        Benchmark pipeline latency across different image sizes and prompts
        
        Returns:
            benchmark_results: Detailed timing statistics
        """
        results = {
            "image_sizes": image_sizes,
            "prompts": prompts,
            "num_runs": num_runs,
            "results": []
        }
        
        for h, w in image_sizes:
            for prompt in prompts:
                # Create dummy image
                dummy_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                
                times = []
                for _ in range(num_runs):
                    result = self.predict(dummy_image, prompt, return_masks=True)
                    times.append(result["timings"]["total_ms"])
                
                results["results"].append({
                    "image_size": (h, w),
                    "prompt": prompt,
                    "mean_ms": np.mean(times),
                    "std_ms": np.std(times),
                    "min_ms": np.min(times),
                    "max_ms": np.max(times),
                    "times": times
                })
        
        return results
