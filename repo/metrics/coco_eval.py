"""
COCO evaluation for OVOD pipeline
"""
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from dataclasses import dataclass

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask
import cv2

from ovod.pipeline import OVODPipeline


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    coco_path: str = "data/coco"
    subset: str = "val2017"  # val2017, test2017, or custom subset
    max_images: int = 1000  # For faster evaluation
    device: str = "cuda"
    batch_size: int = 1
    image_size: int = 640
    
    # Detection thresholds
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    nms_threshold: float = 0.5
    max_detections: int = 100
    
    # Output paths
    results_dir: str = "metrics/results"
    save_predictions: bool = True


class COCOEvaluator:
    """COCO evaluation for OVOD pipeline"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.pipeline = None
        self.coco_gt = None
        self.category_mapping = {}
        
        # COCO category names for prompts
        self.coco_categories = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
    def setup(self):
        """Initialize pipeline and COCO dataset"""
        
        print(f"🔧 Setting up OVOD evaluation...")
        
        # Initialize pipeline
        self.pipeline = OVODPipeline(
            device=self.config.device,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            nms_threshold=self.config.nms_threshold
        )
        
        # Load COCO dataset
        ann_file = Path(self.config.coco_path) / "annotations" / f"instances_{self.config.subset}.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotations not found: {ann_file}")
            
        self.coco_gt = COCO(str(ann_file))
        
        # Create category mapping
        cats = self.coco_gt.loadCats(self.coco_gt.getCatIds())
        self.category_mapping = {cat['name']: cat['id'] for cat in cats}
        
        print(f"✅ Loaded COCO {self.config.subset} with {len(self.coco_gt.getImgIds())} images")
        print(f"✅ Found {len(self.category_mapping)} categories")
        
    def create_detection_prompt(self) -> str:
        """Create comprehensive prompt for all COCO categories"""
        # Use all COCO categories as prompt
        prompt = " . ".join(self.coco_categories) + " ."
        return prompt
    
    def load_image(self, image_info: dict) -> np.ndarray:
        """Load and preprocess image"""
        
        img_path = Path(self.config.coco_path) / self.config.subset / image_info['file_name']
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Resize if needed
        if self.config.image_size:
            # Maintain aspect ratio
            w, h = image.size
            scale = min(self.config.image_size / w, self.config.image_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        
        return np.array(image)
    
    def convert_to_coco_format(self, 
                              results: dict, 
                              image_id: int, 
                              original_size: Tuple[int, int]) -> List[dict]:
        """Convert OVOD results to COCO detection format"""
        
        detections = []
        boxes = results["boxes"]
        labels = results["labels"]
        scores = results["scores"]
        masks = results.get("masks", [])
        
        orig_h, orig_w = original_size
        img_h, img_w = results["image_shape"]
        
        # Scale factor to original image size
        scale_x = orig_w / img_w
        scale_y = orig_h / img_h
        
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            
            # Map label to COCO category ID
            if label not in self.category_mapping:
                continue  # Skip unknown categories
                
            category_id = self.category_mapping[label]
            
            # Scale box to original image size
            x1, y1, x2, y2 = box
            x1_orig = x1 * scale_x
            y1_orig = y1 * scale_y
            w_orig = (x2 - x1) * scale_x
            h_orig = (y2 - y1) * scale_y
            
            detection = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1_orig, y1_orig, w_orig, h_orig],  # COCO format: [x, y, width, height]
                "score": float(score),
                "area": w_orig * h_orig
            }
            
            # Add segmentation if available
            if masks and i < len(masks):
                mask = masks[i]
                
                # Resize mask to original image size
                mask_resized = cv2.resize(
                    mask.astype(np.uint8), 
                    (orig_w, orig_h), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                
                # Convert to COCO RLE format
                rle = coco_mask.encode(np.asfortranarray(mask_resized))
                rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string
                detection["segmentation"] = rle
                
            detections.append(detection)
        
        return detections
    
    def evaluate_subset(self, max_images: Optional[int] = None) -> Dict:
        """Evaluate on a subset of COCO images"""
        
        # Get image IDs
        img_ids = self.coco_gt.getImgIds()
        if max_images:
            img_ids = img_ids[:max_images]
        
        print(f"📊 Evaluating on {len(img_ids)} images...")
        
        # Create detection prompt
        prompt = self.create_detection_prompt()
        print(f"🔍 Using prompt with {len(self.coco_categories)} categories")
        
        all_detections = []
        timing_stats = []
        
        # Process images
        for i, img_id in enumerate(img_ids):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(img_ids)} images...")
            
            try:
                # Load image info and image
                img_info = self.coco_gt.loadImgs(img_id)[0]
                image = self.load_image(img_info)
                original_size = (img_info['height'], img_info['width'])
                
                # Run detection
                start_time = time.perf_counter()
                results = self.pipeline.predict(
                    image, 
                    prompt,
                    return_masks=True,
                    max_detections=self.config.max_detections
                )
                inference_time = time.perf_counter() - start_time
                
                # Track timing
                timing_stats.append({
                    "image_id": img_id,
                    "inference_ms": inference_time * 1000,
                    "detection_ms": results["timings"].get("detection_ms", 0),
                    "segmentation_ms": results["timings"].get("segmentation_ms", 0),
                    "total_ms": results["timings"].get("total_ms", 0),
                    "num_detections": len(results["boxes"])
                })
                
                # Convert to COCO format
                detections = self.convert_to_coco_format(results, img_id, original_size)
                all_detections.extend(detections)
                
            except Exception as e:
                print(f"❌ Error processing image {img_id}: {e}")
                continue
        
        print(f"✅ Generated {len(all_detections)} detections")
        
        # Save predictions
        if self.config.save_predictions:
            Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
            pred_file = Path(self.config.results_dir) / f"predictions_{self.config.subset}.json"
            with open(pred_file, 'w') as f:
                json.dump(all_detections, f)
            print(f"💾 Saved predictions to {pred_file}")
        
        # Calculate timing statistics
        timing_summary = self._calculate_timing_stats(timing_stats)
        
        # Run COCO evaluation
        eval_results = self._run_coco_evaluation(all_detections)
        
        return {
            "detections": all_detections,
            "timing": timing_summary,
            "evaluation": eval_results,
            "config": self.config.__dict__
        }
    
    def _calculate_timing_stats(self, timing_stats: List[dict]) -> Dict:
        """Calculate timing statistics"""
        
        if not timing_stats:
            return {}
        
        # Convert to arrays for easier calculation
        inference_times = [t["inference_ms"] for t in timing_stats]
        detection_times = [t["detection_ms"] for t in timing_stats]
        segmentation_times = [t["segmentation_ms"] for t in timing_stats]
        total_times = [t["total_ms"] for t in timing_stats]
        num_detections = [t["num_detections"] for t in timing_stats]
        
        summary = {
            "inference_ms": {
                "mean": np.mean(inference_times),
                "std": np.std(inference_times),
                "min": np.min(inference_times),
                "max": np.max(inference_times),
                "p95": np.percentile(inference_times, 95)
            },
            "detection_ms": {
                "mean": np.mean(detection_times),
                "std": np.std(detection_times),
                "min": np.min(detection_times),
                "max": np.max(detection_times),
                "p95": np.percentile(detection_times, 95)
            },
            "segmentation_ms": {
                "mean": np.mean(segmentation_times),
                "std": np.std(segmentation_times),
                "min": np.min(segmentation_times),
                "max": np.max(segmentation_times),
                "p95": np.percentile(segmentation_times, 95)
            },
            "total_ms": {
                "mean": np.mean(total_times),
                "std": np.std(total_times),
                "min": np.min(total_times),
                "max": np.max(total_times),
                "p95": np.percentile(total_times, 95)
            },
            "detections_per_image": {
                "mean": np.mean(num_detections),
                "std": np.std(num_detections),
                "min": np.min(num_detections),
                "max": np.max(num_detections)
            },
            "total_images": len(timing_stats)
        }
        
        return summary
    
    def _run_coco_evaluation(self, detections: List[dict]) -> Dict:
        """Run COCO evaluation metrics"""
        
        if not detections:
            return {"error": "No detections to evaluate"}
        
        try:
            # Create results in COCO format
            coco_dt = self.coco_gt.loadRes(detections)
            
            # Run bbox evaluation
            coco_eval_bbox = COCOeval(self.coco_gt, coco_dt, 'bbox')
            coco_eval_bbox.evaluate()
            coco_eval_bbox.accumulate()
            coco_eval_bbox.summarize()
            
            bbox_stats = coco_eval_bbox.stats
            
            # Run segmentation evaluation if masks are available
            segm_stats = None
            if any('segmentation' in det for det in detections):
                coco_eval_segm = COCOeval(self.coco_gt, coco_dt, 'segm')
                coco_eval_segm.evaluate()
                coco_eval_segm.accumulate()
                coco_eval_segm.summarize()
                segm_stats = coco_eval_segm.stats
            
            # Package results
            results = {
                "bbox": {
                    "mAP": bbox_stats[0],
                    "mAP_50": bbox_stats[1],
                    "mAP_75": bbox_stats[2],
                    "mAP_small": bbox_stats[3],
                    "mAP_medium": bbox_stats[4],
                    "mAP_large": bbox_stats[5],
                    "mAR_1": bbox_stats[6],
                    "mAR_10": bbox_stats[7],
                    "mAR_100": bbox_stats[8],
                    "mAR_small": bbox_stats[9],
                    "mAR_medium": bbox_stats[10],
                    "mAR_large": bbox_stats[11]
                }
            }
            
            if segm_stats is not None:
                results["segmentation"] = {
                    "mAP": segm_stats[0],
                    "mAP_50": segm_stats[1],
                    "mAP_75": segm_stats[2],
                    "mAP_small": segm_stats[3],
                    "mAP_medium": segm_stats[4],
                    "mAP_large": segm_stats[5],
                    "mAR_1": segm_stats[6],
                    "mAR_10": segm_stats[7],
                    "mAR_100": segm_stats[8],
                    "mAR_small": segm_stats[9],
                    "mAR_medium": segm_stats[10],
                    "mAR_large": segm_stats[11]
                }
            
            return results
            
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}
    
    def run_evaluation(self) -> Dict:
        """Run complete evaluation"""
        
        print("🚀 Starting COCO evaluation...")
        
        # Setup
        self.setup()
        
        # Run evaluation
        results = self.evaluate_subset(self.config.max_images)
        
        # Save results
        if self.config.save_predictions:
            results_file = Path(self.config.results_dir) / f"evaluation_results_{self.config.subset}.json"
            with open(results_file, 'w') as f:
                # Convert numpy types for JSON serialization
                json_results = self._convert_numpy_types(results)
                json.dump(json_results, f, indent=2)
            print(f"💾 Saved evaluation results to {results_file}")
        
        return results
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj


def print_evaluation_summary(results: Dict):
    """Print formatted evaluation summary"""
    
    print("\n" + "="*60)
    print("📊 OVOD COCO Evaluation Summary")
    print("="*60)
    
    # Timing results
    if "timing" in results:
        timing = results["timing"]
        print(f"\n⏱️  Timing Performance:")
        print(f"   Inference (mean): {timing['inference_ms']['mean']:.1f}ms")
        print(f"   Detection (mean): {timing['detection_ms']['mean']:.1f}ms") 
        print(f"   Segmentation (mean): {timing['segmentation_ms']['mean']:.1f}ms")
        print(f"   Total images: {timing['total_images']}")
        print(f"   Avg detections/image: {timing['detections_per_image']['mean']:.1f}")
    
    # mAP results
    if "evaluation" in results and "bbox" in results["evaluation"]:
        bbox = results["evaluation"]["bbox"]
        print(f"\n🎯 Detection mAP:")
        print(f"   mAP@[.50:.95]: {bbox['mAP']:.3f}")
        print(f"   mAP@.50:      {bbox['mAP_50']:.3f}")
        print(f"   mAP@.75:      {bbox['mAP_75']:.3f}")
        print(f"   mAP (small):  {bbox['mAP_small']:.3f}")
        print(f"   mAP (medium): {bbox['mAP_medium']:.3f}")
        print(f"   mAP (large):  {bbox['mAP_large']:.3f}")
    
    # Segmentation results
    if "evaluation" in results and "segmentation" in results["evaluation"]:
        segm = results["evaluation"]["segmentation"]
        print(f"\n🎭 Segmentation mAP:")
        print(f"   mAP@[.50:.95]: {segm['mAP']:.3f}")
        print(f"   mAP@.50:      {segm['mAP_50']:.3f}")
        print(f"   mAP@.75:      {segm['mAP_75']:.3f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="OVOD COCO Evaluation")
    parser.add_argument("--coco-path", default="data/coco", help="Path to COCO dataset")
    parser.add_argument("--subset", default="val2017", choices=["val2017", "test2017"])
    parser.add_argument("--max-images", type=int, default=1000, help="Max images to evaluate")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--image-size", type=int, default=640, help="Input image size")
    parser.add_argument("--results-dir", default="metrics/results", help="Results directory")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="Box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Text confidence threshold")
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="NMS IoU threshold")
    
    args = parser.parse_args()
    
    # Create config
    config = EvalConfig(
        coco_path=args.coco_path,
        subset=args.subset,
        max_images=args.max_images,
        device=args.device,
        image_size=args.image_size,
        results_dir=args.results_dir,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_threshold=args.nms_threshold
    )
    
    # Run evaluation
    evaluator = COCOEvaluator(config)
    results = evaluator.run_evaluation()
    
    # Print summary
    print_evaluation_summary(results)


if __name__ == "__main__":
    main()