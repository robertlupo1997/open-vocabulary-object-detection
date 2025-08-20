#!/usr/bin/env python3
"""
Minimal COCO val2017 evaluation for the OVOD pipeline.
- Measures average per-image latency (ms)
- Computes COCO mAP (bbox) with pycocotools

Usage:
  python eval.py --data-dir data/coco --max-images 200 --device auto --box-thr 0.35 --text-thr 0.25 --nms 0.50
"""
import os, json, time, argparse
from pathlib import Path
import numpy as np

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ovod.pipeline import OVODPipeline

# COCO 80 class names in canonical form
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

# Optional simple aliases (model → coco)
ALIASES = {
    "tv monitor": "tv",
    "tvmonitor": "tv",
    "cellphone": "cell phone",
    "aeroplane": "airplane",
    "trafficlight": "traffic light",
    "pottedplant": "potted plant",
    "diningtable": "dining table",
    "wineglass": "wine glass",
}

# Label aliasing to handle common variations
ALIASES = {
    "airplane": "aeroplane",
    "couch": "sofa", 
    "cell phone": "cellphone",
    "dining table": "diningtable", 
    "tv": "tv monitor",
    "motorbike": "motorcycle",
    "car bike": "bicycle",  # Handle compound detections
    "bike": "bicycle",
}

def norm_label(s: str) -> str:
    """Normalize label strings to match COCO categories"""
    s = s.replace("_", " ").replace("-", " ")
    s = s.strip().lower()
    s = ALIASES.get(s, s)
    return s

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def to_coco_xywh(box, img_w, img_h):
    """
    Convert box to COCO xywh format with auto-detection of input format.
    Handles: normalized cxcywh, pixel xyxy, normalized xyxy
    """
    b = [float(x) for x in box]
    
    # Heuristic: if values look like centers/normalized (<=1) and w/h <=1 → treat as cxcywh
    is_normalized = all(0.0 <= v <= 1.0 for v in b)
    looks_cxcywh = is_normalized and b[2] <= 1.0 and b[3] <= 1.0
    looks_xyxy = (b[2] > b[0]) and (b[3] > b[1]) and (b[2] > 1.0 or b[3] > 1.0)

    if looks_cxcywh:
        # Normalized center format: cx, cy, w, h
        cx, cy, bw, bh = b
        w = bw * img_w
        h = bh * img_h
        x = (cx * img_w) - w / 2.0
        y = (cy * img_h) - h / 2.0
        x = max(0.0, min(x, img_w))
        y = max(0.0, min(y, img_h))
        w = max(0.0, min(w, img_w - x))
        h = max(0.0, min(h, img_h - y))
        return [x, y, w, h]

    if looks_xyxy:
        # Pixel xyxy format
        x1, y1, x2, y2 = b
        x = max(0.0, min(x1, img_w))
        y = max(0.0, min(y1, img_h))
        w = max(0.0, min(x2 - x1, img_w - x))
        h = max(0.0, min(y2 - y1, img_h - y))
        return [x, y, w, h]

    # Fallback: treat as normalized xyxy
    x1, y1, x2, y2 = b
    x1, y1, x2, y2 = x1*img_w, y1*img_h, x2*img_w, y2*img_h
    x = max(0.0, min(x1, img_w))
    y = max(0.0, min(y1, img_h))
    w = max(0.0, min(x2 - x1, img_w - x))
    h = max(0.0, min(y2 - y1, img_h - y))
    return [x, y, w, h]

def find_coco_data(provided_path: str):
    """Find COCO data with fallback search"""
    candidates = [
        provided_path,
        "data/coco", 
        "../data/coco",
        "../../data/coco"
    ]
    
    for path in candidates:
        data_dir = Path(path)
        img_dir = data_dir / "val2017" 
        ann_path = data_dir / "annotations/instances_val2017.json"
        
        if img_dir.is_dir() and ann_path.is_file():
            return data_dir
    
    print(f"\n❌ Missing COCO data. Tried:")
    for path in candidates:
        print(f"   - {path}")
    print(f"\nEither pass --data-dir /abs/path/to/data/coco")
    print(f"or run: (from repo/)")
    print(f"  mkdir -p data && ln -s ../data/coco data/coco")
    exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/coco", help="root with val2017/ and annotations/")
    ap.add_argument("--ann", type=str, default="annotations/instances_val2017.json")
    ap.add_argument("--split", type=str, default="val2017")
    ap.add_argument("--max-images", type=int, default=200, help="limit for quick run; set 5000 for full val")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--box-thr", type=float, default=0.35)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=0.50)
    ap.add_argument("--save-json", type=str, default="tmp/coco_ovod_preds.json")
    ap.add_argument("--prompt", type=str, default="common",
                    choices=["common", "full", "person"],
                    help="Prompt strategy: common (30 classes), full (all 80 COCO), person (single class)")
    args = ap.parse_args()

    # Find COCO data with fallbacks
    data_dir = find_coco_data(args.data_dir)
    img_dir = data_dir / args.split
    ann_path = data_dir / args.ann
    
    print(f"✅ Using COCO data: {data_dir}")
    print(f"   Images: {len(list(img_dir.glob('*.jpg')))} files")
    print(f"   Annotations: {ann_path}")
    

    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cuda" if args.device == "cuda" else "cpu")
    )

    # Build pipeline (cached in the app, but here we init once)
    pipe = OVODPipeline(device=device)
    if hasattr(pipe, "load_model"):
        pipe.load_model()
    
    # Set thresholds
    pipe.box_threshold = args.box_thr
    pipe.text_threshold = args.text_thr
    pipe.nms_threshold = args.nms

    # Build prompt based on strategy
    if args.prompt == "person":
        prompt = "person"
    elif args.prompt == "full":
        prompt = ", ".join(COCO_CLASSES)
    else:  # common
        common_classes = [
            "person", "car", "truck", "bus", "motorcycle", "bicycle", "dog", "cat",
            "chair", "couch", "bed", "dining table", "tv", "laptop", "cell phone", "bottle", "cup", "bowl"
        ]
        prompt = ", ".join(common_classes)
    
    print(f"🎯 Using {args.prompt} prompt ({len(prompt.split(', '))} classes): '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
    print(f"🎯 Thresholds: box={args.box_thr}, text={args.text_thr}, nms={args.nms}")

    coco = COCO(str(ann_path))
    img_ids = coco.getImgIds()
    if args.max_images > 0:
        img_ids = img_ids[:args.max_images]

    cat_name_to_id = {coco.loadCats([cid])[0]["name"]: cid for cid in coco.getCatIds()}
    # Normalize keys
    cat_name_to_id = {norm_label(k): v for k, v in cat_name_to_id.items()}

    results = []
    times = []

    for i, img_id in enumerate(img_ids, 1):
        img_info = coco.loadImgs([img_id])[0]
        img_path = img_dir / img_info["file_name"]
        img = _imread_rgb(img_path)

        # Inference timing (sync GPU)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        pred = pipe.predict(
            img,
            prompt,
            return_masks=False,
            max_detections=100,
        )

        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)

        boxes = pred.get("boxes", [])
        labels = pred.get("labels", [])
        scores = pred.get("scores", [])

        # Get image dimensions for coordinate scaling
        h, w = img.shape[:2]
        
        for b, l, s in zip(boxes, labels, scores):
            name = norm_label(str(l))
            if name not in cat_name_to_id:
                continue
            cat_id = cat_name_to_id[name]
            
            # Auto-detect box format and convert to COCO pixel xywh
            bbox_xywh = to_coco_xywh(b, w, h)
            
            results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox_xywh,
                "score": float(s),
            })

        if i % 50 == 0:
            avg = sum(times)/len(times)
            print(f"[{i}/{len(img_ids)}] avg latency: {avg:.1f} ms, detections so far: {len(results)}")

    # Save predictions
    out_json = Path(args.save_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f)

    # Evaluate with COCOeval
    coco_dt = coco.loadRes(str(out_json)) if len(results) else None
    if coco_dt is None:
        print("No detections to evaluate.")
        return
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mean_ms = sum(times)/len(times) if times else 0.0
    print(f"\nLatency: mean {mean_ms:.2f} ms/image over {len(times)} images")
    
    # Micro-metrics summary for quick verification
    print(f"\n📊 Summary:")
    print(f"   Detections kept: {len(results)} | Images processed: {len(img_ids)}")
    if results:
        from collections import Counter
        cat_counts = Counter(r["category_id"] for r in results)
        cat_names = {v: k for k, v in cat_name_to_id.items()}
        top_cats = [(cat_names.get(cat_id, f"id_{cat_id}"), count) for cat_id, count in cat_counts.most_common(5)]
        print(f"   Top categories: {top_cats}")
        avg_score = sum(r["score"] for r in results) / len(results)
        print(f"   Average confidence: {avg_score:.3f}")

def _imread_rgb(path: Path):
    # Use OpenCV if available (faster), otherwise PIL
    try:
        import cv2
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2 failed to read {path}")
        rgb = bgr[:, :, ::-1]
        return rgb
    except Exception:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))

if __name__ == "__main__":
    main()