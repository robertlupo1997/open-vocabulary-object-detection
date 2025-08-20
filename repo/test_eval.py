#!/usr/bin/env python3
"""
Quick test of eval.py setup
"""

import os
from pathlib import Path

def test_eval_setup():
    """Test evaluation setup"""
    print("🔍 Testing COCO Evaluation Setup")
    print("=" * 40)
    
    # Check data structure
    data_dir = Path("data/coco")
    images_dir = data_dir / "val2017"
    annotations_file = data_dir / "annotations/instances_val2017.json"
    
    print(f"Data directory: {data_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Annotations file: {annotations_file}")
    
    if not images_dir.exists():
        print("❌ Images directory missing")
        return False
        
    if not annotations_file.exists():
        print("❌ Annotations file missing") 
        return False
    
    # Count images
    image_files = list(images_dir.glob("*.jpg"))
    print(f"✅ Found {len(image_files)} images")
    
    # Test imports
    try:
        from pycocotools.coco import COCO
        print("✅ pycocotools import OK")
    except ImportError:
        print("❌ pycocotools not available")
        return False
    
    # Test annotations loading
    try:
        coco = COCO(str(annotations_file))
        img_ids = coco.getImgIds()
        print(f"✅ COCO annotations loaded: {len(img_ids)} images")
    except Exception as e:
        print(f"❌ Failed to load COCO annotations: {e}")
        return False
    
    # Test eval.py exists
    eval_script = Path("eval.py")
    if not eval_script.exists():
        print("❌ eval.py script missing")
        return False
    print("✅ eval.py script ready")
    
    print("\n🎉 Evaluation setup complete!")
    print("\nTo run evaluation:")
    print("python eval.py --max-images 50 --device auto")
    
    return True

if __name__ == "__main__":
    test_eval_setup()