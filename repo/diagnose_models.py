#!/usr/bin/env python3
"""
Comprehensive model diagnostics
"""
import os
import sys
from pathlib import Path
import torch

def diagnose_models():
    print("🔬 OVOD Model Diagnostics")
    print("=" * 40)
    
    # 1. Check weights
    print("1️⃣  Checking model weights...")
    weights_dir = Path("weights")
    if weights_dir.exists():
        weights = list(weights_dir.glob("*.pth")) + list(weights_dir.glob("*.pt"))
        for w in weights:
            size_mb = w.stat().st_size / (1024*1024)
            print(f"   ✅ {w.name} ({size_mb:.1f}MB)")
    else:
        print("   ❌ No weights directory found")
    
    # 2. Check GroundingDINO
    print("\n2️⃣  Checking GroundingDINO...")
    gdino_paths = [
        "../GroundingDINO",
        "../../GroundingDINO", 
        "/mnt/e/DEV-PROJECT/01-open-vocabulary-object-finder/GroundingDINO"
    ]
    
    gdino_found = False
    for path in gdino_paths:
        if Path(path).exists():
            print(f"   ✅ Found at: {path}")
            config_path = Path(path) / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
            if config_path.exists():
                print(f"   ✅ Config: {config_path}")
            else:
                print(f"   ❌ Config not found: {config_path}")
            gdino_found = True
            break
    
    if not gdino_found:
        print("   ❌ GroundingDINO not found in any expected location")
    
    # 3. Check SAM2
    print("\n3️⃣  Checking SAM2...")
    sam2_paths = [
        "../sam2",
        "../../sam2",
        "/mnt/e/DEV-PROJECT/01-open-vocabulary-object-finder/sam2"
    ]
    
    sam2_found = False  
    for path in sam2_paths:
        if Path(path).exists():
            print(f"   ✅ Found at: {path}")
            if (Path(path) / "sam2").exists():
                print(f"   ✅ Python package: {path}/sam2")
            sam2_found = True
            break
            
    if not sam2_found:
        print("   ❌ SAM2 not found in any expected location")
    
    # 4. Test imports
    print("\n4️⃣  Testing imports...")
    
    # Test GroundingDINO import
    try:
        sys.path.insert(0, str(Path("../GroundingDINO").resolve()))
        from groundingdino.models import build_model
        print("   ✅ GroundingDINO imports OK")
    except Exception as e:
        print(f"   ❌ GroundingDINO import failed: {e}")
    
    # Test SAM2 import  
    try:
        from sam2.build_sam import build_sam2
        print("   ✅ SAM2 imports OK")
    except Exception as e:
        print(f"   ❌ SAM2 import failed: {e}")
    
    # 5. Test basic pipeline construction
    print("\n5️⃣  Testing pipeline construction...")
    try:
        from ovod.pipeline import OVODPipeline
        
        # Try with explicit paths
        dino_config = None
        dino_checkpoint = None
        sam2_checkpoint = None
        
        # Find weights
        if weights_dir.exists():
            gdino_weights = list(weights_dir.glob("*grounding*"))
            sam2_weights = list(weights_dir.glob("*sam2*"))
            
            if gdino_weights:
                dino_checkpoint = str(gdino_weights[0])
                print(f"   📦 Using GroundingDINO: {dino_checkpoint}")
            
            if sam2_weights:
                sam2_checkpoint = str(sam2_weights[0])
                print(f"   📦 Using SAM2: {sam2_checkpoint}")
        
        # Create pipeline
        pipe = OVODPipeline(
            device="cuda" if torch.cuda.is_available() else "cpu",
            dino_checkpoint=dino_checkpoint,
            sam2_checkpoint=sam2_checkpoint
        )
        print("   ✅ Pipeline created")
        
        # Test load_model
        if hasattr(pipe, 'load_model'):
            pipe.load_model()
        
        # Test simple predict
        dummy_img = torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8).numpy()
        results = pipe.predict(dummy_img, "person", return_masks=False, max_detections=5)
        print(f"   ✅ Prediction test: {len(results.get('boxes', []))} boxes")
        
    except Exception as e:
        print(f"   ❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 40)
    print("📋 Summary:")
    print("   If models aren't loading, run: python demo_setup.py")
    print("   If imports fail, check GroundingDINO and SAM2 installation")

if __name__ == "__main__":
    diagnose_models()