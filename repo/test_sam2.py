#!/usr/bin/env python3
"""
Test SAM2 installation and pipeline
"""

def test_sam2_import():
    """Test SAM2 import"""
    try:
        from src.segmenter import SAM2Segmenter, SAM2_AVAILABLE
        print(f"SAM2_AVAILABLE: {SAM2_AVAILABLE}")
        
        if SAM2_AVAILABLE:
            print("✅ SAM2 imports successfully")
            return True
        else:
            print("❌ SAM2 not available")
            return False
    except Exception as e:
        print(f"❌ SAM2 import failed: {e}")
        return False

def test_pipeline():
    """Test full pipeline"""
    try:
        import numpy as np
        from ovod.pipeline import OVODPipeline
        
        print("Creating pipeline...")
        pipeline = OVODPipeline(device="cuda")
        print("✅ Pipeline created successfully")
        
        # Test with dummy image
        print("Testing with dummy image...")
        img = (np.random.rand(480, 640, 3) * 255).astype('uint8')
        results = pipeline.predict(img, "person", return_masks=False, max_detections=5)
        
        print(f"✅ Pipeline prediction completed")
        print(f"   Results keys: {sorted(results.keys())}")
        print(f"   Boxes found: {len(results.get('boxes', []))}")
        print(f"   Timings: {results.get('timings', {})}")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing SAM2 and Pipeline")
    print("=" * 40)
    
    # Test 1: SAM2 import
    sam2_ok = test_sam2_import()
    
    print("\n" + "-" * 40)
    
    # Test 2: Full pipeline  
    if sam2_ok:
        pipeline_ok = test_pipeline()
    else:
        print("⚠️ Skipping pipeline test (SAM2 not available)")
        pipeline_ok = False
    
    print("\n" + "=" * 40)
    print("📋 Test Summary:")
    print(f"   SAM2 Import: {'✅ PASS' if sam2_ok else '❌ FAIL'}")
    print(f"   Pipeline: {'✅ PASS' if pipeline_ok else '❌ FAIL'}")
    
    if sam2_ok and pipeline_ok:
        print("\n🎉 Everything working! Ready for demo.")
    else:
        print("\n⚠️ Issues detected. Check above for details.")