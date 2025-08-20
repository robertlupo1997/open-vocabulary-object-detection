#!/usr/bin/env python3
"""
Quick smoke test for demo fixes
"""
import numpy as np
import sys
from pathlib import Path

def test_imports():
    """Test that all imports work"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from src.visualize import create_detection_visualization
        print("✅ Visualization utils imported")
    except ImportError as e:
        print(f"❌ Visualization import failed: {e}")
        return False
    
    try:
        from ovod.pipeline import OVODPipeline  
        print("✅ OVOD pipeline imported")
    except ImportError as e:
        print(f"❌ Pipeline import failed: {e}")
        return False
        
    return True

def test_visualization():
    """Test visualization function with dummy data"""
    print("\n🎨 Testing visualization...")
    
    try:
        from src.visualize import create_detection_visualization
        
        # Create dummy image and results
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = {
            "boxes": [[100, 100, 200, 200], [300, 150, 450, 300]],
            "labels": ["person", "car"],
            "scores": [0.85, 0.92],
            "masks": []
        }
        
        # Test visualization
        vis_image = create_detection_visualization(image, results, show_masks=False)
        print(f"✅ Visualization created: {vis_image.size}")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_pipeline_loading():
    """Test pipeline loading (without model weights)"""
    print("\n⚙️  Testing pipeline loading...")
    
    try:
        from ovod.pipeline import OVODPipeline
        
        # This will likely fail without weights, but we check the error handling
        try:
            pipeline = OVODPipeline(device="cpu")
            print("✅ Pipeline loaded successfully")
            return True
        except Exception as e:
            if "checkpoint" in str(e) or "weights" in str(e):
                print(f"⚠️  Expected error (missing weights): {e}")
                print("💡 Run 'python demo_setup.py' to download weights")
                return True
            else:
                print(f"❌ Unexpected pipeline error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Pipeline import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 OVOD Demo Smoke Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Visualization", test_visualization),
        ("Pipeline Loading", test_pipeline_loading)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📋 Test Results:")
    
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {name}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Demo should work.")
        print("\nNext steps:")
        print("1. python demo_setup.py  # Download weights")
        print("2. streamlit run demo_app.py  # Run demo")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())