#!/usr/bin/env python3
"""
Test the import fixes
"""

def test_imports():
    """Test that imports work"""
    try:
        from src.segmenter import SAM2Segmenter
        print("✅ SAM2Segmenter import fixed")
    except Exception as e:
        print(f"❌ SAM2Segmenter import failed: {e}")
        return False
    
    try:
        from ovod.pipeline import OVODPipeline
        print("✅ OVODPipeline import works")
    except Exception as e:
        print(f"❌ OVODPipeline import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"❌ PyTorch failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🔧 Testing Import Fixes")
    print("=" * 30)
    
    if test_imports():
        print("\n🎉 All imports working!")
        print("\nNow run: streamlit run demo_app.py")
    else:
        print("\n❌ Some imports still failing")