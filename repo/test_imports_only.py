#!/usr/bin/env python3
"""
Minimal import test for OVOD demo
"""
import sys

def test_basic_imports():
    """Test basic Python imports"""
    print("🔍 Testing basic imports...")
    
    try:
        import os
        import sys
        from pathlib import Path
        print("✅ Basic Python modules")
    except ImportError as e:
        print(f"❌ Basic imports failed: {e}")
        return False
    
    return True

def test_repo_structure():
    """Test that repo structure is correct"""
    print("\n📁 Testing repo structure...")
    
    from pathlib import Path
    
    expected_files = [
        "demo_app.py",
        "demo_setup.py", 
        "ovod/pipeline.py",
        "src/visualize.py",
        "src/nms.py",
        "src/prompts.py",
        "src/detector.py",
        "src/segmenter.py"
    ]
    
    missing = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    else:
        print("✅ All expected files present")
        return True

def main():
    """Run minimal tests"""
    print("🚀 OVOD Minimal Test")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_repo_structure
    ]
    
    all_passed = True
    for test_func in tests:
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✅ Basic structure is good!")
        print("\nTo complete setup:")
        print("1. Install dependencies: pip install torch torchvision streamlit numpy pillow opencv-python")
        print("2. Download weights: python demo_setup.py")
        print("3. Run demo: streamlit run demo_app.py")
    else:
        print("❌ Structure issues found")
        
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())