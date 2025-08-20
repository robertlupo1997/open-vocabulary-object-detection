"""
Demo setup script: Download model weights and prepare environment
"""
import os
import sys
import subprocess
import urllib.request
from pathlib import Path
import zipfile
import tarfile
import argparse

# Model download URLs and checksums
MODEL_CONFIGS = {
    "grounding_dino": {
        "config": "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "checkpoint_url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "checkpoint_path": "weights/groundingdino_swint_ogc.pth",
        "size_mb": 694
    },
    "sam2_small": {
        "config": "sam2_hiera_s.yaml", 
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "checkpoint_path": "weights/sam2_hiera_small.pt",
        "size_mb": 185
    },
    "sam2_base": {
        "config": "sam2_hiera_b+.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt", 
        "checkpoint_path": "weights/sam2_hiera_base_plus.pt",
        "size_mb": 320
    },
    "sam2_large": {
        "config": "sam2_hiera_l.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "checkpoint_path": "weights/sam2_hiera_large.pt", 
        "size_mb": 900
    }
}

def download_file(url: str, filepath: str, desc: str = ""):
    """Download file with progress bar"""
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f"\r{desc}: {percent:.1f}% ({downloaded//1024//1024:.1f}MB/{total_size//1024//1024:.1f}MB)", end="")
    
    try:
        print(f"Downloading {desc}...")
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\n✅ Downloaded: {filepath}")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {desc}: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    
    # Map package names to import names
    required_packages = {
        "torch": "torch",
        "torchvision": "torchvision", 
        "transformers": "transformers",
        "timm": "timm",
        "opencv-python": "cv2",
        "streamlit": "streamlit", 
        "numpy": "numpy",
        "PIL": "PIL"
    }
    
    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    
    dirs = ["weights", "data", "logs", "notebooks", "metrics", "tests"]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}/")

def check_gpu():
    """Check GPU availability"""
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("⚠️  No GPU detected. Models will run on CPU (slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def download_models(models: list = None, force: bool = False):
    """Download specified model weights"""
    
    if models is None:
        models = ["grounding_dino", "sam2_small"]  # Default lightweight setup
    
    total_size = sum(MODEL_CONFIGS[model]["size_mb"] for model in models)
    print(f"📦 Will download {len(models)} models (~{total_size}MB total)")
    
    for model_name in models:
        if model_name not in MODEL_CONFIGS:
            print(f"❌ Unknown model: {model_name}")
            continue
            
        config = MODEL_CONFIGS[model_name]
        checkpoint_path = config["checkpoint_path"]
        
        # Skip if already exists and not forcing
        if os.path.exists(checkpoint_path) and not force:
            print(f"✅ {model_name} already exists: {checkpoint_path}")
            continue
        
        # Download model
        success = download_file(
            config["checkpoint_url"],
            checkpoint_path, 
            f"{model_name} ({config['size_mb']}MB)"
        )
        
        if not success:
            print(f"❌ Failed to download {model_name}")
            return False
    
    return True

def verify_installation():
    """Verify that everything is set up correctly"""
    
    print("\n🔍 Verifying installation...")
    
    # Check model files
    missing_models = []
    for model_name, config in MODEL_CONFIGS.items():
        if not os.path.exists(config["checkpoint_path"]):
            missing_models.append(model_name)
    
    if missing_models:
        print(f"⚠️  Missing models: {missing_models}")
    else:
        print("✅ All model weights found")
    
    # Test imports
    try:
        from ovod.pipeline import OVODPipeline
        print("✅ OVOD pipeline imports successfully")
    except Exception as e:
        print(f"❌ Pipeline import failed: {e}")
        return False
    
    # Test GPU
    gpu_available = check_gpu()
    
    print("\n🎉 Setup verification complete!")
    return True

def create_requirements_file():
    """Create requirements.txt with exact versions"""
    
    requirements = """
# Core ML frameworks
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
timm>=0.9.0

# Computer vision
opencv-python>=4.8.0
pillow>=9.5.0

# Object detection utilities  
supervision>=0.16.0
pycocotools>=2.0.6

# Web demo
streamlit>=1.25.0
matplotlib>=3.7.0

# Utilities
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0
addict>=2.4.0

# Development and testing
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0

# Data handling
pandas>=2.0.0
requests>=2.31.0
""".strip()

    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")

def main():
    parser = argparse.ArgumentParser(description="OVOD Demo Setup")
    parser.add_argument("--models", nargs="+", 
                       choices=list(MODEL_CONFIGS.keys()),
                       default=["grounding_dino", "sam2_small"],
                       help="Models to download")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download existing models")
    parser.add_argument("--skip-deps", action="store_true", 
                       help="Skip dependency check")
    parser.add_argument("--requirements-only", action="store_true",
                       help="Only create requirements.txt")
    
    args = parser.parse_args()
    
    print("🚀 OVOD Demo Setup")
    print("=" * 50)
    
    if args.requirements_only:
        create_requirements_file()
        return
    
    # Check dependencies
    if not args.skip_deps:
        print("📋 Checking dependencies...")
        if not check_dependencies():
            print("\n💡 Install missing packages first, then re-run setup")
            return
        print("✅ All dependencies satisfied")
    
    # Create directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Create requirements file
    create_requirements_file()
    
    # Check GPU
    print("\n🔧 Checking hardware...")
    gpu_available = check_gpu()
    
    # Download models
    print(f"\n📦 Downloading models: {args.models}")
    if not download_models(args.models, args.force):
        print("❌ Model download failed")
        return
    
    # Verify installation
    if not verify_installation():
        print("❌ Verification failed")
        return
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run demo: streamlit run demo_app.py")
    print("2. Test pipeline: python smoke_model_load.py") 
    print("3. Run tests: pytest tests/")
    
    if not gpu_available:
        print("\n💡 For better performance, set up CUDA GPU support")

if __name__ == "__main__":
    main()