#!/bin/bash
# Simple environment setup script for OVOD project

set -e

echo "🚀 OVOD Environment Setup"
echo "========================"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh || true
if conda env list | grep -q "^ovod "; then
    echo "✅ Found conda environment 'ovod'"
    conda activate ovod
else
    echo "❌ Conda environment 'ovod' not found. Please create it first:"
    echo "  conda create -n ovod python=3.10 -y"
    echo "  conda activate ovod"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

# Test basic functionality
echo "🧪 Testing installation..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
python -c "import cv2; print('✅ OpenCV available')"
python -c "from pycocotools.coco import COCO; print('✅ COCO API available')"

# Test CUDA
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "✅ CUDA available"
else
    echo "⚠️  CUDA not available, using CPU"
fi

# Check data symlink
if [ -L "data/coco" ] && [ -d "data/coco/val2017" ]; then
    echo "✅ COCO data symlink working"
else
    echo "⚠️  COCO data symlink issue"
fi

# Check weights
if [ -f "weights/groundingdino_swint_ogc.pth" ]; then
    echo "✅ Grounding DINO weights available"
else
    echo "⚠️  Grounding DINO weights missing"
fi

echo ""
echo "✅ Environment ready!"
echo "Next steps:"
echo "  make test    # Run basic tests"
echo "  make demo    # Launch Streamlit demo" 
echo "  make eval-50 # Run evaluation on 50 images"