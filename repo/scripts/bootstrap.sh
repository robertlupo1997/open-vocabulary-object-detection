#!/bin/bash
# Bootstrap script for Open-Vocabulary Object Detection (OVOD) project
# Sets up complete development environment

set -e

echo "🚀 OVOD Project Bootstrap"
echo "========================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV_NAME="ovod"
PYTHON_VERSION="3.10"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${PROJECT_ROOT}/weights"
DATA_DIR="${PROJECT_ROOT}/data"

echo -e "${BLUE}Project root: ${PROJECT_ROOT}${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Conda not found. Please install Miniconda or Anaconda first.${NC}"
    exit 1
fi

# 1. Create conda environment
echo -e "\n${YELLOW}📦 Creating conda environment '${CONDA_ENV_NAME}'${NC}"
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo -e "${GREEN}✅ Environment '${CONDA_ENV_NAME}' already exists${NC}"
else
    conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y
    echo -e "${GREEN}✅ Created environment '${CONDA_ENV_NAME}'${NC}"
fi

# Activate environment
echo -e "\n${YELLOW}🔧 Activating environment${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

# 2. Install core dependencies
echo -e "\n${YELLOW}📚 Installing core dependencies${NC}"
pip install --upgrade pip

# Core ML packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers timm pillow opencv-python
pip install streamlit plotly pycocotools
pip install numpy pandas matplotlib seaborn
pip install supervision addict yapf

echo -e "${GREEN}✅ Core dependencies installed${NC}"

# 3. Clone and setup GroundingDINO
echo -e "\n${YELLOW}🔍 Setting up Grounding DINO${NC}"
GDINO_DIR="${PROJECT_ROOT}/../GroundingDINO"
if [ ! -d "${GDINO_DIR}" ]; then
    echo "Cloning Grounding DINO..."
    cd "$(dirname "${PROJECT_ROOT}")"
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO
    pip install -e .
    echo -e "${GREEN}✅ Grounding DINO cloned and installed${NC}"
else
    echo -e "${GREEN}✅ Grounding DINO already exists${NC}"
fi

# 4. Clone and setup SAM2
echo -e "\n${YELLOW}🎭 Setting up SAM 2${NC}"
SAM2_DIR="${PROJECT_ROOT}/../sam2"
if [ ! -d "${SAM2_DIR}" ]; then
    echo "Cloning SAM 2..."
    cd "$(dirname "${PROJECT_ROOT}")"
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    pip install -e .
    echo -e "${GREEN}✅ SAM 2 cloned and installed${NC}"
else
    echo -e "${GREEN}✅ SAM 2 already exists${NC}"
fi

# 5. Download model weights
echo -e "\n${YELLOW}⚖️  Downloading model weights${NC}"
cd "${PROJECT_ROOT}"
mkdir -p "${WEIGHTS_DIR}"

# Grounding DINO weights
GDINO_WEIGHTS="${WEIGHTS_DIR}/groundingdino_swint_ogc.pth"
if [ ! -f "${GDINO_WEIGHTS}" ]; then
    echo "Downloading Grounding DINO weights..."
    wget -O "${GDINO_WEIGHTS}" "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    echo -e "${GREEN}✅ Grounding DINO weights downloaded${NC}"
else
    echo -e "${GREEN}✅ Grounding DINO weights already exist${NC}"
fi

# SAM 2 weights (will be auto-downloaded by the library)
echo -e "${GREEN}✅ SAM 2 weights will be auto-downloaded${NC}"

# 6. Setup data directory and download COCO subset
echo -e "\n${YELLOW}📊 Setting up COCO validation data${NC}"
mkdir -p "${DATA_DIR}/coco"

# Download COCO validation annotations
ANN_FILE="${DATA_DIR}/coco/annotations/instances_val2017.json"
if [ ! -f "${ANN_FILE}" ]; then
    echo "Downloading COCO validation annotations..."
    mkdir -p "${DATA_DIR}/coco/annotations"
    wget -O "${DATA_DIR}/coco/annotations/instances_val2017.json" \
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    cd "${DATA_DIR}/coco/annotations"
    unzip -q annotations_trainval2017.zip
    mv annotations/* .
    rm -rf annotations annotations_trainval2017.zip
    echo -e "${GREEN}✅ COCO annotations downloaded${NC}"
else
    echo -e "${GREEN}✅ COCO annotations already exist${NC}"
fi

# Download subset of COCO validation images (first 1000)
IMG_DIR="${DATA_DIR}/coco/val2017"
if [ ! -d "${IMG_DIR}" ] || [ $(ls "${IMG_DIR}"/*.jpg 2>/dev/null | wc -l) -lt 100 ]; then
    echo "Downloading COCO validation images subset (first 1000)..."
    mkdir -p "${IMG_DIR}"
    
    # Download images list and get first 1000
    wget -q -O - "http://images.cocodataset.org/zips/val2017.zip" | python3 -c "
import zipfile
import sys
from pathlib import Path
import io

# Read zip from stdin
zip_data = sys.stdin.buffer.read()
with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
    # Get all jpg files, take first 1000
    jpg_files = [f for f in zf.namelist() if f.endswith('.jpg')][:1000]
    
    for jpg_file in jpg_files:
        # Extract to val2017 directory
        with zf.open(jpg_file) as source:
            target_path = Path('${IMG_DIR}') / Path(jpg_file).name
            with open(target_path, 'wb') as target:
                target.write(source.read())
    
    print(f'Extracted {len(jpg_files)} images')
"
    echo -e "${GREEN}✅ COCO validation images downloaded${NC}"
else
    echo -e "${GREEN}✅ COCO validation images already exist${NC}"
fi

# 7. Create symlink for data access from repo
echo -e "\n${YELLOW}🔗 Creating data symlink${NC}"
cd "${PROJECT_ROOT}"
if [ ! -L "data/coco" ]; then
    mkdir -p data
    ln -sf "../data/coco" "data/coco"
    echo -e "${GREEN}✅ Data symlink created${NC}"
else
    echo -e "${GREEN}✅ Data symlink already exists${NC}"
fi

# 8. Test installation
echo -e "\n${YELLOW}🧪 Testing installation${NC}"
python -c "
import torch
import cv2
import transformers
import streamlit
from pycocotools.coco import COCO
print('✅ All core packages imported successfully')

# Test CUDA
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  CUDA not available, using CPU')

# Test paths
from pathlib import Path
weights_dir = Path('weights')
data_dir = Path('data/coco')

if weights_dir.exists():
    print(f'✅ Weights directory: {len(list(weights_dir.glob(\"*.pth\")))} files')
if data_dir.exists():
    val_imgs = len(list(data_dir.glob('val2017/*.jpg')))
    print(f'✅ COCO data: {val_imgs} validation images')
"

echo -e "\n${GREEN}🎉 Bootstrap completed successfully!${NC}"
echo -e "\n${BLUE}Next steps:${NC}"
echo "1. Activate environment: conda activate ${CONDA_ENV_NAME}"
echo "2. Run demo: streamlit run demo_app.py"
echo "3. Run evaluation: python eval.py --max-images 50"
echo "4. Use Makefile: make demo, make eval-50, etc."

echo -e "\n${YELLOW}Environment summary:${NC}"
echo "- Conda environment: ${CONDA_ENV_NAME}"
echo "- Python: ${PYTHON_VERSION}"
echo "- Grounding DINO: ${GDINO_DIR}"
echo "- SAM 2: ${SAM2_DIR}"
echo "- Weights: ${WEIGHTS_DIR}"
echo "- Data: ${DATA_DIR}"