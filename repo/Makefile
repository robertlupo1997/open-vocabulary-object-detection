# Open-Vocabulary Object Detection (OVOD) Makefile
# Provides common development and evaluation targets

# Use bash as shell for conda commands
SHELL := /bin/bash

.PHONY: help env setup data link-data demo eval-50 eval-200 eval-full clean lint test

# Default target
help:
	@echo "🚀 OVOD Project Makefile"
	@echo "======================="
	@echo ""
	@echo "Available targets:"
	@echo "  help      - Show this help message"
	@echo "  env       - Create conda environment and install dependencies"
	@echo "  setup     - Complete project setup (env + weights + data)"
	@echo "  data      - Download COCO validation data"
	@echo "  link-data - Create symlink to data directory"
	@echo "  demo      - Launch Streamlit demo app"
	@echo "  eval-50   - Run COCO evaluation on 50 images (common prompt)"
	@echo "  eval-200  - Run COCO evaluation on 200 images (common prompt)"
	@echo "  eval-full - Run full COCO evaluation (5000 images)"
	@echo "  eval-person      - Run evaluation with person-only prompt (200 imgs)"
	@echo "  eval-full-prompt - Run evaluation with full 80-class prompt (200 imgs)"
	@echo "  clean     - Clean up cache and temporary files"
	@echo "  lint      - Run code linting and formatting"
	@echo "  test      - Run basic functionality tests"
	@echo ""
	@echo "Environment:"
	@echo "  CONDA_ENV - Conda environment name (default: ovod)"
	@echo "  DEVICE    - Device to use (default: auto)"
	@echo ""

# Environment variables
CONDA_ENV ?= ovod
DEVICE ?= auto
PROJECT_ROOT := $(shell pwd)

# Check if conda environment exists
ENV_EXISTS := $(shell conda env list | grep -q "^$(CONDA_ENV) " && echo "yes" || echo "no")

# Conda activation command - use explicit path
CONDA_ACTIVATE = source ~/miniconda3/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)

# Environment setup
env:
	@echo "📦 Setting up conda environment '$(CONDA_ENV)'"
	@if [ "$(ENV_EXISTS)" = "no" ]; then \
		conda create -n $(CONDA_ENV) python=3.10 -y; \
		echo "✅ Created environment '$(CONDA_ENV)'"; \
	else \
		echo "✅ Environment '$(CONDA_ENV)' already exists"; \
	fi
	@$(CONDA_ACTIVATE) && pip install --upgrade pip
	@$(CONDA_ACTIVATE) && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
	@$(CONDA_ACTIVATE) && pip install transformers timm pillow opencv-python streamlit plotly pycocotools
	@$(CONDA_ACTIVATE) && pip install numpy pandas matplotlib seaborn supervision addict yapf
	@echo "✅ Dependencies installed in '$(CONDA_ENV)'"

# Complete setup using bootstrap script
setup:
	@echo "🚀 Running complete project setup"
	@chmod +x scripts/bootstrap.sh
	@./scripts/bootstrap.sh

# Download COCO data
data:
	@echo "📊 Downloading COCO validation data"
	@mkdir -p data/coco/annotations data/coco/val2017
	@if [ ! -f "data/coco/annotations/instances_val2017.json" ]; then \
		echo "Downloading COCO annotations..."; \
		wget -O data/coco/annotations/annotations.zip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"; \
		cd data/coco/annotations && unzip -q annotations.zip && mv annotations/* . && rm -rf annotations annotations.zip; \
	fi
	@if [ $$(ls data/coco/val2017/*.jpg 2>/dev/null | wc -l) -lt 100 ]; then \
		echo "Downloading COCO validation images (first 1000)..."; \
		python3 -c "import urllib.request, zipfile, io; \
		zip_data = urllib.request.urlopen('http://images.cocodataset.org/zips/val2017.zip').read(); \
		with zipfile.ZipFile(io.BytesIO(zip_data)) as zf: \
			jpg_files = [f for f in zf.namelist() if f.endswith('.jpg')][:1000]; \
			[zf.extract(f, 'data/coco') for f in jpg_files]; \
		print(f'Extracted {len(jpg_files)} images')"; \
	fi
	@echo "✅ COCO data ready"

# Create data symlink
link-data:
	@echo "🔗 Creating data symlink"
	@[ -d ../data/coco ] || { echo "⚠️  Put COCO dataset at ./data/coco first (from project root)"; exit 1; }
	@mkdir -p data
	@if [ ! -L "data/coco" ]; then \
		ln -sf "../../data/coco" "data/coco"; \
		echo "✅ Data symlink created"; \
	else \
		echo "✅ Data symlink already exists"; \
	fi

# Launch Streamlit demo
demo:
	@echo "🎭 Launching Streamlit demo"
	@$(CONDA_ACTIVATE) && streamlit run demo_app.py --server.port 8501 --server.headless true

# Evaluation targets
eval-50:
	@echo "📊 Running COCO evaluation on 50 images (common prompt)"
	@$(CONDA_ACTIVATE) && python eval.py --max-images 50 --device $(DEVICE) --prompt common

eval-200:
	@echo "📊 Running COCO evaluation on 200 images (common prompt)"
	@$(CONDA_ACTIVATE) && python eval.py --max-images 200 --device $(DEVICE) --prompt common

eval-full:
	@echo "📊 Running full COCO evaluation (5000 images)"
	@$(CONDA_ACTIVATE) && python eval.py --max-images 5000 --device $(DEVICE) --prompt common

# Evaluation with different prompt strategies  
eval-person:
	@echo "📊 Running evaluation with person-only prompt"
	@$(CONDA_ACTIVATE) && python eval.py --max-images 200 --prompt person --box-thr 0.2 --text-thr 0.15 --device $(DEVICE)

eval-full-prompt:
	@echo "📊 Running evaluation with full 80-class COCO prompt"
	@$(CONDA_ACTIVATE) && python eval.py --max-images 200 --prompt full --box-thr 0.25 --text-thr 0.2 --device $(DEVICE)

# Test basic functionality
test:
	@echo "🧪 Running basic functionality tests"
	@$(CONDA_ACTIVATE) && python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
	@$(CONDA_ACTIVATE) && python -c "import streamlit; print('✅ Streamlit available')"
	@$(CONDA_ACTIVATE) && python -c "from pycocotools.coco import COCO; print('✅ COCO API available')"
	@$(CONDA_ACTIVATE) && python -c "from ovod.pipeline import OVODPipeline; print('✅ OVOD pipeline importable')"
	@if [ -d "data/coco/val2017" ]; then echo "✅ COCO data available"; else echo "⚠️  COCO data missing"; fi
	@if [ -d "weights" ]; then echo "✅ Weights directory exists"; else echo "⚠️  Weights directory missing"; fi

# Code linting and formatting  
lint:
	@echo "🔧 Running code linting"
	@$(CONDA_ACTIVATE) && python -m py_compile *.py src/*.py ovod/*.py || echo "⚠️  Some Python files have syntax errors"
	@echo "✅ Basic linting complete"

# Clean up
clean:
	@echo "🧹 Cleaning up cache and temporary files"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Development helpers
dev-info:
	@echo "📋 Development Information"
	@echo "========================"
	@echo "Project: Open-Vocabulary Object Detection"
	@echo "Environment: $(CONDA_ENV)"
	@echo "Device: $(DEVICE)"
	@echo "Project root: $(PROJECT_ROOT)"
	@echo ""
	@echo "Key files:"
	@echo "  demo_app.py    - Streamlit web interface"
	@echo "  eval.py        - COCO evaluation script"
	@echo "  ovod/pipeline.py - Main detection pipeline"
	@echo "  src/detector.py  - Grounding DINO wrapper"
	@echo "  src/segmenter.py - SAM 2 wrapper"
	@echo ""
	@echo "Data:"
	@if [ -d "data/coco/val2017" ]; then \
		echo "  COCO images: $$(ls data/coco/val2017/*.jpg | wc -l) files"; \
	else \
		echo "  COCO images: Not downloaded"; \
	fi
	@if [ -f "data/coco/annotations/instances_val2017.json" ]; then \
		echo "  COCO annotations: Available"; \
	else \
		echo "  COCO annotations: Not downloaded"; \
	fi
	@echo ""
	@echo "Models:"
	@if [ -f "weights/groundingdino_swint_ogc.pth" ]; then \
		echo "  Grounding DINO: Available"; \
	else \
		echo "  Grounding DINO: Not downloaded"; \
	fi
	@echo "  SAM 2: Auto-downloaded on first use"

# Quick start sequence
quickstart: setup link-data
	@echo ""
	@echo "🎉 Quick start complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. conda activate $(CONDA_ENV)"
	@echo "2. make demo        # Launch web interface"
	@echo "3. make eval-50     # Run evaluation"
	@echo ""