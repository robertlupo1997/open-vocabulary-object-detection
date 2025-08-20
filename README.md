# Open-Vocabulary Object Detection (OVOD) 

[![CI](https://github.com/robertlupo1997/open-vocabulary-object-detection/actions/workflows/smoke.yml/badge.svg?branch=main)](https://github.com/robertlupo1997/open-vocabulary-object-detection/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)

**Production-ready OVOD system combining Grounding DINO + SAM 2 for text-conditioned object detection and segmentation.**

## ✨ Features

- 🎯 **Text-to-Detection**: Query images with natural language prompts  
- 🔍 **Auto Box Format**: Handles normalized cxcywh, pixel xyxy, normalized xyxy
- ⚡ **Optimized Performance**: ~265ms/image on RTX 3070
- 📊 **COCO Evaluation**: Multiple prompt strategies with micro-metrics
- 🖥️ **Streamlit Demo**: Interactive web interface
- 🔧 **Production Tools**: Makefile targets, environment locking

## 🚀 Quickstart

```bash
# 0) Environment setup
conda env create -f env-ovod.yml
conda activate ovod

# Install GroundingDINO (required dependency)
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# Alternative: manual setup
# conda create -n ovod python=3.10 -y && conda activate ovod
# pip install -r requirements.lock.txt
# pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# 1) Data preparation  
make link-data          # expects data/coco at project root; creates repo/data/coco -> ../../data/coco

# 2) Run demo
cd repo && export PYTHONPATH=$PWD
streamlit run demo_app.py

<!-- Live demo (add your Streamlit URL here when deployed)
[🚀 Live Demo](https://your-demo-url.example)
-->

# 3) Evaluation (prompt strategies)
make eval-50                     # common classes (50 images)
make eval-person                 # person-only (200 images)  
make eval-full-prompt            # 80 COCO classes (200 images)

# Custom evaluation
python eval.py --max-images 200 --prompt common --box-thr 0.25 --text-thr 0.20 --nms 0.5
```

## 🎯 Prompt Strategies

| Strategy | Classes | Best For | Recommended Thresholds |
|----------|---------|----------|------------------------|
| `person` | 1 | Person detection | `--box-thr 0.2 --text-thr 0.15` |
| `common` | 18 | General use | `--box-thr 0.25 --text-thr 0.20` |
| `full` | 80 | Full COCO eval | `--box-thr 0.25 --text-thr 0.20` |

## 🏗️ Architecture

- **Grounding DINO**: Text-conditioned object detection
- **SAM 2**: High-quality segmentation masks
- **Auto Format Detection**: Handles varying coordinate formats
- **Smart Aliasing**: Maps "car bike" → "bicycle", "motorbike" → "motorcycle"

## ⚡ Performance

- **Latency**: 265-490 ms/image (RTX 3070)
- **Detection Quality**: Auto-tuned thresholds per prompt strategy  
- **COCO mAP**: Non-zero evaluation with proper coordinate handling

## 📁 Repository Structure

```
repo/
├── demo_app.py         # Streamlit interface
├── eval.py             # COCO evaluation with multiple strategies  
├── Makefile           # Production targets (eval-50, demo, etc.)
├── ovod/
│   └── pipeline.py     # Main OVOD pipeline
└── src/
    ├── detector.py     # Grounding DINO wrapper
    ├── segmenter.py    # SAM 2 wrapper
    └── visualize.py    # Detection visualization

env-ovod.yml           # Locked conda environment
requirements.lock.txt  # Locked pip requirements
data/coco/            # COCO validation dataset
```

## 🛠️ Development

```bash
# Environment
export PYTHONPATH=$(pwd)/repo

# Testing
make test                        # Basic functionality tests
make lint                        # Code linting

# Evaluation sweep
python eval.py --prompt person --box-thr 0.15 --text-thr 0.10  # High recall
python eval.py --prompt common --box-thr 0.30 --text-thr 0.25  # High precision
```

## 📝 Notes

- **Box Formats**: Detector outputs vary. `eval.py` auto-detects and converts to COCO `xywh` pixels.
- **SAM-2**: Repo submodule or editable install supported; paths auto-discovered.
- **Data**: Expects COCO val2017 dataset at project root `data/coco/`
- **CUDA vs CPU**: Performance metrics are for CUDA. CI uses CPU-only PyTorch for compatibility.
- **Dependencies**: GroundingDINO installed from source for latest features.

## 🔧 Troubleshooting

- **SAM-2 not found**: Ensure editable install or local `sam2/` path exists
- **0.000 mAP**: Now auto-handled via coordinate format detection + aliasing  
- **Streamlit warnings**: Fixed with `use_container_width=True`

## ✅ Status

- [x] Streamlit demo with prompt search
- [x] COCO evaluation pipeline
- [x] Auto box format detection
- [x] Production Makefile targets
- [x] Environment locking
- [x] Performance benchmarks (265ms/img RTX 3070)

## 📜 Third-Party Acknowledgments

This project builds upon excellent work from:

- **[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)** (Apache-2.0) - Text-conditioned object detection
- **[SAM 2](https://github.com/facebookresearch/segment-anything-2)** (Apache-2.0) - Segment Anything Model 2
- **[COCO Dataset](https://cocodataset.org/)** (CC BY 4.0) - Evaluation benchmarks

See [NOTICE](NOTICE) for complete attribution details.

---

🤖 **Generated with [Claude Code](https://claude.ai/code)**