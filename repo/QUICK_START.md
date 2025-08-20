# OVOD Quick Start Guide

## Status ✅
All critical demo fixes are complete! The code should now run without the import/function errors.

## Fixed Issues:
- ✅ Added missing imports (`io`, `time`, `torch`)
- ✅ Implemented `load_pipeline()` function with proper error handling
- ✅ Fixed `st.expander()` usage to use context managers
- ✅ Created complete `src/visualize.py` with proper visualization functions
- ✅ Removed duplicate function definitions

## Windows RTX 3070 Setup Commands

### 1. Environment Setup
```bash
# Open Anaconda Prompt as Administrator
conda create -n ovod python=3.10 -y
conda activate ovod
```

### 2. Install PyTorch with CUDA 12.1
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

### 3. Install Dependencies
```bash
pip install streamlit numpy pillow opencv-python matplotlib transformers timm einops
```

### 4. Navigate and Setup
```bash
cd /path/to/repo  # e.g. C:\ml\ovod\repo
python demo_setup.py --models grounding_dino sam2_small
```

### 5. Test Installation
```bash
python test_imports_only.py  # Should show "✅ Basic structure is good!"
```

### 6. Run Demo
```bash
streamlit run demo_app.py
```

## Expected Results:
- Demo loads without import errors
- GPU detection works (shows RTX 3070)
- You can upload images and enter prompts like "person, car, dog"
- Pipeline will fail gracefully if weights aren't downloaded yet

## If Something Fails:

**"No module named X"**: Install missing packages with pip
**"Pipeline failed to load"**: Run `python demo_setup.py` to download weights
**CUDA errors**: Try CPU mode in sidebar or check CUDA installation

## Performance Targets (RTX 3070):
- ~100-150ms per image at 640px
- ~4GB VRAM usage (plenty of headroom on 8GB card)

## Next Steps After Demo Works:
1. Download COCO val2017 for evaluation
2. Run benchmark: `python metrics/benchmark.py`
3. Run COCO eval: `python run_eval.py --quick`