from pathlib import Path

# Paths (assumes this file is at repo/ovod/config.py)
REPO_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = REPO_DIR.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"

for d in (WEIGHTS_DIR, DATA_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    return torch.device("cuda:0")

# set DTYPE lazily to avoid importing torch at module import
DTYPE = None  # set to torch.float16 at runtime if needed

# Official checkpoints
GDINO_FILENAME = "groundingdino_swint_ogc.pth"
# GH v0.1.0 release (works), keep HF fallback in demo_setup.py
GDINO_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0/groundingdino_swint_ogc.pth"

SAM2_FILENAME = "sam2_hiera_small.pt"
# Newer public file; we'll also try an older path as fallback
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
