import sys, torch
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda, "CUDA avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
try:
    from groundingdino.models import build_model  # noqa: F401
    print("✅ GroundingDINO import OK")
except Exception as e:
    print("❌ GroundingDINO import failed:", repr(e)); sys.exit(1)
try:
    from sam2.build_sam import build_sam2  # noqa: F401
    print("✅ SAM2 import OK")
except Exception as e:
    print("❌ SAM2 import failed:", repr(e)); sys.exit(1)
print("✅ Smoke test passed")