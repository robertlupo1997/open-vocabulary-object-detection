# notebooks/utils_ovod_demo.py
"""
Utility functions for OVOD demo notebook:
- timing
- simple synthetic image
- drawing boxes
- system probe
- quiet context manager
"""

import os
import sys
import time
import base64
import io
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Timer:
    """Simple context manager for timing code blocks"""
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        print(f"⏱️ {self.description}: {self.elapsed:.2f}s")


def create_demo_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a simple demo image with geometric shapes"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    # red rectangle
    img[200:280, 100:200] = [220, 50, 50]
    # blue circle
    cy, cx, r = 240, 400, 40
    Y, X = np.ogrid[:height, :width]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    img[mask] = [50, 50, 220]
    # green triangle
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.polygon([(500, 350), (450, 450), (550, 450)], fill=(50, 200, 50))
    return np.array(pil_img)


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: List[str],
    scores: np.ndarray,
    confidence_threshold: float = 0.3,
) -> Image.Image:
    """Draw bounding boxes and labels on image"""
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    colors = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 160, 0), (255, 0, 255)]
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score < confidence_threshold:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label}: {score:.2f}"
        try:
            font = ImageFont.load_default()
            bbox = draw.textbbox((x1, max(0, y1 - 18)), text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, max(0, y1 - 18)), text, fill="white", font=font)
        except Exception:
            draw.text((x1, max(0, y1 - 18)), text, fill=color)
    return img_pil


def get_system_info() -> Dict[str, Any]:
    """Probe minimal system info for the notebook header"""
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
    }
    try:
        import torch
        info.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count())
                if torch.cuda.is_available()
                else 0,
            }
        )
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info.update(
            {"torch_version": None, "cuda_available": False, "cuda_device_count": 0}
        )
    return info


class suppress_output:
    """Context manager to silence stdout/stderr"""
    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._null = open(os.devnull, "w")
        sys.stdout = self._null
        sys.stderr = self._null

    def __exit__(self, *exc):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._null.close()


def format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"