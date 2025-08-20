"""
Visualization utilities for OVOD detection results
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Any
import colorsys

def generate_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for visualization"""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        saturation = 0.8
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

def draw_box_with_label(
    draw: ImageDraw.Draw,
    box: List[float],
    label: str,
    score: float,
    color: Tuple[int, int, int],
    font_size: int = 16
) -> None:
    """Draw bounding box with label"""
    x1, y1, x2, y2 = box
    
    # Draw box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    # Prepare label text
    text = f"{label}: {score:.2f}"
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Draw label background
    label_bg = [x1, y1 - text_height - 4, x1 + text_width + 8, y1]
    draw.rectangle(label_bg, fill=color)
    
    # Draw text
    draw.text((x1 + 4, y1 - text_height - 2), text, fill=(255, 255, 255), font=font)

def apply_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.5
) -> np.ndarray:
    """Apply colored mask overlay to image"""
    # Ensure mask is binary
    mask = (mask > 0.5).astype(np.uint8)
    
    # Create colored overlay
    overlay = np.zeros_like(image)
    overlay[mask == 1] = color
    
    # Blend with original image
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return result

def create_detection_visualization(
    image: np.ndarray,
    results: Dict[str, Any],
    show_masks: bool = True,
    mask_alpha: float = 0.3,
    box_thickness: int = 3,
    font_size: int = 16
) -> Image.Image:
    """
    Create visualization of detection results
    
    Args:
        image: Input image as numpy array (H, W, 3)
        results: Detection results dictionary containing:
            - boxes: List of [x1, y1, x2, y2] coordinates
            - labels: List of object labels
            - scores: List of confidence scores
            - masks: Optional list of segmentation masks
        show_masks: Whether to show segmentation masks
        mask_alpha: Transparency of mask overlays
        box_thickness: Thickness of bounding boxes
        font_size: Size of label text
    
    Returns:
        PIL Image with visualizations
    """
    # Convert to PIL for drawing
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
    else:
        pil_image = image.copy()
    
    # Convert back to numpy for mask operations
    img_array = np.array(pil_image)
    
    boxes = results.get("boxes", [])
    labels = results.get("labels", [])
    scores = results.get("scores", [])
    masks = results.get("masks", [])
    
    if not boxes:
        return pil_image
    
    # Generate colors for each detection
    colors = generate_colors(len(boxes))
    
    # Apply masks if available and requested
    if show_masks and masks:
        for i, mask in enumerate(masks):
            if i < len(colors):
                color = colors[i]
                img_array = apply_mask_overlay(img_array, mask, color, mask_alpha)
    
    # Convert back to PIL for box drawing
    pil_image = Image.fromarray(img_array)
    draw = ImageDraw.Draw(pil_image)
    
    # Draw bounding boxes and labels
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if i < len(colors):
            color = colors[i]
            draw_box_with_label(draw, box, label, score, color, font_size)
    
    return pil_image

def create_grid_visualization(
    images: List[np.ndarray],
    results_list: List[Dict[str, Any]],
    titles: Optional[List[str]] = None,
    show_masks: bool = True,
    grid_cols: int = 2
) -> Image.Image:
    """Create grid visualization of multiple detection results"""
    if not images:
        return Image.new('RGB', (400, 300), color=(128, 128, 128))
    
    # Create individual visualizations
    vis_images = []
    for i, (img, results) in enumerate(zip(images, results_list)):
        vis_img = create_detection_visualization(img, results, show_masks)
        
        # Add title if provided
        if titles and i < len(titles):
            # Add title bar
            title_img = Image.new('RGB', (vis_img.width, 30), color=(50, 50, 50))
            draw = ImageDraw.Draw(title_img)
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            draw.text((10, 8), titles[i], fill=(255, 255, 255), font=font)
            
            # Combine title and image
            combined = Image.new('RGB', (vis_img.width, vis_img.height + 30))
            combined.paste(title_img, (0, 0))
            combined.paste(vis_img, (0, 30))
            vis_img = combined
        
        vis_images.append(vis_img)
    
    # Calculate grid dimensions
    num_images = len(vis_images)
    grid_rows = (num_images + grid_cols - 1) // grid_cols
    
    # Get max dimensions
    max_width = max(img.width for img in vis_images)
    max_height = max(img.height for img in vis_images)
    
    # Create grid
    grid_width = grid_cols * max_width
    grid_height = grid_rows * max_height
    grid_image = Image.new('RGB', (grid_width, grid_height), color=(64, 64, 64))
    
    # Place images in grid
    for i, img in enumerate(vis_images):
        row = i // grid_cols
        col = i % grid_cols
        x = col * max_width
        y = row * max_height
        grid_image.paste(img, (x, y))
    
    return grid_image

def save_detection_results(
    image: np.ndarray,
    results: Dict[str, Any],
    output_path: str,
    show_masks: bool = True
) -> None:
    """Save detection visualization to file"""
    vis_image = create_detection_visualization(image, results, show_masks)
    vis_image.save(output_path)
    print(f"Saved visualization: {output_path}")