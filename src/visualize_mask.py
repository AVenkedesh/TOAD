"""
Yo! This script contains helper functions to visualize segmentation masks:
    - build_palette: Create a color palette for visualizing class IDs
    - save_grayscale: Save a class ID mask as a viewable grayscale image
    - overlay_mask: Overlay a class ID mask on the original RGB image for easy visual inspection
"""

import numpy as np
import cv2

# Explicit per-tissue RGB colors — defined as constants so SEKO and SAM 3
# outputs are always rendered with identical colors regardless of pipeline.
TISSUE_COLORS = {
    0: (0,   0,   0),    # background
    1: (255, 0,   0),    # bone
    2: (0,   255, 0),    # cartilage
    3: (0,   0,   255),  # gp (growth plate)
    4: (255, 255, 0),    # marrow
    5: (255, 0,   255),  # osteophyte
}


def build_palette(n_classes: int) -> np.ndarray:
    palette = np.zeros((n_classes, 3), dtype=np.uint8)

    for label_id, rgb in TISSUE_COLORS.items():
        if label_id < n_classes:
            palette[label_id] = rgb

    # Extend with pseudo-random colors for any labels beyond the defined set.
    for i in range(len(TISSUE_COLORS), n_classes):
        palette[i] = [(37*i) % 256, (91*i) % 256, (53*i) % 256]

    return palette


def save_grayscale(mask_ids: np.ndarray, path: str):
    maxv = int(mask_ids.max())
    if maxv > 0:
        vis = ((mask_ids.astype(np.float32) / maxv) * 255).astype(np.uint8)
    else:
        vis = mask_ids.astype(np.uint8)

    cv2.imwrite(path, vis)


def overlay_mask(image_rgb: np.ndarray,
                 mask_ids: np.ndarray,
                 n_classes: int,
                 alpha: float = 0.45,
                 background_class: int = 0):

    palette = build_palette(n_classes)
    mask_rgb = palette[mask_ids]

    out = image_rgb.astype(np.float32).copy()
    fg = mask_ids != background_class

    out[fg] = (1 - alpha) * out[fg] + alpha * mask_rgb[fg].astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)