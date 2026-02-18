"""
Yo! This script contains helper functions to visualize segmentation masks:
    - build_palette: Create a color palette for visualizing class IDs
    - save_grayscale: Save a class ID mask as a viewable grayscale image
    - overlay_mask: Overlay a class ID mask on the original RGB image for easy visual inspection
"""

import numpy as np
import cv2


def build_palette(n_classes: int) -> np.ndarray:
    base = np.array([
        [0, 0, 0],        # 0 background
        [255, 0, 0],      # 1
        [0, 255, 0],      # 2
        [0, 0, 255],      # 3
        [255, 255, 0],    # 4
        [255, 0, 255],    # 5
        [0, 255, 255],    # 6
    ], dtype=np.uint8)

    if n_classes <= len(base):
        return base[:n_classes]

    palette = np.zeros((n_classes, 3), dtype=np.uint8)
    palette[:len(base)] = base

    for i in range(len(base), n_classes):
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