"""
sam_prompt_gen.py
Task 5 — TOAD × SAM 3 Integration

Generates SAM-compatible point and bounding box prompts from
TOAD ground-truth segmentation masks.

Each mask is a single-channel PNG/NPY with pixel values:
    0 = background
    1 = bone
    2 = cartilage
    3 = gp (growth plate)
    4 = marrow
    5 = osteophyte

Outputs per image:
    - One bounding box prompt  (x1, y1, x2, y2) per tissue class present
    - One centroid point prompt (x, y) per tissue class present
    - Optionally N random foreground points per class (for multi-point prompting)

Usage:
    from src.utils.sam_prompt_gen import SamPromptGenerator

    gen = SamPromptGenerator()
    prompts = gen.from_mask_path("path/to/mask.png")
    # or
    prompts = gen.from_mask_array(mask_np)
"""

import os
import numpy as np
import cv2
from typing import Optional

# ── Label schema (must match src/utils/label_mask_gen.py) ──────────────────
TARGET_VALUES = {
    "bone":       1,
    "cartilage":  2,
    "gp":         3,
    "marrow":     4,
    "osteophyte": 5,
}

LABEL_NAMES = {v: k for k, v in TARGET_VALUES.items()}  # reverse lookup


class SamPromptGenerator:
    """
    Generates SAM-compatible prompts from TOAD multi-class segmentation masks.

    Args:
        n_random_points (int): Number of additional random foreground points
            to sample per class beyond the centroid. Set to 0 to return only
            the centroid. Default: 3.
        min_area_px (int): Minimum connected-component area (pixels) to
            generate a prompt for. Filters out annotation noise. Default: 50.
    """

    def __init__(self, n_random_points: int = 3, min_area_px: int = 50):
        self.n_random_points = n_random_points
        self.min_area_px = min_area_px

    # ── Public API ────────────────────────────────────────────────────────

    def from_mask_path(self, mask_path: str) -> dict:
        """Load a mask from disk and generate prompts."""
        ext = os.path.splitext(mask_path)[-1].lower()
        if ext == ".npy":
            mask = np.load(mask_path)
        else:
            # Load as grayscale PNG (pixel values are class labels)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")
        return self.from_mask_array(mask)

    def from_mask_array(self, mask: np.ndarray) -> dict:
        """
        Generate prompts from a numpy mask array.

        Args:
            mask: H×W uint8 array with pixel values 0-5.

        Returns:
            dict keyed by class label int (1-5), each containing:
            {
                "name":        str,           # tissue name
                "bbox":        (x1,y1,x2,y2), # tight bounding box
                "centroid":    (cx, cy),       # centroid of binary mask
                "points":      [(x,y), ...],   # centroid + random points
                "point_labels":[1, 1, ...],    # all foreground (SAM convention)
                "binary_mask": np.ndarray,     # H×W binary mask for this class
            }
            Classes not present in the mask are omitted.
        """
        mask = np.squeeze(mask)  # handle (H,W,1) if needed
        assert mask.ndim == 2, f"Expected 2D mask, got shape {mask.shape}"

        prompts = {}
        for label_val in TARGET_VALUES.values():
            binary = (mask == label_val).astype(np.uint8)
            if binary.sum() < self.min_area_px:
                continue  # class absent or too small

            bbox   = self._get_bbox(binary)
            points = self._get_points(binary)

            if bbox is None or not points:
                continue

            prompts[label_val] = {
                "name":         LABEL_NAMES[label_val],
                "bbox":         bbox,
                "centroid":     points[0],          # first point is centroid
                "points":       points,
                "point_labels": [1] * len(points),  # 1 = foreground in SAM
                "binary_mask":  binary,
            }

        return prompts

    def batch_from_dir(self, mask_dir: str, ext: str = ".png") -> dict:
        """
        Process all masks in a directory.

        Args:
            mask_dir: Path to directory containing mask files.
            ext:      File extension to filter on (".png" or ".npy").

        Returns:
            dict mapping filename -> prompts dict (from from_mask_array).
        """
        results = {}
        files = sorted([
            f for f in os.listdir(mask_dir)
            if f.endswith(ext)
        ])
        if not files:
            raise FileNotFoundError(
                f"No {ext} files found in {mask_dir}"
            )
        for fname in files:
            fpath = os.path.join(mask_dir, fname)
            try:
                results[fname] = self.from_mask_path(fpath)
            except Exception as e:
                print(f"[WARN] Skipping {fname}: {e}")
        return results

    # ── Private helpers ───────────────────────────────────────────────────

    def _get_bbox(self, binary: np.ndarray) -> Optional[tuple]:
        """
        Compute tight axis-aligned bounding box over all foreground pixels.

        Returns (x1, y1, x2, y2) in image coordinates (col, row),
        matching SAM's expected box format.
        Returns None if no foreground pixels exist.
        """
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)

        if not rows.any():
            return None

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return (int(x1), int(y1), int(x2), int(y2))

    def _get_points(self, binary: np.ndarray) -> list:
        """
        Returns a list of (x, y) foreground points:
            [0]   = centroid of the largest connected component
            [1:N] = random foreground points sampled from the full mask

        Uses connected components so the centroid lands inside the mask
        even for irregular shapes (e.g. thin cartilage strips).
        """
        # Find connected components — use largest for centroid
        num_labels, cc_map, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels <= 1:  # only background
            return []

        # Skip label 0 (background), pick largest foreground component
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(areas)) + 1  # +1 offset for bg

        # Filter out components below min_area_px
        if areas[largest_idx - 1] < self.min_area_px:
            return []

        cx, cy = centroids[largest_idx]

        # Snap centroid to nearest actual foreground pixel
        # (centroid may fall outside mask for crescent/donut shapes)
        cx, cy = self._snap_to_foreground(binary, cx, cy)
        points = [(int(cx), int(cy))]

        # Sample additional random foreground points if requested
        if self.n_random_points > 0:
            fg_coords = np.argwhere(binary > 0)  # shape (N, 2) as (row, col)
            if len(fg_coords) >= self.n_random_points:
                chosen = fg_coords[
                    np.random.choice(len(fg_coords), self.n_random_points,
                                     replace=False)
                ]
                for (row, col) in chosen:
                    points.append((int(col), int(row)))  # convert to (x,y)

        return points

    def _snap_to_foreground(
        self, binary: np.ndarray, cx: float, cy: float
    ) -> tuple:
        """
        If the centroid (cx, cy) doesn't fall on a foreground pixel,
        find the nearest foreground pixel using distance transform.
        """
        cx_int, cy_int = int(round(cx)), int(round(cy))
        h, w = binary.shape

        # Clamp to image bounds
        cx_int = max(0, min(cx_int, w - 1))
        cy_int = max(0, min(cy_int, h - 1))

        if binary[cy_int, cx_int] > 0:
            return cx_int, cy_int

        # Distance transform: find nearest fg pixel
        dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
        # Zero out bg pixels in distance map, find min over fg
        fg_coords = np.argwhere(binary > 0)
        if len(fg_coords) == 0:
            return cx_int, cy_int

        dists = dist[fg_coords[:, 0], fg_coords[:, 1]]
        # Actually we want the fg pixel nearest to (cx_int, cy_int)
        row_dists = np.sqrt(
            (fg_coords[:, 0] - cy_int) ** 2 +
            (fg_coords[:, 1] - cx_int) ** 2
        )
        nearest = fg_coords[np.argmin(row_dists)]
        return int(nearest[1]), int(nearest[0])  # (x, y)


# ── Visualization helper (optional, for debugging) ────────────────────────

def visualize_prompts(
    image: np.ndarray,
    prompts: dict,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Draw bounding boxes and centroid points on an image for debugging.

    Args:
        image:     H×W×3 BGR image (or H×W grayscale, will be converted).
        prompts:   Output of SamPromptGenerator.from_mask_array().
        save_path: If provided, saves the visualization to this path.

    Returns:
        Annotated BGR image.
    """
    # Color palette per class (BGR)
    COLORS = {
        1: (255, 180,  60),   # bone       — blue
        2: ( 60, 220,  60),   # cartilage  — green
        3: ( 60,  60, 255),   # gp         — red
        4: (200,  60, 200),   # marrow     — purple
        5: ( 60, 220, 220),   # osteophyte — yellow
    }

    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for label_val, p in prompts.items():
        color = COLORS.get(label_val, (255, 255, 255))
        x1, y1, x2, y2 = p["bbox"]

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

        # Label text
        cv2.putText(
            vis, p["name"], (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA
        )

        # Points: centroid = filled circle, random = small dot
        for i, (px, py) in enumerate(p["points"]):
            radius = 4 if i == 0 else 2
            cv2.circle(vis, (px, py), radius, color, -1)

    if save_path:
        cv2.imwrite(save_path, vis)

    return vis


# ── Quick smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== sam_prompt_gen.py smoke test ===\n")

    # Create a synthetic 192×256 mask matching TOAD dims
    H, W = 192, 256
    mask = np.zeros((H, W), dtype=np.uint8)

    # bone: large region top-left
    mask[10:80,  10:120] = 1
    # cartilage: thin strip
    mask[80:90,  10:120] = 2
    # gp: small blob
    mask[90:110, 40:80]  = 3
    # marrow: interior of bone region
    mask[20:70,  20:110] = 4
    # osteophyte: small irregular blob
    mask[50:65,  180:210] = 5

    gen = SamPromptGenerator(n_random_points=3)
    prompts = gen.from_mask_array(mask)

    for label_val, p in prompts.items():
        print(f"  [{label_val}] {p['name']}")
        print(f"       bbox:     {p['bbox']}")
        print(f"       centroid: {p['centroid']}")
        print(f"       n_points: {len(p['points'])}")
        print()

    # Visualize on blank image
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    vis = visualize_prompts(blank, prompts, save_path="/tmp/prompt_vis.png")
    print("Visualization saved to /tmp/prompt_vis.png")
    print("\nSmoke test passed.")
