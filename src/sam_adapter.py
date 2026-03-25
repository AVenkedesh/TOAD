# -*- coding: utf-8 -*-
"""
SAM 3 inference wrapper for TOAD histology segmentation.

Loads a SAM 3 checkpoint, accepts box + point prompts per tissue class
(as produced by utils/sam_prompt_gen.py), and returns a single H×W uint8
multi-class mask in TOAD's label schema:
    0 = background
    1 = bone
    2 = cartilage
    3 = gp (growth plate)
    4 = marrow
    5 = osteophyte

The file is safely importable even when no SAM 3 checkpoint is present;
inference calls will raise NotImplementedError in that case.
"""

import os

import cv2
import numpy as np

# Tissue label IDs that SAM 3 is responsible for segmenting.
TISSUE_LABELS = {1, 2, 3, 4, 5}


class SamAdapter:
    """
    Wraps a SAM 3 image predictor for multi-class TOAD segmentation.

    Parameters
    ----------
    ckpt_path : str
        Path to the SAM 3 checkpoint file (.pt).
    device : str
        PyTorch device string, e.g. "cpu" or "cuda".
    """

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.ckpt_path = ckpt_path
        self.device = device
        self._predictor = None

        if os.path.isfile(ckpt_path):
            self._load_predictor()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_predictor(self):
        from sam3 import build_sam3_image_predictor  # noqa: import inside method
        self._predictor = build_sam3_image_predictor(
            ckpt_path=self.ckpt_path,
            device=self.device,
        )

    def _check_ready(self):
        if self._predictor is None:
            raise NotImplementedError(
                f"SAM 3 checkpoint not found at '{self.ckpt_path}'. "
                "Download the checkpoint and update sam3.ckpt_path in "
                "configs/baseline.yml to enable SAM 3 inference."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray, prompts: dict) -> np.ndarray:
        """
        Run SAM 3 on a single image using box + point prompts per class.

        Parameters
        ----------
        image : np.ndarray
            RGB image array, shape (H, W, 3), dtype uint8.
        prompts : dict
            Output of sam_prompt_gen.generate_prompts(). Keys are tissue
            label ints (1-5); values are dicts with keys:
              - "box"    : np.ndarray shape (4,)  in xyxy format
              - "points" : np.ndarray shape (N, 2) in xy format
              - "labels" : np.ndarray shape (N,)  1=foreground, 0=background

        Returns
        -------
        np.ndarray
            Multi-class mask, shape (H, W), dtype uint8.
            Pixel values: 0=background, 1=bone, 2=cartilage,
                          3=gp, 4=marrow, 5=osteophyte.
            Later classes (higher label_id) overwrite earlier ones where
            masks overlap.
        """
        self._check_ready()

        H, W = image.shape[:2]
        output_mask = np.zeros((H, W), dtype=np.uint8)

        self._predictor.set_image(image)

        for label_id, prompt in prompts.items():
            if label_id not in TISSUE_LABELS:
                continue

            box = prompt.get("box", None)
            points = prompt.get("points", None)
            point_labels = prompt.get("labels", None)

            # SAM 3 expects box shape (1, 4).
            if box is not None:
                box = np.array(box, dtype=np.float32)
                if box.ndim == 1:
                    box = box[None]

            masks, _scores, _logits = self._predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                box=box,
                multimask_output=False,
            )

            # masks: (1, H, W) boolean — take the single output mask.
            binary_mask = masks[0].astype(bool)
            output_mask[binary_mask] = label_id

        return output_mask

    def predict_image(self, image_path: str, prompts: dict) -> np.ndarray:
        """
        Load an image from disk and run full SAM 3 inference.

        Parameters
        ----------
        image_path : str
            Path to the input image (.tif / .tiff / .png etc.).
        prompts : dict
            Prompt dict as returned by sam_prompt_gen.generate_prompts().

        Returns
        -------
        np.ndarray
            Multi-class mask, shape (H, W), dtype uint8.
        """
        self._check_ready()

        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Could not read image: {image_path}")

        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.predict(image_rgb, prompts)


def load_sam_adapter(ckpt_path: str, device: str = "cpu") -> "SamAdapter":
    """
    Convenience factory that returns a SamAdapter.
    Safe to call without a checkpoint — inference will raise NotImplementedError.
    """
    return SamAdapter(ckpt_path=ckpt_path, device=device)
