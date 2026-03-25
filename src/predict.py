# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:51:11 2024

@author: justinjoseph

Unified inference script for TOAD histology segmentation.
Supports single-image and batch modes, SEKO and SAM 3 backbones.

Outputs per image (always three files):
  1. Raw predicted mask  — class IDs as uint8
  2. Grayscale visualisation
  3. RGB overlay on the original image

Single-image mode  (--image):
    python src/predict.py --image path/to/img.tif --output path/to/mask.png

Batch mode (default, reads input_dir from config):
    python src/predict.py
    python src/predict.py --backbone sam3 --prompt-mask-dir outputs/Original_Predictions
"""

import argparse
import os
import glob
import time

import cv2
import numpy as np
import yaml

from visualize_mask import save_grayscale, overlay_mask

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "baseline.yml"
)

# Number of tissue classes SAM 3 can return (labels 0-5).
SAM3_N_CLASSES = 6


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_model_from_path(model_path):
    from keras.models import load_model
    return load_model(model_path, compile=False)


# ── Image loading ──────────────────────────────────────────────────────────

def _load_image(img_path: str, img_height: int, img_width: int, rgb: bool):
    """
    Load and validate an image from disk.

    Returns
    -------
    feed : (1, H, W, 3) float32 in [0, 1]  — model input tensor
    img_rgb : (H, W, 3) uint8 RGB           — used for overlay / SAM 3
    """
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"Could not read image: {img_path}")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"Image must be 3-channel (got shape {img_bgr.shape})")

    if img_bgr.shape[:2] != (img_height, img_width):
        print(f"  Resizing to {img_width}x{img_height}")
        img_bgr = cv2.resize(img_bgr, (img_width, img_height))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    feed_src = img_rgb if rgb else img_bgr
    feed = (feed_src.astype(np.float32) / 255.0)[None, ...]  # (1,H,W,3)

    return feed, img_rgb


# ── Output saving ──────────────────────────────────────────────────────────

def _save_outputs(pred_ids: np.ndarray, img_rgb: np.ndarray,
                  raw_path: str, vis_path: str, overlay_path: str,
                  n_classes: int, alpha: float, background_class: int):
    """Write raw mask, grayscale visualisation, and colour overlay."""
    cv2.imwrite(raw_path, pred_ids)
    print(f"  Raw mask  : {raw_path}")

    save_grayscale(pred_ids, vis_path)
    print(f"  Grayscale : {vis_path}")

    ov = overlay_mask(img_rgb, pred_ids, n_classes, alpha, background_class)
    cv2.imwrite(overlay_path, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    print(f"  Overlay   : {overlay_path}")


# ── Per-backbone inference ─────────────────────────────────────────────────

def _infer_seko(model, feed: np.ndarray, img_path: str):
    """
    Run one SEKO forward pass.

    Returns
    -------
    pred_ids  : (H, W) uint8
    n_classes : int
    """
    print(f"  Input  shape={feed.shape}  min/max={feed.min():.3f}/{feed.max():.3f}")
    t0 = time.time()
    pred = model.predict(feed, verbose=0)   # (1, H, W, K)
    elapsed = (time.time() - t0) * 1000

    n_classes = int(pred.shape[-1])
    pred_ids = np.argmax(pred, axis=-1)[0].astype(np.uint8)

    unique = np.unique(pred_ids)
    print(f"  Forward pass {elapsed:.1f} ms  |  labels={unique}  n_classes={n_classes}")
    return pred_ids, n_classes


def _infer_sam3(adapter, img_rgb: np.ndarray, img_path: str,
                prompt_mask_dir: str, prompt_gen):
    """
    Generate prompts from the matching mask in prompt_mask_dir and run SAM 3.

    Returns
    -------
    pred_ids : (H, W) uint8
    """
    name = os.path.basename(img_path)
    stem = os.path.splitext(name)[0]
    candidates = glob.glob(os.path.join(prompt_mask_dir, stem + ".*"))
    if not candidates:
        raise FileNotFoundError(
            f"No prompt mask for '{name}' in {prompt_mask_dir}"
        )
    prompts = prompt_gen.from_mask_path(candidates[0])

    t0 = time.time()
    pred_ids = adapter.predict(img_rgb, prompts)
    elapsed = (time.time() - t0) * 1000

    unique = np.unique(pred_ids)
    print(f"  SAM 3  {elapsed:.1f} ms  |  labels={unique}")
    return pred_ids


# ── Path helpers ───────────────────────────────────────────────────────────

def _batch_output_paths(name: str, out_dir: str):
    """Return (raw, vis, overlay) paths for batch mode."""
    raw  = os.path.join(out_dir, "Original_Predictions",  f"prediction_{name}")
    vis  = os.path.join(out_dir, "Visualized_Predictions", f"visualized_prediction_{name}")
    ov   = os.path.join(out_dir, "Overlay_Predictions",    f"overlay_prediction_{name}")
    return raw, vis, ov


def _single_output_paths(output_arg: str):
    """Return (raw, vis, overlay) paths for single-image mode."""
    base, _ = os.path.splitext(output_arg)
    return output_arg, base + "_vis.png", base + "_overlay.png"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TOAD inference — single image or batch, SEKO or SAM 3"
    )

    # ── Input / output ──
    parser.add_argument(
        "--image", default=None,
        help="Single-image mode: path to input image",
    )
    parser.add_argument(
        "--output", default=None,
        help="Single-image mode: path for raw output mask (vis + overlay saved alongside)",
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Batch mode: directory of .tif* images (overrides config)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Batch mode: output root directory (overrides config)",
    )

    # ── Backbone ──
    parser.add_argument(
        "--backbone", choices=["seko", "sam3"], default="seko",
        help="Inference backbone: 'seko' (default) or 'sam3'",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to SEKO .keras checkpoint (overrides config)",
    )
    parser.add_argument(
        "--prompt-mask-dir", default=None,
        help=(
            "sam3 only: directory of single-channel masks used to generate "
            "prompts. Files must share the same stem as the input images."
        ),
    )

    # ── Preprocessing / visualisation ──
    parser.add_argument(
        "--height", type=int, default=None,
        help="Model input height (overrides config, default 192)",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Model input width (overrides config, default 256)",
    )
    parser.add_argument(
        "--rgb", action="store_true",
        help="Convert BGR→RGB before feeding SEKO (use if model was trained on RGB)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.45,
        help="Overlay blend strength 0..1 (default 0.45)",
    )
    parser.add_argument(
        "--background-class", type=int, default=0,
        help="Class ID to treat as background in overlay (default 0)",
    )

    args = parser.parse_args()

    cfg = load_config()

    img_height = args.height or cfg["data"]["img_height"]
    img_width  = args.width  or cfg["data"]["img_width"]

    # ── Resolve image list ──────────────────────────────────────────────
    single_mode = args.image is not None

    if single_mode:
        if not args.output:
            raise SystemExit("error: --image requires --output")
        image_paths = [args.image]
    else:
        input_dir = args.input_dir or cfg["inference"]["input_dir"]
        image_paths = sorted(glob.glob(os.path.join(input_dir, "*.tif*")))
        if not image_paths:
            raise SystemExit(f"No .tif* files found in {input_dir}")
        print(f"Found {len(image_paths)} images in {input_dir}")

        output_dir = args.output_dir or cfg["paths"]["output_dir"]
        for sub in ("Original_Predictions", "Visualized_Predictions", "Overlay_Predictions"):
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    # ── Load backbone ───────────────────────────────────────────────────
    if args.backbone == "seko":
        model_path = args.model or cfg["paths"]["weights_in"]
        print(f"Loading SEKO model: {model_path}")
        model = load_model_from_path(model_path)
        adapter = prompt_gen = None

    else:  # sam3
        if not args.prompt_mask_dir:
            raise SystemExit(
                "error: --backbone sam3 requires --prompt-mask-dir"
            )
        from sam_adapter import load_sam_adapter
        from utils.sam_prompt_gen import SamPromptGenerator
        sam_cfg   = cfg.get("sam3", {})
        ckpt_path = sam_cfg.get("ckpt_path", "")
        device    = sam_cfg.get("device", "cpu")
        print(f"Loading SAM 3 adapter: {ckpt_path}")
        adapter    = load_sam_adapter(ckpt_path=ckpt_path, device=device)
        prompt_gen = SamPromptGenerator()
        model      = None

    # ── Inference loop ──────────────────────────────────────────────────
    for img_path in image_paths:
        name = os.path.basename(img_path)
        print(f"\n[{name}]")

        try:
            feed, img_rgb = _load_image(img_path, img_height, img_width, args.rgb)
        except (IOError, ValueError) as e:
            print(f"  Skipping: {e}")
            continue

        if args.backbone == "seko":
            pred_ids, n_classes = _infer_seko(model, feed, img_path)
        else:
            try:
                pred_ids = _infer_sam3(adapter, img_rgb, img_path,
                                       args.prompt_mask_dir, prompt_gen)
            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
                continue
            n_classes = SAM3_N_CLASSES

        if single_mode:
            raw_path, vis_path, ov_path = _single_output_paths(args.output)
            os.makedirs(os.path.dirname(os.path.abspath(raw_path)), exist_ok=True)
        else:
            raw_path, vis_path, ov_path = _batch_output_paths(name, output_dir)

        _save_outputs(pred_ids, img_rgb, raw_path, vis_path, ov_path,
                      n_classes, args.alpha, args.background_class)

        unique = np.unique(pred_ids)
        if len(unique) == 1 and unique[0] == args.background_class:
            print(
                "  NOTE: prediction is all background. Try:\n"
                "    - rerun with/without --rgb (BGR vs RGB mismatch)\n"
                "    - verify preprocessing matches training\n"
                "    - confirm correct checkpoint"
            )


if __name__ == "__main__":
    main()
