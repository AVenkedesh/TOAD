# -*- coding: utf-8 -*-
"""
train_sam3.py  —  Finetune SAM 3 on TOAD histology segmentation

Strategy
--------
- Use build_sam3_image_model(eval_mode=False) — the full image model with Hungarian matcher.
- Freeze the ViT backbone (32-layer, 1024-dim) to keep GPU memory and wall-time manageable.
- Finetune: geometry encoder, transformer encoder+decoder, segmentation head.
- One image per step (batch_size=1); all tissue classes in that image are prompted together.
- Loss: Dice + BCE on matched (pred, GT) mask pairs, computed directly from model output.

Inputs (from train.csv)
-----------------------
  image_path : RGB .tif, any size  → resized to 1008×1008
  mask_path  : uint8 .tif, values 0-5 (TOAD label schema)
  stain      : HE | SafO | TolBlu  (unused during training, kept for stratified splits)

Usage
-----
  conda activate seko-sam3
  python src/train_sam3.py
  python src/train_sam3.py --ckpt models/sam3_checkpoint.pt --epochs 30 --lr 1e-4
"""

import argparse
import contextlib
import json
import os
import time

from scipy.optimize import linear_sum_assignment

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from sam3 import build_sam3_image_model
from sam3.model.data_misc import BatchedFindTarget, FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.train.loss.loss_fns import dice_loss

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "baseline.yml")

# ── Constants ──────────────────────────────────────────────────────────────
"""
Need to play around with some of these numbers.
MIN_MASK_AREA_PX is a heuristic to filter out tiny tissue classes that are likely noise.
The tradeoff -> some classes (like ostephytes) may be small but clinically important, so we don't want to set this too high.
Need to figure out the proper values.
"""
SAM3_RESOLUTION  = 1008      # ViT input size
SAM3_IMAGE_MEAN  = [0.5, 0.5, 0.5]
SAM3_IMAGE_STD   = [0.5, 0.5, 0.5]
MIN_MASK_AREA_PX = 50        # ignore tissue classes with < 50 px in mask
TISSUE_LABELS    = [1, 2, 3, 4, 5]  # bone, cartilage, gp, marrow, osteophyte


# ── Data augmentation ────────────────────────────────────────────────────
_augmentation_transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=20),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize((SAM3_RESOLUTION, SAM3_RESOLUTION)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=SAM3_IMAGE_MEAN, std=SAM3_IMAGE_STD),
])

def preprocess_image_with_augmentation(image_rgb: np.ndarray) -> torch.Tensor:
    """Apply augmentation and preprocessing: (H, W, 3) uint8 → (1, 3, 1008, 1008) float32."""
    t = torch.from_numpy(image_rgb).permute(2, 0, 1)  # (3, H, W)
    return _augmentation_transform(t).unsqueeze(0)

# ── Dataset ────────────────────────────────────────────────────────────────

class ToadSam3Dataset(Dataset):
    """
    Loads image-mask pairs from train.csv.

    __getitem__ returns a dict:
        image_rgb    : (H, W, 3) uint8  — original image for display
        image_tensor : (1, 3, 1008, 1008) float32  — model input
        boxes_cxcywh : (N, 4) float32  — GT boxes, normalized cxcywh [0, 1]
        binary_masks : (N, H_orig, W_orig) bool — GT per-class binary masks
        label_ids    : (N,) long — tissue class IDs (1-5)
    N is the number of tissue classes present in this image (≥1).
    Images with no tissue classes after filtering are skipped.
    """

    def __init__(self, records: list[dict], img_height: int = 192, img_width: int = 256):
        self.records    = records
        self.img_height = img_height
        self.img_width  = img_width

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # ── Load image ──────────────────────────────────────────────────
        img_bgr = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Cannot read image: {rec['image_path']}")
        if img_bgr.shape[:2] != (self.img_height, self.img_width):
            img_bgr = cv2.resize(img_bgr, (self.img_width, self.img_height))
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── Load mask ───────────────────────────────────────────────────
        mask = np.array(Image.open(rec["mask_path"])).astype(np.uint8)
        if mask.shape[:2] != (self.img_height, self.img_width):
            mask = cv2.resize(
                mask, (self.img_width, self.img_height),
                interpolation=cv2.INTER_NEAREST
            )

        H, W = mask.shape

        # ── Extract per-class boxes + masks ─────────────────────────────
        boxes_list, masks_list, labels_list = [], [], []
        for label_id in TISSUE_LABELS:
            binary = (mask == label_id)
            if binary.sum() < MIN_MASK_AREA_PX:
                continue

            rows = np.any(binary, axis=1)
            cols = np.any(binary, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

            # Normalize to [0, 1] in cxcywh format
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H

            boxes_list.append([cx, cy, bw, bh])
            masks_list.append(binary)
            labels_list.append(label_id)

        if not boxes_list:
            # Rare edge-case: mask is all background after filtering.
            # Return a dummy foreground so DataLoader doesn't crash.
            boxes_list  = [[0.5, 0.5, 1.0, 1.0]]
            masks_list  = [np.ones((H, W), dtype=bool)]
            labels_list = [0]

        return {
            "image_rgb":    image_rgb,
            "image_tensor": preprocess_image(image_rgb),
            "boxes_cxcywh": torch.tensor(boxes_list, dtype=torch.float32),
            "binary_masks": torch.from_numpy(np.stack(masks_list)),  # (N, H, W) bool
            "label_ids":    torch.tensor(labels_list, dtype=torch.long),
        }


# ── Prompt / target builders ───────────────────────────────────────────────

def build_prompt(boxes_cxcywh: torch.Tensor, device) -> Prompt:
    """
    Build a SAM 3 Prompt from N GT bounding boxes.

    Prompt expects:
        box_embeddings : (N, B, 4)  — N instances, B=1 image
        box_mask       : (B, N)     — True = valid
        box_labels     : (N, B)     — 1 = positive prompt
    """
    N = boxes_cxcywh.shape[0]
    boxes = boxes_cxcywh.to(device).unsqueeze(1)           # (N, 1, 4)
    box_mask   = torch.ones(1, N, device=device, dtype=torch.bool)
    box_labels = torch.ones(N, 1, device=device, dtype=torch.long)
    return Prompt(box_embeddings=boxes, box_mask=box_mask, box_labels=box_labels)


def build_find_input(device) -> FindStage:
    """Minimal FindStage with no input boxes (prompts passed via geometric_prompt)."""
    return FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


def build_find_target(
    boxes_cxcywh: torch.Tensor,
    binary_masks: torch.Tensor,
    device,
) -> BatchedFindTarget:
    """
    Build a BatchedFindTarget for one image with N tissue instances.

    boxes_cxcywh : (N, 4) normalized cxcywh
    binary_masks : (N, H, W) bool
    """
    N = boxes_cxcywh.shape[0]
    H, W = binary_masks.shape[1], binary_masks.shape[2]

    boxes    = boxes_cxcywh.to(device)           # (N, 4)
    segments = binary_masks.to(device)            # (N, H, W) bool

    # Semantic mask = union of all instance masks — (1, H, W)
    semantic = segments.any(dim=0, keepdim=True)  # (1, H, W)

    return BatchedFindTarget(
        num_boxes=torch.tensor([N], device=device, dtype=torch.long),
        boxes=boxes,
        boxes_padded=boxes.unsqueeze(0),          # (1, N, 4)
        repeated_boxes=boxes.repeat(2, 1),        # (2N, 4) — for DAC/hybrid matching
        segments=segments,
        semantic_segments=semantic,
        is_valid_segment=torch.ones(N, device=device, dtype=torch.bool),
        is_exhaustive=torch.ones(1, device=device, dtype=torch.bool),
        object_ids=torch.arange(N, device=device, dtype=torch.long),
        object_ids_padded=torch.arange(N, device=device, dtype=torch.long).unsqueeze(0),
    )


# ── Loss ───────────────────────────────────────────────────────────────────

def compute_loss_and_dice(
    out: dict, boxes_cxcywh: torch.Tensor, binary_masks: torch.Tensor, device
) -> tuple[torch.Tensor, float]:
    """
    Compute Dice + BCE loss and DICE metric on Hungarian-matched (pred, GT) pairs.

    Returns (loss, dice_metric) where dice_metric is a plain float [0, 1].

    out["pred_masks"] : (B, Q, H_pred, W_pred) — raw logits from model
    out["pred_boxes"] : (B, Q, 4) — predicted boxes in cxcywh normalized
    boxes_cxcywh      : (N, 4) GT boxes
    binary_masks      : (N, H_orig, W_orig) bool GT masks
    """
    pred_masks = out.get("pred_masks")   # (1, Q, H, W)
    pred_boxes = out.get("pred_boxes")   # (1, Q, 4)
    if pred_masks is None or pred_boxes is None:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0

    N = boxes_cxcywh.shape[0]
    gt_boxes = boxes_cxcywh.to(device)       # (N, 4)
    pred_b   = pred_boxes[0].detach()         # (Q, 4)

    # L1 cost between each GT box and each predicted query → (N, Q)
    cost = torch.cdist(gt_boxes, pred_b, p=1).cpu().numpy()

    # Hungarian: assign each GT to the best predicted query
    gt_idx, pred_idx = linear_sum_assignment(cost)   # both shape (N,)

    if len(pred_idx) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0

    matched_preds = pred_masks[0, pred_idx]                       # (M, H_pred, W_pred)
    matched_gt    = binary_masks[gt_idx].to(device).float()       # (M, H_orig, W_orig)

    H_pred, W_pred = matched_preds.shape[-2], matched_preds.shape[-1]

    # Resize GT to match prediction resolution
    matched_gt = F.interpolate(
        matched_gt.unsqueeze(1),
        size=(H_pred, W_pred),
        mode="nearest",
    ).squeeze(1)   # (M, H_pred, W_pred)

    M = matched_preds.shape[0]

    loss_bce  = F.binary_cross_entropy_with_logits(
        matched_preds.flatten(1),
        matched_gt.flatten(1),
        reduction="mean",
    )
    loss_dice = dice_loss(
        matched_preds.flatten(1),
        matched_gt.flatten(1),
        num_boxes=torch.tensor(M, device=device, dtype=torch.float32),
    )

    # DICE metric: threshold predictions at 0.5, no gradient needed
    with torch.no_grad():
        pred_bin  = (matched_preds.sigmoid() > 0.5).float()
        inter     = (pred_bin * matched_gt).sum(dim=(-2, -1))
        union     = pred_bin.sum(dim=(-2, -1)) + matched_gt.sum(dim=(-2, -1))
        dice_vals = (2.0 * inter + 1e-6) / (union + 1e-6)
        dice_metric = dice_vals.mean().item()

    return loss_bce + loss_dice, dice_metric


# ── Training / validation steps ────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train: bool) -> dict:
    """Returns {"loss": float, "dice": float}."""
    model.train(train)
    context = torch.enable_grad if train else torch.no_grad
    # SAM3 is designed to run under bfloat16 autocast on CUDA
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    total_loss, total_dice, n = 0.0, 0.0, 0

    with context():
        for batch in loader:
            image_tensor  = batch["image_tensor"].squeeze(0).to(device)  # (1, 3, H, W)
            boxes_cxcywh  = batch["boxes_cxcywh"].squeeze(0)             # (N, 4)
            binary_masks  = batch["binary_masks"].squeeze(0)             # (N, H, W)

            with autocast_ctx:
                # ── Backbone forward (frozen — no grad) ──────────────────
                with torch.no_grad():
                    backbone_out = model.backbone.forward_image(image_tensor)
                    text_out     = model.backbone.forward_text(["visual"], device=device)
                    backbone_out.update(text_out)

                # ── Build prompts + target ───────────────────────────────
                prompt      = build_prompt(boxes_cxcywh, device)
                find_input  = build_find_input(device)
                find_target = build_find_target(boxes_cxcywh, binary_masks, device)

                # ── Forward through trainable components ─────────────────
                out = model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    find_target=find_target,
                    geometric_prompt=prompt,
                )

            loss, dice = compute_loss_and_dice(out, boxes_cxcywh, binary_masks, device)

            if train and loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice
            n += 1

    return {"loss": total_loss / max(n, 1), "dice": total_dice / max(n, 1)}


# ── Freezing ────────────────────────────────────────────────────────────────

def freeze_backbone(model):
    """Freeze the ViT image encoder — the most expensive component."""
    frozen, total = 0, 0
    for name, param in model.named_parameters():
        total += 1
        if "backbone.vision_backbone.trunk" in name:
            param.requires_grad_(False)
            frozen += 1
    print(f"Froze {frozen}/{total} parameters (ViT backbone)")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Finetune SAM 3 on TOAD histology data")
    parser.add_argument("--ckpt",       default=cfg["sam3"].get("ckpt_path", None),
                        help="SAM 3 checkpoint path (downloads from HF if omitted)")
    parser.add_argument("--csv",        default=cfg["paths"]["train_csv"],
                        help="Path to train.csv (image_path, mask_path, stain)")
    parser.add_argument("--output-dir", default=cfg["paths"]["output_dir"],
                        help="Directory for saved checkpoints")
    parser.add_argument("--device",     default=cfg["sam3"].get("device", "cuda"),
                        help="'cuda', 'mps', or 'cpu'")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--val-frac",   type=float, default=0.15,
                        help="Fraction of data held out for validation")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    records = df.to_dict("records")
    train_recs, val_recs = train_test_split(
        records, test_size=args.val_frac, random_state=args.seed
    )
    print(f"Train: {len(train_recs)}  Val: {len(val_recs)}")

    img_h = cfg["data"]["img_height"]
    img_w = cfg["data"]["img_width"]

    train_ds = ToadSam3Dataset(train_recs, img_h, img_w)
    val_ds   = ToadSam3Dataset(val_recs,   img_h, img_w)

    # DataLoader with batch_size=1 — SAM 3 processes one image at a time
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2)

    # ── Model ───────────────────────────────────────────────────────────
    load_from_hf = (args.ckpt is None) or (not os.path.isfile(args.ckpt))
    print(f"Loading SAM 3 model (checkpoint: {'HuggingFace' if load_from_hf else args.ckpt})")
    model = build_sam3_image_model(
        checkpoint_path=args.ckpt if not load_from_hf else None,
        load_from_HF=load_from_hf,
        device=str(device),
        eval_mode=False,
    )

    freeze_backbone(model)

    # ── Optimizer ───────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameter tensors: {len(trainable)}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # ── Training loop ────────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.output_dir, "sam3_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_val_dice = 0.0
    best_ckpt_path = os.path.join(ckpt_dir, "sam3_toad_best.pt")

    metrics_log = {"epoch": [], "train_loss": [], "val_loss": [],
                   "train_dice": [], "val_dice": []}
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    print(f"\n{'Epoch':>5}  {'TrainLoss':>9}  {'ValLoss':>8}  "
          f"{'TrainDICE':>9}  {'ValDICE':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 68)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_stats = run_epoch(model, train_loader, optimizer, device, train=True)
        val_stats   = run_epoch(model, val_loader,   optimizer, device, train=False)

        scheduler.step()

        elapsed = time.time() - t0
        marker = " ★" if val_stats["dice"] > best_val_dice else ""

        print(
            f"{epoch:5d}  "
            f"{train_stats['loss']:9.4f}  {val_stats['loss']:8.4f}  "
            f"{train_stats['dice']:9.4f}  {val_stats['dice']:8.4f}  "
            f"{scheduler.get_last_lr()[0]:.2e}  {elapsed:5.0f}s"
            f"{marker}",
            flush=True,
        )

        # Update metrics log and persist after every epoch
        metrics_log["epoch"].append(epoch)
        metrics_log["train_loss"].append(round(train_stats["loss"], 6))
        metrics_log["val_loss"].append(round(val_stats["loss"], 6))
        metrics_log["train_dice"].append(round(train_stats["dice"], 6))
        metrics_log["val_dice"].append(round(val_stats["dice"], 6))
        with open(metrics_path, "w") as f:
            json.dump(metrics_log, f, indent=2)

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
        if val_stats["dice"] > best_val_dice:
            best_val_dice = val_stats["dice"]
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_loss": val_stats["loss"], "val_dice": val_stats["dice"]},
                best_ckpt_path,
            )

    # Always save the final epoch too
    final_ckpt = os.path.join(ckpt_dir, f"sam3_toad_epoch{args.epochs}.pt")
    torch.save(
        {"epoch": args.epochs, "model_state_dict": model.state_dict(),
         "val_loss": val_stats["loss"], "val_dice": val_stats["dice"]},
        final_ckpt,
    )
    print(f"\nDone. Best val DICE: {best_val_dice:.4f}  Best val loss: {best_val_loss:.4f}")
    print(f"Metrics saved → {metrics_path}")
    print(f"Best checkpoint → {best_ckpt_path}")


if __name__ == "__main__":
    main()
