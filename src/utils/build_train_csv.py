"""
build_train_csv.py

Scans Labelled_images/ and Input_images/ under rat_original, matches
image-mask pairs by filename stem, and writes a CSV used by train_sam3.py.

Matching strategy (applied in order, first hit wins):
  1. Exact stem match   — strip known suffixes, compare directly
  2. Suffix-token match — compare only the final alphanumeric token
     (handles TolBlu case/hyphen divergence: GALIDO_TOL_BLUE_06r ↔ Gal-IDO_Tol-Blue_06r)

Unmatched files are reported but not written to the CSV.

Output CSV columns:
  image_path, mask_path, stain

Usage:
    python src/utils/build_train_csv.py
    python src/utils/build_train_csv.py --data-root /path/to/rat_original --out data/train.csv
"""

import argparse
import csv
import os
import re
import glob
from collections import defaultdict


# ── Stain config ──────────────────────────────────────────────────────────
# Each entry: (labelled_subdir, input_subdir, stain_label)
STAIN_CONFIG = [
    ("H&E",    "H&E/Raw256x192",     "HE"),
    ("Safo",   "SafO/Raw 256x192",   "SafO"),
    ("TolBlu", "TolBlu/Raw 256x192", "TolBlu"),
]


# ── Stem extraction ───────────────────────────────────────────────────────

def mask_stem(filename: str) -> str:
    """Strip _bone.tif / _bone_binary.tif suffix."""
    return re.sub(r'(_bone_binary|_bone)\.tif$', '', filename)


def image_stem(filename: str) -> str:
    """Strip ,_resized_gray.tif / _resized_gray.tif / plain .tif suffix."""
    s = re.sub(r',?_resized_gray\.tif$', '', filename)
    if s == filename:                       # no resized_gray suffix
        s = re.sub(r'\.tif$', '', filename) # fall back to stripping plain .tif
    return s


def suffix_token(stem: str) -> str:
    """
    Last alphanumeric token after splitting on [-_\\s]+.
    e.g. 'GALIDO_TOL_BLUE_06r' → '06r'
         'Gal-IDO_Tol-Blue_06r' → '06r'
         'MMT_71'               → '71'
    """
    tokens = re.split(r'[-_\s]+', stem.lower())
    return tokens[-1] if tokens else stem.lower()


# ── Per-stain pairing ─────────────────────────────────────────────────────

def pair_stain(labelled_dir: str, input_dir: str, stain: str) -> tuple[list, list]:
    """
    Match masks to images for one stain type.

    Returns
    -------
    pairs    : list of (image_path, mask_path, stain)
    unmatched: list of mask paths with no corresponding image
    """
    masks  = sorted(glob.glob(os.path.join(labelled_dir, "*.tif")))
    images = sorted(glob.glob(os.path.join(input_dir,    "*.tif")))

    # Build image lookup: exact stem → path
    img_by_exact  = {image_stem(os.path.basename(f)): f for f in images}
    # Fallback: suffix token → path (skip ambiguous tokens)
    img_by_suffix = defaultdict(list)
    for f in images:
        img_by_suffix[suffix_token(image_stem(os.path.basename(f)))].append(f)
    img_by_suffix = {k: v[0] for k, v in img_by_suffix.items() if len(v) == 1}

    pairs, unmatched = [], []

    for mpath in masks:
        mname  = os.path.basename(mpath)
        mstem  = mask_stem(mname)
        mtoken = suffix_token(mstem)

        if mstem in img_by_exact:
            pairs.append((img_by_exact[mstem], mpath, stain))
        elif mtoken in img_by_suffix:
            pairs.append((img_by_suffix[mtoken], mpath, stain))
        else:
            unmatched.append(mpath)

    return pairs, unmatched


# ── Main ──────────────────────────────────────────────────────────────────

def build(data_root: str, out_path: str):
    all_pairs    = []
    all_unmatched = []

    for labelled_sub, input_sub, stain in STAIN_CONFIG:
        labelled_dir = os.path.join(data_root, "Labelled_images", labelled_sub)
        input_dir    = os.path.join(data_root, "Input_images",    input_sub)

        if not os.path.isdir(labelled_dir):
            print(f"[WARN] Labelled dir not found, skipping {stain}: {labelled_dir}")
            continue
        if not os.path.isdir(input_dir):
            print(f"[WARN] Input dir not found, skipping {stain}: {input_dir}")
            continue

        pairs, unmatched = pair_stain(labelled_dir, input_dir, stain)
        all_pairs.extend(pairs)
        all_unmatched.extend(unmatched)

        print(f"  {stain:8s}  {len(pairs):3d} matched  {len(unmatched):3d} unmatched")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path", "stain"])
        writer.writerows(all_pairs)

    print(f"\nWrote {len(all_pairs)} pairs → {out_path}")

    if all_unmatched:
        print(f"\nUnmatched masks ({len(all_unmatched)}) — no corresponding image found:")
        for p in all_unmatched:
            print(f"  {p}")


def main():
    # Default data_root relative to this file: ../../.. from src/utils/
    default_root = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "seko_storage", "data", "rat_original"
    ))
    default_out = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "seko_storage", "data", "rat_original", "train.csv"
    ))

    parser = argparse.ArgumentParser(description="Build image-mask pair CSV for TOAD training")
    parser.add_argument("--data-root", default=default_root,
                        help="Path to rat_original/ directory")
    parser.add_argument("--out", default=default_out,
                        help="Output CSV path")
    args = parser.parse_args()

    print(f"Scanning: {args.data_root}")
    build(args.data_root, args.out)


if __name__ == "__main__":
    main()
