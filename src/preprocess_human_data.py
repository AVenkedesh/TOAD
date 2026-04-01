"""
Resize human histology images (.jpg/.jpeg/.tif in the input folder only, not subfolders).
Default: resize only (no color edits); save as ``{stem}_processed.png``; write
``rename_mapping.csv`` mapping output names to source paths unless disabled.

Default paths: ``human_data/original_data`` -> ``human_data/processed_data`` (under repo root).
Original images must be in the original_data folder.
Processed images will be saved in the processed_data folder.

Default size: 256 x 192. Run ``python src/preprocess_human_data.py -h`` for argparse help.

CLI (one line each):
  --input-dir       Folder with source images.
  --output-dir      Folder for processed images (created if missing).
  --width           Output width in pixels (default 256).
  --height          Output height in pixels (default 192).
  --white-balance   Optional Gray World color balance (off unless set).
  --clahe           Optional local contrast on luminance (off unless set).
  --clahe-clip-limit   CLAHE strength when --clahe is set (default 2.0).
  --clahe-tile-grid    CLAHE tile size when --clahe is set (default 8).
  --keep-original-names   Use ``{stem}.{ext}`` instead of ``{stem}_processed.{ext}``.
  --output-ext      Output extension, e.g. .png (default .png).
  --no-mapping      Do not write rename_mapping.csv.
  --mapping-filename   Name of the CSV in output-dir (default rename_mapping.csv).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def gray_world_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """Apply Gray World white balance in BGR space."""
    img = img_bgr.astype(np.float32)
    channel_means = img.reshape(-1, 3).mean(axis=0)
    mean_gray = channel_means.mean()

    # Avoid division-by-zero if an image has an empty/constant channel.
    scales = np.where(channel_means > 1e-6, mean_gray / channel_means, 1.0).astype(np.float32)
    img *= scales

    return np.clip(img, 0, 255).astype(np.uint8)


def normalize_color_stain_agnostic(
    img_bgr: np.ndarray,
    apply_white_balance: bool = False,
    apply_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: int = 8,
) -> np.ndarray:
    """
    Normalize color/contrast without assuming a specific stain.

    Steps:
    - Optional Gray World white balance (robust baseline color correction)
    - Optional CLAHE on LAB L channel (local contrast normalization)
    """
    out = img_bgr

    if apply_white_balance:
        out = gray_world_white_balance(out)

    if apply_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=(int(clahe_tile_grid), int(clahe_tile_grid)),
        )
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return out


def resize_image(img_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize image to model-compatible dimensions."""
    return cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)


def iter_jpeg_paths(input_dir: Path) -> list[Path]:
    """Collect jpg/jpeg files (case-insensitive) from the input folder."""
    exts = {".jpg", ".jpeg", ".tif"}
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _normalize_ext(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        return ".png"
    return ext if ext.startswith(".") else f".{ext}"


def processed_output_filename(src_path: Path, output_ext: str) -> str:
    """Build name like ``{stem}_processed.png``."""
    ext = _normalize_ext(output_ext)
    return f"{src_path.stem}_processed{ext}"


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    apply_white_balance: bool,
    apply_clahe: bool,
    clahe_clip_limit: float,
    clahe_tile_grid: int,
    *,
    keep_original_names: bool = False,
    output_ext: str = ".png",
    write_mapping: bool = True,
    mapping_filename: str = "rename_mapping.csv",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_jpeg_paths(input_dir)
    if not image_paths:
        raise ValueError(f"No supported images (.jpg/.jpeg/.tif) found in: {input_dir}")

    print(f"Found {len(image_paths)} JPEG files in: {input_dir}")
    print(f"Writing processed images to: {output_dir}")
    if not keep_original_names:
        print(f"Output naming: <original_stem>_processed{_normalize_ext(output_ext)}")
    if apply_white_balance or apply_clahe:
        print("Color: white_balance=%s, clahe=%s" % (apply_white_balance, apply_clahe))
    else:
        print("Color: unchanged (resize only)")

    processed = 0
    skipped = 0
    mapping_rows: list[tuple[str, str]] = []

    for src_path in image_paths:
        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            print(f"[skip] Could not read image: {src_path.name}")
            continue

        img = resize_image(img, width=width, height=height)
        img = normalize_color_stain_agnostic(
            img,
            apply_white_balance=apply_white_balance,
            apply_clahe=apply_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid=clahe_tile_grid,
        )

        if keep_original_names:
            out_name = f"{src_path.stem}{_normalize_ext(output_ext)}"
        else:
            out_name = processed_output_filename(src_path, output_ext)

        out_path = output_dir / out_name
        ok = cv2.imwrite(str(out_path), img)
        if not ok:
            skipped += 1
            print(f"[skip] Could not write image: {out_path.name}")
            continue

        if not keep_original_names and write_mapping:
            mapping_rows.append((out_name, str(src_path.resolve())))

        processed += 1

    if not keep_original_names and write_mapping and mapping_rows:
        map_path = output_dir / mapping_filename
        with map_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["output_filename", "original_path"])
            w.writerows(mapping_rows)
        print(f"Wrote rename mapping: {map_path}")

    print(f"Done. processed={processed}, skipped={skipped}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "human_data" / "original_data"
    default_output = repo_root / "human_data" / "processed_data"

    parser = argparse.ArgumentParser(
        description="Preprocess human images: resize (default); optional color normalization; PNG output."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Folder containing original JPEG images (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Folder to save processed images (default: {default_output})",
    )

    # Modify the width and height to the desired size of the images
    parser.add_argument("--width", type=int, default=256, help="Output width")
    parser.add_argument("--height", type=int, default=192, help="Output height")

    parser.add_argument(
        "--white-balance",
        action="store_true",
        help="Apply Gray World white balance (off by default)",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Apply CLAHE on LAB luminance (off by default)",
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit (used if CLAHE enabled)",
    )
    parser.add_argument(
        "--clahe-tile-grid",
        type=int,
        default=8,
        help="CLAHE tile grid size (used if CLAHE enabled)",
    )

    parser.add_argument(
        "--keep-original-names",
        action="store_true",
        help="Save with the same filename as the input (no _processed suffix)",
    )
    parser.add_argument(
        "--output-ext",
        type=str,
        default=".png",
        help="Output file extension (default: .png, lossless)",
    )
    parser.add_argument(
        "--no-mapping",
        action="store_true",
        help="Do not write rename_mapping.csv",
    )
    parser.add_argument(
        "--mapping-filename",
        type=str,
        default="rename_mapping.csv",
        help="CSV filename inside output-dir listing output -> original path",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        apply_white_balance=args.white_balance,
        apply_clahe=args.clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=args.clahe_tile_grid,
        keep_original_names=args.keep_original_names,
        output_ext=args.output_ext,
        write_mapping=not args.no_mapping,
        mapping_filename=args.mapping_filename,
    )


if __name__ == "__main__":
    main()
