"""
Resize human histology images (.jpg/.jpeg/.tif/.tiff).

Default input: ``Allen_Lab_SafO-FastGreen_images/TIFF_second_batch`` — collects images
from that folder and from each immediate subfolder (e.g. ``*_bone_TIFF/*.tif``),
excluding ``processed_images``.

Default output: ``.../TIFF_second_batch/processed_images`` (created if missing).

Also supports (from human_preprocess_colab workflow):
  - normalize_to_uint8: 16-bit / grayscale / RGBA -> uint8 BGR
  - Reinhard LAB stain normalization toward rat training reference (on by default; use ``--no-reinhard`` to disable)
  - Tiling: fixed-size patches (``--height`` x ``--width``) with optional near-white skip;
    on by default (use ``--no-export-tiles`` for one resized image per input).

Default: Reinhard on, tile export (no WB/CLAHE); output extension matches each input
(rename_mapping.csv unless disabled).

Very large TIFFs: ``OPENCV_IO_MAX_IMAGE_PIXELS`` is raised by default (see top of file);
set it in the environment before running if you need a different cap.

CLI summary: run ``python src/preprocess_human_data.py -h``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

# Large Leica stitches exceed OpenCV's default read cap (~2^30 pixels).
# Set before ``import cv2`` so ``imread`` allows huge TIFFs (override via env if needed).
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40 - 1))

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Reinhard reference (LAB), from 50 sampled rat training images — human_preprocess_colab.ipynb
# ---------------------------------------------------------------------------
DEFAULT_REINHARD_REF_MEANS = (155.6966, 144.2858, 102.3828)  # L, A, B
DEFAULT_REINHARD_REF_STDS = (40.1614, 25.1199, 25.5855)

# Subfolder under TIFF_second_batch where outputs go; excluded from input discovery
PROCESSED_SUBDIR_NAME = "processed_images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff"}


def gray_world_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """Apply Gray World white balance in BGR space."""
    img = img_bgr.astype(np.float32)
    channel_means = img.reshape(-1, 3).mean(axis=0)
    mean_gray = channel_means.mean()

    scales = np.where(channel_means > 1e-6, mean_gray / channel_means, 1.0).astype(np.float32)
    img *= scales

    return np.clip(img, 0, 255).astype(np.uint8)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert any bit-depth image to 3-channel uint8 BGR (Colab-compatible)."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        img_f = image.astype(np.float32)
        mx = img_f.max()
        if mx > 0:
            img_f /= mx
        image = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return image


def reinhard_normalize(
    image_uint8: np.ndarray,
    ref_means: tuple[float, float, float] | None = None,
    ref_stds: tuple[float, float, float] | None = None,
    tissue_gray_threshold: int = 220,
) -> np.ndarray:
    """
    Reinhard stain normalization in LAB space (matches Colab notebook).
    Shifts source LAB statistics (on tissue mask) to match reference.
    """
    if ref_means is None:
        ref_means = DEFAULT_REINHARD_REF_MEANS
    if ref_stds is None:
        ref_stds = DEFAULT_REINHARD_REF_STDS

    lab = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)

    gray = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    tissue_mask = gray < tissue_gray_threshold
    if tissue_mask.sum() == 0:
        tissue_mask = np.ones_like(gray, dtype=bool)

    for ch in range(3):
        src_mean = float(lab[:, :, ch][tissue_mask].mean())
        src_std = float(lab[:, :, ch][tissue_mask].std())
        if src_std < 1e-6:
            continue
        lab[:, :, ch] = (lab[:, :, ch] - src_mean) * (ref_stds[ch] / src_std) + ref_means[ch]

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_image_bgr(path: Path) -> np.ndarray | None:
    """Load image with IMREAD_UNCHANGED, then normalize to uint8 BGR."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    return normalize_to_uint8(raw)


def is_background_patch(patch_uint8: np.ndarray, threshold: float) -> bool:
    """True if fraction of near-white pixels >= threshold (Colab-compatible)."""
    gray = cv2.cvtColor(patch_uint8, cv2.COLOR_BGR2GRAY) if patch_uint8.ndim == 3 else patch_uint8
    return float(np.mean(gray > 220)) >= threshold


def tile_image(
    image_uint8: np.ndarray,
    tile_h: int,
    tile_w: int,
    overlap: int = 0,
    bg_threshold: float | None = 0.9,
) -> tuple[list[np.ndarray], list[tuple[int, int]], tuple[int, int]]:
    """
    Tile into (tile_h x tile_w) patches. Returns (patches_uint8, (y,x) coords, grid_shape).
    If bg_threshold is None, do not skip background patches.
    """
    h, w = image_uint8.shape[:2]
    step_h = max(1, tile_h - overlap)
    step_w = max(1, tile_w - overlap)

    if h < tile_h or w < tile_w:
        return [], [], (0, 0)

    y_starts = list(range(0, h - tile_h + 1, step_h))
    x_starts = list(range(0, w - tile_w + 1, step_w))
    if y_starts[-1] + tile_h < h:
        y_starts.append(h - tile_h)
    if x_starts[-1] + tile_w < w:
        x_starts.append(w - tile_w)

    patches: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    skipped = 0
    for y in y_starts:
        for x in x_starts:
            p = image_uint8[y : y + tile_h, x : x + tile_w]
            if bg_threshold is not None and is_background_patch(p, bg_threshold):
                skipped += 1
                continue
            patches.append(p.copy())
            coords.append((y, x))

    grid = (len(y_starts), len(x_starts))
    print(f"  Tiles: {len(patches)} kept, {skipped} bg-skipped (grid {grid[0]}x{grid[1]})")
    return patches, coords, grid


def load_reinhard_ref_json(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """JSON keys: L_mean, L_std, A_mean, A_std, B_mean, B_std."""
    with path.open(encoding="utf-8") as f:
        d = json.load(f)
    means = (float(d["L_mean"]), float(d["A_mean"]), float(d["B_mean"]))
    stds = (float(d["L_std"]), float(d["A_std"]), float(d["B_std"]))
    return means, stds


def normalize_color_stain_agnostic(
    img_bgr: np.ndarray,
    apply_white_balance: bool = False,
    apply_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: int = 8,
) -> np.ndarray:
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
    return cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)


def iter_image_paths(input_dir: Path) -> list[Path]:
    """
    Collect images directly under ``input_dir`` and under each immediate subdirectory,
    except ``processed_images`` (so outputs are never treated as inputs).
    """
    found: set[Path] = set()
    if not input_dir.is_dir():
        return []

    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            found.add(p.resolve())

    for sub in sorted(input_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.lower() == PROCESSED_SUBDIR_NAME.lower():
            continue
        try:
            for p in sorted(sub.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    found.add(p.resolve())
        except OSError:
            continue

    return sorted(found)


def _normalize_ext(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        return ".tif"
    return ext if ext.startswith(".") else f".{ext}"


def effective_output_extension(src_path: Path, output_ext: str) -> str:
    """
    Default behavior: use the source file's suffix (.tif, .jpeg, …).

    ``output_ext`` may be ``auto`` / ``same`` / empty for that behavior, or an
    explicit extension like ``.png`` to force a format.
    """
    policy = output_ext.strip().lower()
    if policy in ("", "auto", "same"):
        suf = src_path.suffix.lower()
        return _normalize_ext(suf) if suf else ".tif"
    return _normalize_ext(output_ext)


def processed_output_filename(src_path: Path, output_ext: str) -> str:
    ext = effective_output_extension(src_path, output_ext)
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
    output_ext: str = "auto",
    write_mapping: bool = True,
    mapping_filename: str = "rename_mapping.csv",
    apply_reinhard: bool = True,
    reinhard_ref_json: Path | None = None,
    export_tiles: bool = True,
    tile_overlap: int = 0,
    tile_skip_bg: bool = True,
    tile_bg_threshold: float = 0.9,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_means: tuple[float, float, float] | None = None
    ref_stds: tuple[float, float, float] | None = None
    if reinhard_ref_json is not None:
        ref_means, ref_stds = load_reinhard_ref_json(reinhard_ref_json)

    image_paths = iter_image_paths(input_dir)
    if not image_paths:
        raise ValueError(f"No supported images (.jpg/.jpeg/.tif/.tiff) found in: {input_dir}")

    print(
        f"Found {len(image_paths)} image files under {input_dir} "
        f"(root + subfolders except '{PROCESSED_SUBDIR_NAME}')"
    )
    print(f"Writing to: {output_dir}")
    if export_tiles:
        print(f"Mode: export tiles {height}x{width} overlap={tile_overlap} skip_bg={tile_skip_bg}")
    else:
        print(f"Mode: single output per image (--no-export-tiles), resize to {width}x{height}")
    if apply_reinhard:
        print("Reinhard LAB normalization: enabled")
        if reinhard_ref_json:
            print(f"  Reference stats from: {reinhard_ref_json}")
        else:
            print("  Reference stats: default (rat training, from Colab notebook)")
    else:
        print("Reinhard LAB normalization: disabled (--no-reinhard)")
    if not keep_original_names and not export_tiles:
        pol = output_ext.strip().lower()
        if pol in ("", "auto", "same"):
            print("Output naming: <stem>_processed.<ext same as input>")
        else:
            print(f"Output naming: <stem>_processed{_normalize_ext(output_ext)}")
    if apply_white_balance or apply_clahe:
        print("Extra color: white_balance=%s, clahe=%s" % (apply_white_balance, apply_clahe))
    elif not apply_reinhard:
        print("Color: unchanged unless --white-balance / --clahe")

    processed = 0
    skipped = 0
    mapping_rows: list[tuple[str, str]] = []

    for src_path in image_paths:
        img = load_image_bgr(src_path)
        if img is None:
            skipped += 1
            print(f"[skip] Could not read image: {src_path.name}")
            continue

        if apply_reinhard:
            img = reinhard_normalize(img, ref_means=ref_means, ref_stds=ref_stds)

        if export_tiles:
            ext_resolved = effective_output_extension(src_path, output_ext)
            bg_thr = tile_bg_threshold if tile_skip_bg else None
            patches, coords, grid = tile_image(img, height, width, tile_overlap, bg_thr)
            if not patches:
                skipped += 1
                print(f"[skip] No tiles for {src_path.name} (too small or all background?)")
                continue
            stem = src_path.stem
            for patch, (y, x) in zip(patches, coords):
                patch = normalize_color_stain_agnostic(
                    patch,
                    apply_white_balance=apply_white_balance,
                    apply_clahe=apply_clahe,
                    clahe_clip_limit=clahe_clip_limit,
                    clahe_tile_grid=clahe_tile_grid,
                )
                out_name = f"{stem}_tile_{y:05d}_{x:05d}{ext_resolved}"
                out_path = output_dir / out_name
                ok = cv2.imwrite(str(out_path), patch)
                if not ok:
                    skipped += 1
                    print(f"[skip] Could not write: {out_name}")
                    continue
                processed += 1
                if write_mapping:
                    mapping_rows.append((out_name, str(src_path.resolve())))
            continue

        img = resize_image(img, width=width, height=height)
        img = normalize_color_stain_agnostic(
            img,
            apply_white_balance=apply_white_balance,
            apply_clahe=apply_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid=clahe_tile_grid,
        )

        ext_resolved = effective_output_extension(src_path, output_ext)
        if keep_original_names:
            out_name = f"{src_path.stem}{ext_resolved}"
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

    if write_mapping and mapping_rows:
        map_path = output_dir / mapping_filename
        with map_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["output_filename", "original_path"])
            w.writerows(mapping_rows)
        print(f"Wrote mapping: {map_path}")

    print(f"Done. written={processed}, skipped={skipped}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_batch_root = repo_root / "Allen_Lab_SafO-FastGreen_images" / "TIFF_second_batch"
    default_input = default_batch_root
    default_output = default_batch_root / PROCESSED_SUBDIR_NAME

    parser = argparse.ArgumentParser(
        description="Preprocess human images: uint8 load, Reinhard (default), tile export (default) or whole-image resize."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=(
            "Root folder: images here and in each immediate subfolder are used "
            f"(excludes '{PROCESSED_SUBDIR_NAME}'). Default: {default_input}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Output folder (created if missing). Default: {default_output}",
    )
    parser.add_argument("--width", type=int, default=256, help="Output width (resize mode) or tile width")
    parser.add_argument("--height", type=int, default=192, help="Output height (resize mode) or tile height")

    parser.add_argument(
        "--no-reinhard",
        action="store_true",
        help="Disable Reinhard LAB normalization (enabled by default; rat reference from Colab)",
    )
    parser.add_argument(
        "--reinhard-ref-json",
        type=Path,
        default=None,
        help="JSON with L_mean,L_std,A_mean,A_std,B_mean,B_std for Reinhard target",
    )

    parser.add_argument(
        "--no-export-tiles",
        action="store_true",
        help="Write one resized image per input instead of tiling (tiling is on by default)",
    )
    parser.add_argument("--tile-overlap", type=int, default=0, help="Pixel overlap between tiles")
    parser.add_argument(
        "--no-tile-skip-bg",
        action="store_true",
        help="Do not skip mostly-white (background) tiles",
    )
    parser.add_argument(
        "--tile-bg-threshold",
        type=float,
        default=0.9,
        help="Skip tile if this fraction of pixels are near-white (default 0.9)",
    )

    parser.add_argument(
        "--white-balance",
        action="store_true",
        help="Apply Gray World white balance after Reinhard/resize (or on each tile)",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Apply CLAHE on LAB luminance",
    )
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument("--clahe-tile-grid", type=int, default=8, help="CLAHE tile grid size")

    parser.add_argument(
        "--keep-original-names",
        action="store_true",
        help="Use {stem}<ext> without _processed; extension follows --output-ext / auto (resize mode only)",
    )
    parser.add_argument(
        "--output-ext",
        type=str,
        default="auto",
        help="Output extension: default 'auto' = same as each input (.tif, .jpeg, …); set e.g. .png to force",
    )
    parser.add_argument("--no-mapping", action="store_true", help="Do not write mapping CSV")
    parser.add_argument("--mapping-filename", type=str, default="rename_mapping.csv")

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
        apply_reinhard=not args.no_reinhard,
        reinhard_ref_json=args.reinhard_ref_json,
        export_tiles=not args.no_export_tiles,
        tile_overlap=args.tile_overlap,
        tile_skip_bg=not args.no_tile_skip_bg,
        tile_bg_threshold=args.tile_bg_threshold,
    )


if __name__ == "__main__":
    main()
