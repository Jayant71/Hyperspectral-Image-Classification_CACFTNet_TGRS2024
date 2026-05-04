#!/usr/bin/env python3
"""
inspect_dataset.py — Standalone dataset inspector for hyperspectral images.

Analyzes a data cube (.dat + .hdr or .mat) and a ground-truth .mat,
then prints a comprehensive report and generates preview plots.

Usage:
  python inspect_dataset.py --data_path ./data/cube.dat --gt_path ./data/Patient_1_GT.mat
  python inspect_dataset.py --data_path ./data/mydata.mat  --gt_path ./data/mydata_gt.mat
"""

import argparse
import os
import sys
import numpy as np
from scipy.io import loadmat

# Try to import matplotlib for plots; gracefully skip if unavailable
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def read_envi_hdr(hdr_path):
    """Parse an ENVI .hdr file into a dict."""
    meta = {}
    with open(hdr_path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip().lower()
                val = val.strip()
                if "{" in val:
                    val = val[val.find("{") + 1 : val.find("}")]
                meta[key] = val
    meta["lines"]   = int(meta.get("lines", 0))
    meta["samples"] = int(meta.get("samples", 0))
    meta["bands"]   = int(meta.get("bands", 0))
    meta["header_offset"] = int(meta.get("header offset", "0"))
    meta["interleave"] = meta.get("interleave", "bsq").lower()
    dtype_map = {
        "1": np.uint8, "2": np.int16, "3": np.int32, "4": np.float32,
        "5": np.float64, "12": np.uint16, "13": np.uint32, "14": np.int64,
        "15": np.uint64,
    }
    meta["dtype"] = dtype_map.get(str(meta.get("data type", "4")), np.float32)
    return meta


def load_envi_dat(dat_path):
    """Load ENVI .dat + .hdr and return (H, W, C) array."""
    base, _ = os.path.splitext(dat_path)
    hdr_path = base + ".hdr"
    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f"ENVI header not found: {hdr_path}")
    meta = read_envi_hdr(hdr_path)
    H, W, C = meta["lines"], meta["samples"], meta["bands"]
    dtype = meta["dtype"]
    offset = meta["header_offset"]
    interleave = meta["interleave"]

    data = np.fromfile(dat_path, dtype=dtype, offset=offset)
    if interleave == "bsq":
        data = data.reshape((C, H, W)).transpose(1, 2, 0)
    elif interleave == "bil":
        data = data.reshape((H, C, W)).transpose(0, 2, 1)
    elif interleave == "bip":
        data = data.reshape((H, W, C))
    else:
        raise ValueError(f"Unsupported interleave: {interleave}")
    return data, meta


def load_mat_data(mat_path):
    """Load a .mat hyperspectral cube, return (H, W, C) array."""
    mat = loadmat(mat_path)
    candidates = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        arr = np.array(v)
        if arr.ndim >= 3:
            candidates.append((k, arr))
    if not candidates:
        raise ValueError(f"No 3D array found in {mat_path}")
    # Pick the largest 3D array
    best_key, best_arr = max(candidates, key=lambda x: x[1].size)
    if best_arr.ndim == 2:
        best_arr = best_arr[:, :, np.newaxis]
    return best_arr, best_key


def load_gt(mat_path):
    """Load a .mat ground truth, return (H, W) int array."""
    mat = loadmat(mat_path)
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        arr = np.array(v)
        if arr.ndim == 2:
            return arr.astype(np.int64), k
    raise ValueError(f"No 2D ground truth array found in {mat_path}")


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def suggest_rgb_bands(band_count):
    """Suggest 3 band indices for RGB composite based on band count."""
    if band_count <= 3:
        return (0, 1, 2)
    r = max(1, band_count // 3)
    g = max(1, band_count // 5)
    b = max(1, band_count // 10)
    return (min(r, band_count - 1), min(g, band_count - 1), min(b, band_count - 1))


def inspect_dataset(data_path, gt_path, out_dir="./inspect_output"):
    """Main inspection routine."""
    print_section("FILE PATHS")
    print(f"Data file : {os.path.abspath(data_path)}")
    print(f"GT file   : {os.path.abspath(gt_path)}")

    if not os.path.exists(data_path):
        print(f"ERROR: Data file does not exist: {data_path}")
        sys.exit(1)
    if not os.path.exists(gt_path):
        print(f"ERROR: GT file does not exist: {gt_path}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    ext = os.path.splitext(data_path)[-1].lower()
    if ext == ".dat":
        data, meta = load_envi_dat(data_path)
        is_envi = True
    elif ext == ".mat":
        data, data_key = load_mat_data(data_path)
        meta = None
        is_envi = False
    else:
        print(f"ERROR: Unsupported data extension '{ext}'. Use .dat or .mat")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load GT
    # -----------------------------------------------------------------------
    gt, gt_key = load_gt(gt_path)

    # -----------------------------------------------------------------------
    # Shapes
    # -----------------------------------------------------------------------
    print_section("SHAPES & ORIENTATION")
    print(f"Data shape  : {data.shape}  (ndim={data.ndim})")
    print(f"GT shape    : {gt.shape}    (ndim={gt.ndim})")

    if data.ndim >= 3:
        D0, D1 = data.shape[0], data.shape[1]
        H_gt, W_gt = gt.shape
        if D0 == H_gt and D1 == W_gt:
            print(f"  Orientation: (H={D0}, W={D1}, C={data.shape[2]})  ->  MATCHES GT spatial dims")
            H, W, C = data.shape
        elif data.shape[2] == H_gt and D0 == W_gt:
            print(f"  WARNING: Data appears to be (W, C, H) or permuted. Spatial dims may be swapped.")
            H, W, C = D1, D0, data.shape[2]
        elif D0 == W_gt and D1 == H_gt:
            print(f"  WARNING: Data appears (W, H, C). Spatial dims swapped vs GT.")
            H, W, C = D1, D0, data.shape[2]
        elif data.shape[0] == C and D0 == H_gt:
            print(f"  WARNING: Data might be (C, H, W) format. Need transpose.")
            H, W, C = D1, data.shape[2], D0
        else:
            print(f"  WARNING: Data spatial dims {data.shape[:2]} do NOT match GT {(H_gt, W_gt)}.")
            print(f"           Manual orientation correction may be needed.")
            H, W, C = data.shape[0], data.shape[1], data.shape[2]
    else:
        print(f"  Data is 2D — treating as single-band image.")
        H, W, C = data.shape[0], data.shape[1], 1

    # -----------------------------------------------------------------------
    # ENVI header details
    # -----------------------------------------------------------------------
    if is_envi and meta:
        print_section("ENVI HEADER METADATA")
        print(f"  lines           : {meta['lines']}")
        print(f"  samples         : {meta['samples']}")
        print(f"  bands           : {meta['bands']}")
        print(f"  interleave      : {meta['interleave'].upper()}")
        print(f"  data type       : {meta['dtype'].__name__}")
        print(f"  header offset   : {meta['header_offset']} bytes")
        print(f"  File size       : {os.path.getsize(data_path) / 1e6:.2f} MB")

    # -----------------------------------------------------------------------
    # Data statistics
    # -----------------------------------------------------------------------
    print_section("DATA CUBE STATISTICS")
    print(f"  Min value       : {data.min():.6f}")
    print(f"  Max value       : {data.max():.6f}")
    print(f"  Mean value      : {data.mean():.6f}")
    print(f"  Std dev         : {data.std():.6f}")
    print(f"  Data dtype      : {data.dtype}")

    # -----------------------------------------------------------------------
    # Ground truth statistics
    # -----------------------------------------------------------------------
    print_section("GROUND TRUTH STATISTICS")
    unique_classes = np.unique(gt)
    num_classes = int(np.max(gt))
    print(f"  GT variable     : '{gt_key}'")
    print(f"  Unique values   : {unique_classes.tolist()}")
    print(f"  Num classes     : {num_classes}")
    print(f"  Background (0)  : {np.sum(gt == 0)} pixels")
    print(f"  Total labeled   : {np.sum(gt > 0)} pixels")
    print(f"  Labeled fraction: {np.sum(gt > 0) / gt.size * 100:.2f}%")

    print("\n  Class distribution:")
    print(f"  {'Class':>6} | {'Count':>10} | {'Fraction %':>12}")
    print("  " + "-" * 32)
    for c in range(1, num_classes + 1):
        count = np.sum(gt == c)
        frac = count / gt.size * 100
        print(f"  {c:>6} | {count:>10} | {frac:>12.4f}")

    # -----------------------------------------------------------------------
    # Recommendations
    # -----------------------------------------------------------------------
    print_section("RECOMMENDATIONS")
    rgb = suggest_rgb_bands(C)
    print(f"  Suggested RGB bands for visualization: {rgb}")
    print(f"    -> use --rgb_bands {rgb[0]} {rgb[1]} {rgb[2]}")

    if C % 8 != 0:
        print(f"  Model choice    : Pavia model (C={C} is NOT divisible by 8)")
        print(f"    -> use --model pavia")
    else:
        print(f"  Model choice    : Indian/Houston model (C={C} is divisible by 8)")
        print(f"    -> use --model indian")

    if num_classes > 20:
        print(f"  Colormap note   : >20 classes — using extended colormap in visualize.py")

    # Estimate memory for patch extraction
    patch = 7
    n_labeled = np.sum(gt > 0)
    patch_mem_mb = n_labeled * patch * patch * C * 4 / 1e6  # float32
    print(f"  Patch memory    : ~{patch_mem_mb:.1f} MB for all labeled patches ({patch}x{patch}x{C}, float32)")

    # -----------------------------------------------------------------------
    # Generate preview plots
    # -----------------------------------------------------------------------
    if _HAS_MPL:
        os.makedirs(out_dir, exist_ok=True)
        print_section("PREVIEW PLOTS")

        # 1. RGB composite
        rgb_data = np.stack([
            data[:, :, min(rgb[0], C - 1)],
            data[:, :, min(rgb[1], C - 1)],
            data[:, :, min(rgb[2], C - 1)]
        ], axis=-1)
        rgb_min, rgb_max = rgb_data.min(), rgb_data.max()
        if rgb_max > rgb_min:
            rgb_norm = (rgb_data - rgb_min) / (rgb_max - rgb_min)
        else:
            rgb_norm = rgb_data

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(np.clip(rgb_norm, 0, 1))
        ax.set_title(f"RGB Composite (bands {rgb})")
        ax.axis("off")
        fig.tight_layout()
        rgb_path = os.path.join(out_dir, "preview_rgb.png")
        fig.savefig(rgb_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved RGB composite   : {rgb_path}")

        # 2. Ground truth map
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(gt, cmap="tab20", vmin=0, vmax=num_classes)
        ax.set_title("Ground Truth Label Map")
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                          ticks=range(num_classes + 1))
        cbar.ax.set_yticklabels(["BG"] + [str(i) for i in range(1, num_classes + 1)])
        fig.tight_layout()
        gt_path_out = os.path.join(out_dir, "preview_gt.png")
        fig.savefig(gt_path_out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved GT map          : {gt_path_out}")

        # 3. Mean spectral signatures
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for c in range(1, num_classes + 1):
            mask = (gt == c)
            if np.sum(mask) == 0:
                continue
            # If data is not (H,W,C) oriented correctly, we handle gracefully
            if data.ndim >= 3 and data.shape[0] == gt.shape[0] and data.shape[1] == gt.shape[1]:
                mean_spec = data[mask].mean(axis=0)
            else:
                # Fallback: just skip per-class spectra if orientations mismatch
                continue
            ax.plot(mean_spec, label=f"Class {c}", alpha=0.7)
        ax.set_xlabel("Band index")
        ax.set_ylabel("Reflectance / DN")
        ax.set_title("Mean Spectral Signatures per Class")
        ax.legend(loc="upper right", fontsize="small", ncol=2)
        fig.tight_layout()
        spec_path = os.path.join(out_dir, "preview_spectra.png")
        fig.savefig(spec_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved spectra plot    : {spec_path}")

        # 4. Class distribution bar chart
        counts = [np.sum(gt == c) for c in range(1, num_classes + 1)]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        bars = ax.bar(range(1, num_classes + 1), counts, color="#4C72B0")
        ax.set_xlabel("Class")
        ax.set_ylabel("Pixel count")
        ax.set_title("Ground Truth Class Distribution")
        ax.set_xticks(range(1, num_classes + 1))
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(cnt), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        dist_path = os.path.join(out_dir, "preview_class_dist.png")
        fig.savefig(dist_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved class dist      : {dist_path}")

        print(f"\nAll preview plots saved to: {os.path.abspath(out_dir)}")
    else:
        print("\n[Note] matplotlib not installed — skipping preview plot generation.")
        print("       Install with: pip install matplotlib")

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a hyperspectral dataset (data + ground truth)."
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data file (.dat + .hdr or .mat)")
    parser.add_argument("--gt_path", type=str, required=True,
                        help="Path to ground-truth .mat file")
    parser.add_argument("--out_dir", type=str, default="./inspect_output",
                        help="Directory to save preview plots (default: ./inspect_output)")
    args = parser.parse_args()

    inspect_dataset(args.data_path, args.gt_path, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
