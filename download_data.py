#!/usr/bin/env python3
"""
Download the standard hyperspectral datasets used by CACFTNet.

Each dataset consists of two separate .mat files:
  - Data cube (H x W x C)
  - Ground-truth label map (H x W)

Datasets:
  Indian Pines   -- Indian_pines_corrected.mat + Indian_pines_gt.mat
  Pavia University -- PaviaU.mat + PaviaU_gt.mat
  Houston 2013   -- Houston.mat + Houston_gt.mat

Sources:
  Indian Pines & Pavia:
    https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
  Houston 2013:
    https://hyperspectral.ee.uh.edu/?page_id=459

Usage:
  python download_data.py                     # download all three
  python download_data.py --dataset Indian     # download Indian Pines only
  python download_data.py --dataset Pavia      # download Pavia University only
  python download_data.py --dataset Houston    # download Houston 2013 only
  python download_data.py --data_dir ./data    # specify output directory
"""

import argparse
import os
import sys
import urllib.request
import hashlib


DATASETS = {
    "Indian": {
        "files": {
            "Indian_pines_corrected.mat": (
                "http://www.ehu.eus/ccwintco/uploads/6/67/"
                "Indian_pines_corrected.mat"
            ),
            "Indian_pines_gt.mat": (
                "http://www.ehu.eus/ccwintco/uploads/c/c4/"
                "Indian_pines_gt.mat"
            ),
        },
    },
    "Pavia": {
        "files": {
            "PaviaU.mat": (
                "http://www.ehu.eus/ccwintco/uploads/e/e3/"
                "PaviaU.mat"
            ),
            "PaviaU_gt.mat": (
                "http://www.ehu.eus/ccwintco/uploads/5/53/"
                "PaviaU_gt.mat"
            ),
        },
    },
    "Houston": {
        "files": {
            "Houston.mat": (
                "https://hyperspectral.ee.uh.edu/?page_id=459"
                "&download=1&kml_id=Houston18"
            ),
            "Houston_gt.mat": (
                "https://hyperspectral.ee.uh.edu/?page_id=459"
                "&download=1&kml_id=Houston18_gt"
            ),
        },
        "note": (
            "Houston 2013 may require manual download from "
            "https://hyperspectral.ee.uh.edu/?page_id=459  "
            "due to access restrictions. Place Houston.mat and "
            "Houston_gt.mat in the data directory yourself if "
            "automatic download fails."
        ),
    },
}


def _download_file(url, dest_path, desc=""):
    """Download a file with progress display."""
    print("  Downloading: {}".format(desc or os.path.basename(dest_path)))
    print("  URL: {}".format(url))
    print("  -> {}".format(dest_path))

    def _report_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            bar = "#" * (pct // 2) + "." * (50 - pct // 2)
            sys.stdout.write(
                "\r  Progress: [{}] {:5.1f}% ({:.1f}/{:.1f} MB)".format(
                    bar, pct,
                    downloaded / 1e6, total_size / 1e6
                )
            )
            sys.stdout.flush()
        else:
            sys.stdout.write(
                "\r  Downloaded: {:.1f} MB".format(downloaded / 1e6)
            )
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=_report_hook)
        sys.stdout.write("\n")
        print("  Done.")
    except Exception as e:
        sys.stdout.write("\n")
        # Remove partial file if it exists
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise RuntimeError(
            "Download failed for {}: {}".format(url, e)
        )


def download_dataset(dataset_name, data_dir="./data"):
    """
    Download all .mat files for a given dataset.

    Args:
        dataset_name: one of 'Indian', 'Pavia', 'Houston'
        data_dir: directory to save files into
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            "Unknown dataset '{}'. Choose from: {}".format(
                dataset_name, ", ".join(DATASETS.keys())
            )
        )

    ds = DATASETS[dataset_name]
    os.makedirs(data_dir, exist_ok=True)

    if "note" in ds:
        print("\nNote: {}".format(ds["note"]))

    all_exist = True
    for fname in ds["files"]:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            all_exist = False
            break

    if all_exist:
        print("[{}] All files already exist in {} -- skipping.".format(
            dataset_name, data_dir))
        return

    print("[{}] Downloading to {} ...".format(dataset_name, data_dir))
    for fname, url in ds["files"].items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            print("  {} already exists -- skipping.".format(fname))
            continue
        try:
            _download_file(url, fpath, desc=fname)
        except RuntimeError as e:
            print("  ERROR: {}".format(e))
            if "note" in ds:
                print("  See note above for manual download instructions.")
            print("  Skipping remaining files for this dataset.")
            break

    print()


def download_all(data_dir="./data"):
    """Download all three datasets."""
    for name in DATASETS:
        download_dataset(name, data_dir)


def ensure_dataset_available(dataset_name, data_dir="./data"):
    """
    Check if the required .mat files exist for a dataset.
    If not, offer to download them, or raise an error.

    Args:
        dataset_name: one of 'Indian', 'Pavia', 'Houston'
        data_dir: directory where .mat files should live

    Returns:
        (data_path, gt_path) - absolute paths to the two .mat files

    Raises:
        FileNotFoundError: if files are missing and download is not possible
    """
    file_map = {
        "Indian": ("Indian_pines_corrected.mat", "Indian_pines_gt.mat"),
        "Pavia": ("PaviaU.mat", "PaviaU_gt.mat"),
        "Houston": ("Houston.mat", "Houston_gt.mat"),
    }

    if dataset_name not in file_map:
        raise ValueError("Unknown dataset: {}".format(dataset_name))

    data_fname, gt_fname = file_map[dataset_name]
    data_path = os.path.join(data_dir, data_fname)
    gt_path = os.path.join(data_dir, gt_fname)

    if os.path.exists(data_path) and os.path.exists(gt_path):
        return data_path, gt_path

    missing = []
    if not os.path.exists(data_path):
        missing.append(data_fname)
    if not os.path.exists(gt_path):
        missing.append(gt_fname)

    print("Missing files for {}: {}".format(dataset_name, ", ".join(missing)))
    print("Attempting automatic download ...")
    download_dataset(dataset_name, data_dir)

    if not os.path.exists(data_path) or not os.path.exists(gt_path):
        still_missing = []
        if not os.path.exists(data_path):
            still_missing.append(data_fname)
        if not os.path.exists(gt_path):
            still_missing.append(gt_fname)
        raise FileNotFoundError(
            "Could not obtain required files for {}: {}. "
            "Please download them manually and place them in {}.".format(
                dataset_name, ", ".join(still_missing), data_dir
            )
        )

    return data_path, gt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download standard hyperspectral datasets for CACFTNet."
    )
    parser.add_argument(
        "--dataset",
        choices=["Indian", "Pavia", "Houston", "all"],
        default="all",
        help="which dataset to download (default: all)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="directory to save .mat files (default: ./data)",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        download_all(args.data_dir)
    else:
        download_dataset(args.dataset, args.data_dir)

    print("Download complete.")