# CACFTNet Fork — Separate Data/Ground-Truth Loader

This repository is a **fork** of the original [CACFTNet](https://github.com/CCRG-XJU/Hyperspectral-Image-Classification_CACFTNet_TGRS2024) implementation by **Cheng, Shuli; Chan, Runze; and Du, Anyu**.

> **Original Paper**: S. Cheng, R. Chan and A. Du, "CACFTNet: A Hybrid Cov-Attention and Cross-Layer Fusion Transformer Network for Hyperspectral Image Classification," in *IEEE Transactions on Geoscience and Remote Sensing*, vol. 62, pp. 1-17, 2024.  
> [DOI: 10.1109/TGRS.2024.3374081](https://doi.org/10.1109/TGRS.2024.3374081)

## What Changed in This Fork

1. **Separate `.mat` file loading**  
   The original code expected a single `.mat` file containing `input`, `TR`, and `TE` variables pre-split by the authors. This fork loads the standard **separate** data and ground-truth `.mat` files:
   - `*_corrected.mat` / `*.mat` — hyperspectral image cube `(H × W × C)`
   - `*_gt.mat` — ground-truth label map `(H × W)`, where `0 = background` and `1..N = class labels`

2. **Local train/test split**  
   The ground-truth map is now split into train and test masks **inside the pipeline** instead of relying on pre-made `TR` / `TE` masks. Two splitting modes are supported via CLI:
   - `fixed` — a fixed number of samples per class (e.g., `200`).
   - `ratio` — a percentage of each class (e.g., `0.1`).

3. **Automatic dataset download**  
   Running `demo.py` or `download_data.py` will automatically download Indian Pines and Pavia University datasets from [EHU's Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) if the `.mat` files are not found locally. Houston 2013 may require manual download due to access restrictions on the UH server.

4. **New command-line arguments**
   - `--data_path` — path to the hyperspectral data `.mat` (auto-resolved by `--dataset` if empty)
   - `--gt_path` — path to the ground-truth `.mat` (auto-resolved by `--dataset` if empty)
   - `--split_mode {fixed,ratio}`
   - `--train_samples` (default `200`)
   - `--train_ratio` (default `0.1`)
   - `--model_path` (default `./log/model.pt`) — configurable save/load path
   - `--channels_band` now auto-falls-back to the number of bands if left at `0`

5. **Added `data_utils.py`** — a clean utility module that handles loading, normalization, mirroring, splitting, and patch extraction.

6. **Added `download_data.py`** — a standalone dataset downloader script.

7. **Added `requirements.txt` and `.gitignore`** for better project hygiene.

Everything else (network architecture, training loop, metrics) is **unchanged** from the original.

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

- PyTorch 1.6+  
- CUDA 10.1+ (optional)  
- Python 3.7+  
- `einops`, `scipy`, `scikit-learn`, `matplotlib`, `numpy`

## Dataset

### Automatic download

Download all datasets at once:

```bash
python download_data.py
```

Download a specific dataset:

```bash
python download_data.py --dataset Indian
python download_data.py --dataset Pavia
python download_data.py --dataset Houston
```

Specify a custom output directory:

```bash
python download_data.py --dataset Indian --data_dir ./data
```

> **Note**: Indian Pines and Pavia University are downloaded from [EHU's hosted repository](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes). Houston 2013 may require manual download from [UH's site](https://hyperspectral.ee.uh.edu/?page_id=459) due to access restrictions — place `Houston.mat` and `Houston_gt.mat` into `./data/` manually.

Files are placed under `./data/` by default:

| Dataset | Data file | Ground-truth file | Bands |
|---------|-----------|-------------------|-------|
| **Indian Pines** | `Indian_pines_corrected.mat` | `Indian_pines_gt.mat` | 200 |
| **Pavia University** | `PaviaU.mat` | `PaviaU_gt.mat` | 103 |
| **Houston 2013** | `Houston.mat` | `Houston_gt.mat` | 144 |

`demo.py` will also auto-download missing files when you specify `--dataset` without explicit `--data_path`/`--gt_path`.

### Manual download

If automatic download fails, download from:
- **Indian Pines & Pavia**: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
- **Houston 2013**: https://hyperspectral.ee.uh.edu/?page_id=459

Place the `.mat` files in `./data/`.

## Network

- **Indian Pines & Houston 2013** → `vit_pytorch_indian_Houston.py`
- **Pavia University** → `vit_pytorch_pavia.py`

## Usage

When `--data_path` and `--gt_path` are omitted, they are auto-resolved from `--dataset` and missing files are downloaded automatically.

### Train

```bash
# Indian Pines (auto-downloads if missing)
python demo.py --dataset Indian --epoches 1500 --patches 7 --band_patches 1 --mode CAF --weight_decay 5e-3 --channels_band 200 --split_mode fixed --train_samples 200 --model_path ./log/IP.pt

# Pavia University
python demo.py --dataset Pavia --epoches 1680 --patches 7 --band_patches 1 --mode CAF --weight_decay 5e-3 --channels_band 103 --split_mode fixed --train_samples 200 --model_path ./log/Pavia.pt

# Houston
python demo.py --dataset Houston --epoches 1500 --patches 7 --band_patches 1 --mode CAF --weight_decay 5e-3 --channels_band 144 --split_mode fixed --train_samples 200 --model_path ./log/Houston.pt
```

Explicit paths can still be provided:

```bash
python demo.py --dataset Indian --data_path ./data/Indian_pines_corrected.mat --gt_path ./data/Indian_pines_gt.mat --epoches 1500 --patches 7 --band_patches 1 --mode CAF --weight_decay 5e-3 --channels_band 200 --split_mode fixed --train_samples 200 --model_path ./log/IP.pt
```

### Test / Inference

```bash
# Indian Pines
python demo.py --dataset Indian --flag_test test --patches 7 --band_patches 1 --mode CAF --channels_band 200 --model_path ./log/IP.pt

# Pavia University
python demo.py --dataset Pavia --flag_test test --patches 7 --band_patches 1 --mode CAF --channels_band 103 --model_path ./log/Pavia.pt

# Houston
python demo.py --dataset Houston --flag_test test --patches 7 --band_patches 1 --mode CAF --channels_band 144 --model_path ./log/Houston.pt
```

## Citation

If you use this code, please cite the **original paper**:

```bibtex
@ARTICLE{10460571,
  author={Cheng, Shuli and Chan, Runze and Du, Anyu},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={CACFTNet: A Hybrid Cov-Attention and Cross-Layer Fusion Transformer Network for Hyperspectral Image Classification},
  year={2024},
  volume={62},
  number={},
  pages={1-17},
  doi={10.1109/TGRS.2024.3374081}
}
```

## Note

This fork is made available for convenience when working with publicly released `.mat` datasets that ship as **separate** image / ground-truth files. Credit and all academic rights belong to the original authors.