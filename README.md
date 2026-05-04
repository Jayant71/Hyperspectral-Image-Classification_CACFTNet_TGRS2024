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

4. **Custom dataset support**  
   Supports arbitrary datasets with auto-detection of file formats. Data can be `.mat` (MATLAB) or `.dat` + `.hdr` (ENVI binary). Ground truth can be `.mat`. The `--dataset` argument accepts any name — built-ins auto-resolve paths, custom datasets require `--data_path` and `--gt_path`. Model architecture auto-selects based on whether the band count is divisible by 8 (Pavia model adds zero-padding for non-divisible counts). Use `--model {auto,indian,pavia}` to override.

5. **Plotting & visualization**
   Added `visualize.py` — a modular visualization module that produces:
   - **RGB composite** of the hyperspectral cube
   - **Ground-truth** label map
   - **Class distribution** bar chart (train/test split)
   - **Mean spectral signatures** per class
   - **Training curves** (loss + accuracy over epochs)
   - **Prediction map** after inference
   - **Side-by-side ground-truth vs prediction**
   - **Confusion matrix** heatmap
   - **Per-class accuracy** bar chart
   
   All figures are auto-saved to `./figures/<dataset>/`. Use `--no_plot` to disable.

6. **New command-line arguments**
   - `--data_path` — path to the hyperspectral data file `.mat` or `.dat` (auto-resolved for built-ins)
   - `--gt_path` — path to the ground-truth `.mat` file (auto-resolved for built-ins)
   - `--model` — model architecture: `auto`, `indian`, or `pavia`
   - `--rgb_bands` — 3 integers for RGB composite (e.g., `40 20 10`)
   - `--split_mode {fixed,ratio}`
   - `--train_samples` (default `200`)
   - `--train_ratio` (default `0.1`)
   - `--model_path` (default `./log/model.pt`)
   - `--output_dir` (default `./figures`)
   - `--no_plot` — disable all plotting
   - `--channels_band` auto-falls-back to band count if `0`

7. **Added `data_utils.py`** — a clean utility module that handles loading, normalization, mirroring, splitting, and patch extraction.

7. **Added `download_data.py`** — a standalone dataset downloader script.

8. **Added `visualize.py`** — a modular plotting and visualization module.

9. **Added `requirements.txt`, `pyproject.toml`, and `.gitignore`** for better project hygiene.

Everything else (network architecture, training loop, metrics) is **unchanged** from the original.

---

## Requirements

### 1. Install PyTorch with CUDA 12.6

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

### 2. Install remaining dependencies

```bash
pip install -r requirements.txt
```

Or using `pyproject.toml` (with [uv](https://github.com/astral-sh/uv) or `pip`):

```bash
pip install -e ".[cuda126] --index-url https://download.pytorch.org/whl/cu126"
```

Dependencies: `einops`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`

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

## Custom Datasets

The code now supports arbitrary datasets with auto-detection of file formats:

### Supported data formats

| Data file | GT file | Notes |
|-----------|---------|-------|
| `.mat` | `.mat` | Standard MATLAB format (auto-detects variable names by shape) |
| `.dat` + `.hdr` (ENVI) | `.mat` | ENVI binary cube + header; GT in `.mat` |

For ENVI `.dat` files, ensure the sibling `.hdr` header file exists in the same directory with the same base name (e.g., `MyData.dat` + `MyData.hdr`).

### Using a custom dataset

```bash
# Custom dataset with .mat data + .mat GT
python demo.py \
  --dataset MyData \
  --data_path ./data/mydata.mat \
  --gt_path ./data/mydata_gt.mat \
  --epoches 500 --patches 7 --band_patches 1 \
  --mode CAF --channels_band 0 \
  --split_mode ratio --train_ratio 0.2 \
  --model_path ./log/mydata.pt

# Custom dataset with ENVI .dat data + .mat GT
python demo.py \
  --dataset MyData \
  --data_path ./data/mydata.dat \
  --gt_path ./data/mydata_gt.mat \
  --epoches 500 --patches 7 --band_patches 1 \
  --mode CAF --channels_band 0 \
  --split_mode ratio --train_ratio 0.2 \
  --model_path ./log/mydata.pt
```

### Model selection for custom datasets

The model architecture differs because the Pavia model adds zero-padding for band counts not divisible by 8. Use `--model` to control it:

| `--model` | When to use |
|-----------|-------------|
| `auto` (default) | Auto-selects: Pavia model if `C % 8 != 0`, else Indian/Houston model |
| `indian` | Force Indian Pines / Houston architecture (no padding) |
| `pavia`  | Force Pavia architecture (adds zero-padding channel for non-divisible-by-8 bands) |

```bash
# Force Pavia model architecture (e.g., for 103 bands)
python demo.py --dataset MyData --data_path ./data/mydata.dat --gt_path ./data/mydata_gt.mat --model pavia ...
```

### Custom RGB bands for visualization

Default RGB band indices are dataset-dependent. For custom datasets, the default is `(C//3, C//5, C//10)`. Override with:

```bash
python demo.py --dataset MyData ... --rgb_bands 40 20 10
```

## Visualization

All visualizations are generated automatically and saved as PNGs under `./figures/<dataset>/`:

| Figure | When generated | File |
|--------|---------------|------|
| RGB composite | Before training | `rgb_composite.png` |
| Ground-truth map | Before training | `ground_truth.png` |
| Class distribution | Before training | `class_distribution.png` |
| Spectral signatures | Before training | `spectral_signatures.png` |
| Training curves | After training | `training_curves.png` |
| Prediction map | After training / test | `prediction_map.png` |
| GT vs Prediction | After training / test | `gt_vs_prediction.png` |
| Confusion matrix | After training / test | `confusion_matrix.png` |
| Per-class accuracy | After training / test | `per_class_accuracy.png` |

Use `--no_plot` to skip all visualization, or `--output_dir` to change the save directory.

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

## Project Structure

```
.
├── demo.py                          # Main training/testing entry point
├── data_utils.py                    # Data loading, splitting, patch extraction
├── download_data.py                 # Automatic dataset downloader
├── visualize.py                     # Plotting & visualization module
├── vit_pytorch_indian_Houston.py    # CACFTNet model (Indian Pines, Houston)
├── vit_pytorch_pavia.py            # CACFTNet model (Pavia University)
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata
├── .gitignore
├── README.md
├── README.txt
└── data/                            # Dataset .mat files (auto-downloaded)
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