# CACFTNet Fork — Separate Data/Ground-Truth Loader

This repository is a fork of the original CACFTNet implementation by Cheng, Shuli; Chan, Runze; and Du, Anyu.
Original repository: https://github.com/CCRG-XJU/Hyperspectral-Image-Classification_CACFTNet_TGRS2024

Original Paper: S. Cheng, R. Chan and A. Du, "CACFTNet: A Hybrid Cov-Attention and Cross-Layer Fusion Transformer Network for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-17, 2024.
DOI: 10.1109/TGRS.2024.3374081

## What Changed in This Fork

1. Separate .mat file loading
   The original code expected a single .mat file containing input, TR, and TE variables pre-split by the authors.
   This fork loads the standard separate data and ground-truth .mat files:
   - *_corrected.mat / *.mat -- hyperspectral image cube (H x W x C)
   - *_gt.mat -- ground-truth label map (H x W), where 0 = background and 1..N = class labels

2. Local train/test split
   The ground-truth map is now split into train and test masks inside the pipeline instead of relying
   on pre-made TR / TE masks. Two splitting modes are supported via CLI:
   - fixed -- a fixed number of samples per class (e.g., 200)
   - ratio -- a percentage of each class (e.g., 0.1)

3. Automatic dataset download
   Running demo.py or download_data.py will automatically download Indian Pines and Pavia University
   datasets from EHU's Hyperspectral Remote Sensing Scenes if the .mat files are not found locally.
   Houston 2013 may require manual download due to access restrictions on the UH server.

4. Plotting & visualization
   Added visualize.py -- a modular visualization module that produces:
   - RGB composite of the hyperspectral cube
   - Ground-truth label map
   - Class distribution bar chart (train/test split)
   - Mean spectral signatures per class
   - Training curves (loss + accuracy over epochs)
   - Prediction map after inference
   - Side-by-side ground-truth vs prediction
   - Confusion matrix heatmap
   - Per-class accuracy bar chart
   All figures are auto-saved to ./figures/<dataset>/. Use --no_plot to disable.

5. New command-line arguments
   --data_path            path to data .mat (auto-resolved by --dataset if empty)
   --gt_path              path to ground-truth .mat (auto-resolved by --dataset if empty)
   --split_mode {fixed,ratio}
   --train_samples        default 200
   --train_ratio          default 0.1
   --model_path           default ./log/model.pt
   --output_dir           default ./figures  -- where visualization PNGs are saved
   --no_plot              disable all plotting
   --channels_band        auto-falls-back to number of bands if left at 0

6. Added data_utils.py -- utility module for loading, normalization, mirroring, splitting, patch extraction.

7. Added download_data.py -- standalone dataset downloader script.

8. Added visualize.py -- modular plotting and visualization module.

9. Added requirements.txt, pyproject.toml, and .gitignore for better project hygiene.

Everything else (network architecture, training loop, metrics) is unchanged from the original.

## Requirements

### 1. Install PyTorch with CUDA 12.6

   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

### 2. Install remaining dependencies

   pip install -r requirements.txt

Dependencies: einops, scipy, scikit-learn, matplotlib, seaborn, numpy

## Dataset

### Automatic download

Download all datasets at once:
   python download_data.py

Download a specific dataset:
   python download_data.py --dataset Indian
   python download_data.py --dataset Pavia
   python download_data.py --dataset Houston

Specify a custom output directory:
   python download_data.py --dataset Indian --data_dir ./data

Note: Indian Pines and Pavia University are downloaded from
https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
Houston 2013 may require manual download from https://hyperspectral.ee.uh.edu/?page_id=459
due to access restrictions -- place Houston.mat and Houston_gt.mat into ./data/ manually.

Files are placed under ./data/ by default:

Dataset            | Data file                    | Ground-truth file          | Bands
-------------------|------------------------------|----------------------------|-------
Indian Pines       | Indian_pines_corrected.mat   | Indian_pines_gt.mat        | 200
Pavia University   | PaviaU.mat                   | PaviaU_gt.mat              | 103
Houston 2013       | Houston.mat                  | Houston_gt.mat             | 144

demo.py will also auto-download missing files when you specify --dataset
without explicit --data_path/--gt_path.

### Manual download

If automatic download fails, download from:
- Indian Pines & Pavia: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
- Houston 2013: https://hyperspectral.ee.uh.edu/?page_id=459

Place the .mat files in ./data/.

## Network

- Indian Pines & Houston 2013 -> vit_pytorch_indian_Houston.py
- Pavia University           -> vit_pytorch_pavia.py

## Visualization

All visualizations are generated automatically and saved as PNGs under ./figures/<dataset>/:

Figure               | When generated       | File
----------------------|---------------------|----------------------------
RGB composite         | Before training     | rgb_composite.png
Ground-truth map      | Before training     | ground_truth.png
Class distribution    | Before training     | class_distribution.png
Spectral signatures   | Before training     | spectral_signatures.png
Training curves       | After training      | training_curves.png
Prediction map        | After training/test | prediction_map.png
GT vs Prediction      | After training/test | gt_vs_prediction.png
Confusion matrix      | After training/test | confusion_matrix.png
Per-class accuracy    | After training/test | per_class_accuracy.png

Use --no_plot to skip all visualization, or --output_dir to change the save directory.

## Usage

When --data_path and --gt_path are omitted, they are auto-resolved from --dataset
and missing files are downloaded automatically.

-------------------------------Train--------------------------------------------------------------------------------------------

Indian Pines (auto-downloads if missing):
python demo.py --dataset='Indian' --epoches=1500 --patches=7 --band_patches=1 --mode='CAF' --weight_decay=5e-3 --channels_band=200 --split_mode=fixed --train_samples=200 --model_path=./log/IP.pt

Pavia University:
python demo.py --dataset='Pavia' --epoches=1680 --patches=7 --band_patches=1 --mode='CAF' --weight_decay=5e-3 --channels_band=103 --split_mode=fixed --train_samples=200 --model_path=./log/Pavia.pt

Houston:
python demo.py --dataset='Houston' --epoches=1500 --patches=7 --band_patches=1 --mode='CAF' --weight_decay=5e-3 --channels_band=144 --split_mode=fixed --train_samples=200 --model_path=./log/Houston.pt

Explicit paths can still be provided:
python demo.py --dataset='Indian' --data_path=./data/Indian_pines_corrected.mat --gt_path=./data/Indian_pines_gt.mat --epoches=1500 --patches=7 --band_patches=1 --mode='CAF' --weight_decay=5e-3 --channels_band=200 --split_mode=fixed --train_samples=200 --model_path=./log/IP.pt

-------------------------------Train--------------------------------------------------------------------------------------------

-------------------------------Test-----------------------------------------------------------------------------------------------

Indian Pines:
python demo.py --dataset='Indian' --flag_test=test --patches=7 --band_patches=1 --mode='CAF' --channels_band=200 --model_path=./log/IP.pt

Pavia University:
python demo.py --dataset='Pavia' --flag_test=test --patches=7 --band_patches=1 --mode='CAF' --channels_band=103 --model_path=./log/Pavia.pt

Houston:
python demo.py --dataset='Houston' --flag_test=test --patches=7 --band_patches=1 --mode='CAF' --channels_band=144 --model_path=./log/Houston.pt

-------------------------------Test-----------------------------------------------------------------------------------------------

## Project Structure

.
|-- demo.py                          # Main training/testing entry point
|-- data_utils.py                    # Data loading, splitting, patch extraction
|-- download_data.py                 # Automatic dataset downloader
|-- visualize.py                     # Plotting & visualization module
|-- vit_pytorch_indian_Houston.py    # CACFTNet model (Indian Pines, Houston)
|-- vit_pytorch_pavia.py            # CACFTNet model (Pavia University)
|-- requirements.txt                 # Python dependencies
|-- pyproject.toml                   # Project metadata
|-- .gitignore
|-- README.md
|-- README.txt
+-- data/                            # Dataset .mat files (auto-downloaded)

## Citation

If you use this code, please cite the original paper:

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

## Note

This fork is made available for convenience when working with publicly released .mat datasets that ship as separate image / ground-truth files. Credit and all academic rights belong to the original authors.