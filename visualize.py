"""
Visualization utilities for hyperspectral image classification.

Functions:
  - plot_rgb_composite       — show a 3-band RGB composite of the hyperspectral cube
  - plot_gt_map              — show the ground-truth label map
  - plot_prediction_map      — show the predicted classification map
  - plot_gt_vs_prediction    — side-by-side ground-truth vs prediction
  - plot_class_distribution  — bar chart of samples per class (train / test)
  - plot_confusion_matrix    — heatmap of the confusion matrix
  - plot_training_curves     — loss and accuracy curves over epochs
  - plot_band_spectra        — mean spectral signature per class
  - save_all_figures         — batch-save generated figures to disk
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _default_colormap(num_classes):
    """Generate a discrete colormap for *num_classes* + 1 (background)."""
    cmap = plt.cm.get_cmap("tab20", num_classes + 1)
    return [cmap(i) for i in range(num_classes + 1)]


def _load_colormap(path, num_classes):
    """Try loading an AVIRIS-style colormap .mat, else fall back to tab20."""
    if os.path.exists(path):
        from scipy.io import loadmat
        mat = loadmat(path)
        for k in mat:
            if k.startswith("__"):
                continue
            arr = mat[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 3:
                return arr
    return np.array(_default_colormap(num_classes))[:num_classes + 1]


# ---------------------------------------------------------------------------
# Dataset visualizations
# ---------------------------------------------------------------------------

def plot_rgb_composite(img, bands=(60, 30, 10), title="RGB Composite", save_path=None):
    """
    Plot an RGB composite from a hyperspectral image.

    Args:
        img: (H, W, C) array, assumed normalized to [0,1].
        bands: tuple of 3 band indices for R, G, B channels.
        title: plot title.
        save_path: if given, save figure to this path.
    """
    rgb = np.stack([img[:, :, bands[0]],
                    img[:, :, bands[1]],
                    img[:, :, bands[2]]], axis=-1)
    rgb = np.clip(rgb, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)
    return fig


def plot_gt_map(gt, num_classes, colormap=None, title="Ground Truth", save_path=None):
    """
    Plot the ground-truth label map.

    Args:
        gt: (H, W) integer label array (0=background).
        num_classes: number of classes.
        colormap: (num_classes+1, 3) RGB array, or None for auto.
        title: plot title.
        save_path: if given, save figure.
    """
    if colormap is None:
        cmap_vals = _default_colormap(num_classes)
        cmap_vals[0] = (0, 0, 0, 1)  # black for background
        colormap = np.array(cmap_vals)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(gt, cmap=mcolors.ListedColormap(colormap),
                   vmin=0, vmax=num_classes)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=range(num_classes + 1))
    cbar.ax.set_yticklabels(["BG"] + [str(i) for i in range(1, num_classes + 1)])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)
    return fig


def plot_class_distribution(gt, TR=None, TE=None, num_classes=None, title="Class Distribution",
                            save_path=None):
    """
    Bar chart of samples per class for train and test splits.

    Args:
        gt: (H, W) ground truth.
        TR: (H, W) train mask, or None.
        TE: (H, W) test mask, or None.
        num_classes: number of classes (auto from gt if None).
        title: plot title.
        save_path: if given, save figure.
    """
    if num_classes is None:
        num_classes = int(np.max(gt))

    classes = list(range(1, num_classes + 1))
    gt_counts = [np.sum(gt == c) for c in classes]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if TR is not None and TE is not None:
        train_counts = [np.sum(TR == c) for c in classes]
        test_counts = [np.sum(TE == c) for c in classes]
        x = np.arange(len(classes))
        w = 0.25
        ax.bar(x - w, train_counts, w, label="Train", color="#4C72B0")
        ax.bar(x, test_counts, w, label="Test", color="#DD8452")
        ax.bar(x + w, gt_counts, w, label="Total", color="#55A868")
        ax.set_xticks(x)
    else:
        ax.bar(classes, gt_counts, color="#4C72B0")

    ax.set_xticklabels([str(c) for c in classes])
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of pixels")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)
    return fig


def plot_band_spectra(img, gt, num_classes=None, title="Mean Spectral Signatures",
                       save_path=None):
    """
    Plot mean spectral signature per class.

    Args:
        img: (H, W, C) hyperspectral image.
        gt: (H, W) ground-truth labels.
        num_classes: auto from gt if None.
        title: plot title.
        save_path: if given, save figure.
    """
    if num_classes is None:
        num_classes = int(np.max(gt))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for c in range(1, num_classes + 1):
        mask = (gt == c)
        if np.sum(mask) == 0:
            continue
        mean_spectrum = img[mask].mean(axis=0)
        ax.plot(mean_spectrum, label=f"Class {c}")

    ax.set_xlabel("Band index")
    ax.set_ylabel("Reflectance (normalized)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Training visualizations
# ---------------------------------------------------------------------------

def plot_training_curves(train_losses, train_accs, test_accs=None,
                          title="Training Curves", save_path=None):
    """
    Plot loss and accuracy curves over epochs.

    Args:
        train_losses: list of training losses per epoch.
        train_accs: list of training accuracies per epoch.
        test_accs: list of test accuracies (same length, or None).
        title: figure title.
        save_path: if given, save figure.
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_accs, "b-", label="Train Acc")
    if test_accs is not None:
        ax2.plot(epochs, test_accs, "r-", label="Test Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Evaluation visualizations
# ---------------------------------------------------------------------------

def plot_prediction_map(prediction, gt, num_classes, colormap=None,
                         title="Prediction Map", save_path=None):
    """
    Plot the predicted classification map.

    Args:
        prediction: (H, W) predicted labels (1-based or 0 for background).
        gt: (H, W) ground-truth labels.
        num_classes: number of classes.
        colormap: optional (num_classes+1, 3) colormap.
        title: plot title.
        save_path: if given, save figure.
    """
    if colormap is None:
        cmap_vals = _default_colormap(num_classes)
        cmap_vals[0] = (0, 0, 0, 1)
        colormap = np.array(cmap_vals)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(prediction, cmap=mcolors.ListedColormap(colormap),
                   vmin=0, vmax=num_classes)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=range(num_classes + 1))
    cbar.ax.set_yticklabels(["BG"] + [str(i) for i in range(1, num_classes + 1)])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)
    return fig


def plot_gt_vs_prediction(gt, prediction, num_classes, colormap=None,
                           title="Ground Truth vs Prediction", save_path=None):
    """
    Side-by-side ground truth and predicted classification maps.

    Args:
        gt: (H, W) ground-truth labels.
        prediction: (H, W) predicted labels.
        num_classes: number of classes.
        colormap: optional colormap array.
        title: figure title.
        save_path: if given, save figure.
    """
    if colormap is None:
        cmap_vals = _default_colormap(num_classes)
        cmap_vals[0] = (0, 0, 0, 1)
        colormap = np.array(cmap_vals)

    cmap_obj = mcolors.ListedColormap(colormap)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    im1 = ax1.imshow(gt, cmap=cmap_obj, vmin=0, vmax=num_classes)
    ax1.set_title("Ground Truth")
    ax1.axis("off")

    im2 = ax2.imshow(prediction, cmap=cmap_obj, vmin=0, vmax=num_classes)
    ax2.set_title("Prediction")
    ax2.axis("off")

    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.04,
                        ticks=range(num_classes + 1))
    cbar.ax.set_yticklabels(["BG"] + [str(i) for i in range(1, num_classes + 1)])

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)
    return fig


def plot_confusion_matrix(tar, pre, num_classes, normalize=True,
                           title="Confusion Matrix", save_path=None):
    """
    Plot a confusion matrix as a heatmap.

    Args:
        tar: 1-D array of true labels.
        pre: 1-D array of predicted labels.
        num_classes: number of classes.
        normalize: if True, show row-normalized percentages.
        title: figure title.
        save_path: if given, save figure.
    """
    labels = list(range(num_classes))
    cm = confusion_matrix(tar, pre, labels=labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm.astype(float) / row_sums * 100
        fmt = ".1f"
        cbar_label = "Percentage (%)"
    else:
        cm_display = cm
        fmt = "d"
        cbar_label = "Count"

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={"label": cbar_label})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)
    return fig


def plot_per_class_accuracy(tar, pre, num_classes, title="Per-Class Accuracy",
                             save_path=None):
    """
    Bar chart of accuracy for each class.

    Args:
        tar: 1-D array of true labels.
        pre: 1-D array of predicted labels.
        num_classes: number of classes.
        title: figure title.
        save_path: if given, save figure.
    """
    labels = list(range(num_classes))
    cm = confusion_matrix(tar, pre, labels=labels)
    class_acc = []
    for i in range(num_classes):
        total = cm[i].sum()
        class_acc.append(cm[i, i] / total * 100 if total > 0 else 0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bars = ax.bar(labels, class_acc, color="#4C72B0")
    ax.axhline(y=np.mean(class_acc), color="r", linestyle="--",
               label=f"Mean AA = {np.mean(class_acc):.1f}%")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.legend()

    for bar, acc in zip(bars, class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def save_all_figures(figures, output_dir="./figures", prefix=""):
    """
    Save a dict of {name: Figure} to disk.

    Args:
        figures: dict mapping name -> matplotlib Figure.
        output_dir: directory to save into.
        prefix: optional filename prefix.
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, fig in figures.items():
        path = os.path.join(output_dir, f"{prefix}{name}.png")
        if fig is not None:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
    plt.close("all")