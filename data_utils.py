import numpy as np
from scipy.io import loadmat


def load_hyperspectral_data(data_path, gt_path):
    """
    Load hyperspectral image and ground truth from separate .mat files.
    Auto-detects variable names by shape.

    Returns:
        input_img: (H, W, C) hyperspectral image array
        gt: (H, W) ground truth label array
        num_classes: int, number of classes in gt
    """
    data_mat = loadmat(data_path)
    gt_mat = loadmat(gt_path)

    def _is_variable(key):
        return not key.startswith("__")

    data_keys = [k for k in data_mat if _is_variable(k)]
    gt_keys = [k for k in gt_mat if _is_variable(k)]

    if len(data_keys) == 0:
        raise ValueError(f"No data variable found in {data_path}")
    if len(gt_keys) == 0:
        raise ValueError(f"No gt variable found in {gt_path}")

    # Find GT variable (2D array) first
    gt_var = None
    for k in gt_keys:
        arr = np.array(gt_mat[k])
        if arr.ndim == 2:
            gt_var = arr
            break
    if gt_var is None:
        raise ValueError(f"Could not find 2D ground truth array in {gt_path}")

    gt = gt_var.astype(np.int64)
    H_gt, W_gt = gt.shape

    # Find data variable whose spatial dimensions match GT
    data_var = None
    for k in data_keys:
        arr = np.array(data_mat[k])
        if arr.ndim < 3:
            continue

        # Prefer (H, W, C) where first two dims match GT spatial dims
        if (arr.shape[0] == H_gt and arr.shape[1] == W_gt) or \
           (arr.shape[0] == W_gt and arr.shape[1] == H_gt):
            data_var = arr
            break
        # Try (C, H, W) where last two dims match GT spatial dims
        if arr.ndim >= 3 and \
           ((arr.shape[1] == H_gt and arr.shape[2] == W_gt) or
            (arr.shape[1] == W_gt and arr.shape[2] == H_gt)):
            # Transpose (C, H, W) -> (H, W, C)
            data_var = arr.transpose(1, 2, 0)
            break

    if data_var is None:
        # Fallback: pick the largest 3D+ array and best-effort orient
        best = None
        best_size = 0
        for k in data_keys:
            arr = np.array(data_mat[k])
            if arr.ndim >= 3 and arr.size > best_size:
                best_size = arr.size
                best = arr
        if best is None:
            raise ValueError(f"Could not find hyperspectral data array (3D+) in {data_path}")
        data_var = best

    input_img = np.array(data_var)

    # Ensure data is (H, W, C)
    if input_img.ndim == 2:
        input_img = input_img[:, :, np.newaxis]

    num_classes = int(np.max(gt))
    if num_classes == 0:
        raise ValueError("Ground truth has no labeled classes (only background/zeros).")
    return input_img, gt, num_classes


def normalize_by_band(img):
    """
    Normalize image band-wise to [0, 1].
    img: (H, W, C)
    Returns: (H, W, C) normalized image
    """
    img_norm = np.zeros_like(img, dtype=np.float64)
    for i in range(img.shape[2]):
        band = img[:, :, i]
        min_val = np.min(band)
        max_val = np.max(band)
        if max_val - min_val > 0:
            img_norm[:, :, i] = (band - min_val) / (max_val - min_val)
        else:
            img_norm[:, :, i] = band
    return img_norm.astype(np.float32)


def mirror_hsi(height, width, band, input_normalize, patch=5):
    """
    Mirror pad the image so patches can be extracted at borders.
    Returns padded image of shape (H+pad*2, W+pad*2, band)
    """
    padding = patch // 2
    mirror_image = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    # center
    mirror_image[padding:(padding + height), padding:(padding + width), :] = input_normalize
    # left mirror
    for i in range(padding):
        mirror_image[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    # right mirror
    for i in range(padding):
        mirror_image[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    # top mirror
    for i in range(padding):
        mirror_image[i, :, :] = mirror_image[padding * 2 - i - 1, :, :]
    # bottom mirror
    for i in range(padding):
        mirror_image[height + padding + i, :, :] = mirror_image[height + padding - 1 - i, :, :]
    return mirror_image


def split_train_test(gt, mode="fixed", train_samples=None, train_ratio=None, seed=None):
    """
    Split ground truth into train and test masks.

    Args:
        gt: (H, W) ground truth label array (0=background, 1..num_classes)
        mode: 'fixed' for fixed samples per class, 'ratio' for ratio split
        train_samples: int, number of samples per class for training (if mode='fixed')
        train_ratio: float, ratio of samples per class for training (if mode='ratio')
        seed: random seed

    Returns:
        TR: (H, W) train mask (same values as gt for train samples, 0 otherwise)
        TE: (H, W) test mask
    """
    if seed is not None:
        np.random.seed(seed)

    height, width = gt.shape
    num_classes = int(np.max(gt))
    TR = np.zeros((height, width), dtype=np.int64)
    TE = np.zeros((height, width), dtype=np.int64)

    for c in range(1, num_classes + 1):
        coords = np.argwhere(gt == c)
        n_total = coords.shape[0]
        if n_total == 0:
            continue
        indices = np.random.permutation(n_total)

        if mode == "fixed":
            n_train = min(train_samples, n_total - 1)
            if n_train < 1:
                n_train = 1
        elif mode == "ratio":
            n_train = max(1, int(np.floor(n_total * train_ratio)))
            if n_train >= n_total:
                n_train = max(1, n_total - 1)
        else:
            raise ValueError(f"Unknown split mode: {mode}")

        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        for idx in train_idx:
            TR[coords[idx, 0], coords[idx, 1]] = c
        for idx in test_idx:
            TE[coords[idx, 0], coords[idx, 1]] = c

    return TR, TE


def choose_train_and_test_point(train_data, test_data, true_data, num_classes):
    """
    Collect coordinates for train, test, and true labels.
    Returns arrays of shape (N, 2).
    """
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}

    # train data
    for i in range(num_classes):
        each_class = np.argwhere(train_data == (i + 1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]
    total_pos_train = total_pos_train.astype(int)

    # test data
    for i in range(num_classes):
        each_class = np.argwhere(test_data == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]
    total_pos_test = total_pos_test.astype(int)

    # true data (all labeled pixels)
    for i in range(num_classes + 1):
        each_class = np.argwhere(true_data == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes + 1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    """
    Extract a patch centered at point[i].
    """
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    """
    Extract neighborhood bands around each spatial patch.
    """
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch * patch * band_patch, band), dtype=float)
    # center region
    x_train_band[:, nn * patch * patch:(nn + 1) * patch * patch, :] = x_train_reshape
    # left mirror
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = x_train_reshape[:, :, band - i - 1:]
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = x_train_reshape[:, :, :band - i - 1]
        else:
            x_train_band[:, i:(i + 1), :(nn - i)] = x_train_reshape[:, 0:1, (band - nn + i):]
            x_train_band[:, i:(i + 1), (nn - i):] = x_train_reshape[:, 0:1, :(band - nn + i)]
    # right mirror
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, :band - i - 1] = x_train_reshape[:, :, i + 1:]
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, band - i - 1:] = x_train_reshape[:, :, :i + 1]
        else:
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train_reshape[:, 0:1, :(i + 1)]
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train_reshape[:, 0:1, (i + 1):]
    return x_train_band


def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    """
    Build 4D patch tensors for train, test, and all labeled pixels.
    """
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k, :, :, :] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)

    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    return x_train_band, x_test_band, x_true_band


def train_and_test_label(number_train, number_test, number_true, num_classes):
    """
    Generate label vectors matching the extracted patches.
    """
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for _ in range(number_train[i]):
            y_train.append(i)
        for _ in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes + 1):
        for _ in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    return y_train, y_test, y_true
