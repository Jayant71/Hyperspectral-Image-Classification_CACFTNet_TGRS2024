import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat, savemat
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os

from data_utils import (
    load_hyperspectral_data, normalize_by_band, mirror_hsi,
    split_train_test, choose_train_and_test_point,
    gain_neighborhood_pixel, gain_neighborhood_band,
    train_and_test_data, train_and_test_label
)
from download_data import ensure_dataset_available
from visualize import (
    plot_rgb_composite, plot_gt_map, plot_class_distribution,
    plot_band_spectra, plot_training_curves,
    plot_prediction_map, plot_gt_vs_prediction,
    plot_confusion_matrix, plot_per_class_accuracy,
    save_all_figures, _load_colormap
)

# -------------------------------------------------------------------------------
# Argument parser
parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston'], default='Indian', help='dataset name for color map')
parser.add_argument('--data_path', type=str, default='', help='path to data .mat file (auto-resolved by --dataset if empty)')
parser.add_argument('--gt_path', type=str, default='', help='path to ground-truth .mat file (auto-resolved by --dataset if empty)')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='ViT', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--test_freq', type=int, default=5, help='evaluation frequency')
parser.add_argument('--patches', type=int, default=7, help='spatial patch size')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='lr scheduler gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--channels_band', type=int, default=0, help='channels band')
parser.add_argument('--split_mode', choices=['fixed', 'ratio'], default='fixed', help='how to split train/test')
parser.add_argument('--train_samples', type=int, default=200, help='samples per class for fixed split')
parser.add_argument('--train_ratio', type=float, default=0.1, help='train ratio for ratio split')
parser.add_argument('--model_path', type=str, default='./log/model.pt', help='path to save/load model')
parser.add_argument('--output_dir', type=str, default='./figures', help='directory to save visualization figures')
parser.add_argument('--no_plot', action='store_true', help='disable all plotting/visualization')
args = parser.parse_args()

# -------------------------------------------------------------------------------
# import the correct ViT model
if args.dataset == 'Pavia':
    from vit_pytorch_pavia import ViT
else:
    from vit_pytorch_indian_Houston import ViT

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

# -------------------------------------------------------------------------------
# Metric utilities
class AvgrageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()

# -------------------------------------------------------------------------------
# Train / Valid / Test epochs
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre

def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return tar, pre

def test_epoch(model, test_loader, criterion, optimizer):
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        batch_pred = model(batch_data)
        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre

# -------------------------------------------------------------------------------
# Evaluation utils
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum_ = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        row_sum = np.sum(matrix[i, :])
        number += matrix[i, i]
        if row_sum > 0:
            AA[i] = matrix[i, i] / row_sum
        else:
            AA[i] = 0.0
        sum_ += row_sum * np.sum(matrix[:, i])
    total = np.sum(matrix)
    OA = number / total if total > 0 else 0.0
    nonzero_AA = AA[AA > 0]
    AA_mean = np.mean(nonzero_AA) if len(nonzero_AA) > 0 else 0.0
    pe = sum_ / (total ** 2) if total > 0 else 0.0
    Kappa = (OA - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    return OA, AA_mean, Kappa, AA

# -------------------------------------------------------------------------------
# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

# -------------------------------------------------------------------------------
# Auto-resolve data/gt paths from dataset name if not provided
DATASET_FILES = {
    'Indian': ('./data/Indian_pines_corrected.mat', './data/Indian_pines_gt.mat'),
    'Pavia': ('./data/PaviaU.mat', './data/PaviaU_gt.mat'),
    'Houston': ('./data/Houston.mat', './data/Houston_gt.mat'),
}

if not args.data_path or not args.gt_path:
    args.data_path, args.gt_path = DATASET_FILES[args.dataset]

# Auto-download data files if they don't exist yet
args.data_path, args.gt_path = ensure_dataset_available(args.dataset, data_dir=os.path.dirname(args.data_path) or './data')

# -------------------------------------------------------------------------------
# Load data from separate .mat files
print("Loading data from:", args.data_path)
print("Loading gt from:", args.gt_path)
input_img, gt, num_classes = load_hyperspectral_data(args.data_path, args.gt_path)
height, width, band = input_img.shape
print("height={}, width={}, band={}, num_classes={}".format(height, width, band, num_classes))

# Normalize band-wise
input_normalize = normalize_by_band(input_img)

# Build train/test split masks locally
if args.split_mode == 'fixed':
    TR, TE = split_train_test(gt, mode='fixed', train_samples=args.train_samples, seed=args.seed)
elif args.split_mode == 'ratio':
    TR, TE = split_train_test(gt, mode='ratio', train_ratio=args.train_ratio, seed=args.seed)

label = gt

# -------------------------------------------------------------------------------
# Colormap
color_matrix = _load_colormap('./data/AVIRIS_colormap.mat', num_classes)

# -------------------------------------------------------------------------------
# Dataset visualizations (before training)
if not args.no_plot:
    print("Generating dataset visualizations ...")
    fig_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(fig_dir, exist_ok=True)

    # RGB composite (pick reasonable bands depending on dataset)
    rgb_bands = {'Indian': (60, 30, 10), 'Pavia': (50, 30, 10), 'Houston': (40, 20, 5)}
    plot_rgb_composite(input_normalize,
                       bands=rgb_bands.get(args.dataset, (60, 30, 10)),
                       title=f"{args.dataset} RGB Composite",
                       save_path=os.path.join(fig_dir, "rgb_composite.png"))

    plot_gt_map(gt, num_classes, colormap=color_matrix,
                title=f"{args.dataset} Ground Truth",
                save_path=os.path.join(fig_dir, "ground_truth.png"))

    plot_class_distribution(gt, TR, TE, num_classes,
                            title=f"{args.dataset} Class Distribution (Train/Test)",
                            save_path=os.path.join(fig_dir, "class_distribution.png"))

    plot_band_spectra(input_normalize, gt, num_classes,
                       title=f"{args.dataset} Mean Spectral Signatures",
                       save_path=os.path.join(fig_dir, "spectral_signatures.png"))

    print("Dataset visualizations saved to:", fig_dir)

# -------------------------------------------------------------------------------
# Extract patches
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = \
    choose_train_and_test_point(TR, TE, label, num_classes)

mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)

x_train_band, x_test_band, x_true_band = train_and_test_data(
    mirror_image, band, total_pos_train, total_pos_test, total_pos_true,
    patch=args.patches, band_patch=args.band_patches)
y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)

# -------------------------------------------------------------------------------
# Build PyTorch loaders
x_train = torch.from_numpy(x_train_band.transpose(0, 2, 1)).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
Label_train = Data.TensorDataset(x_train, y_train)

x_test = torch.from_numpy(x_test_band.transpose(0, 2, 1)).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)
Label_test = Data.TensorDataset(x_test, y_test)

x_true = torch.from_numpy(x_true_band.transpose(0, 2, 1)).type(torch.FloatTensor)
y_true = torch.from_numpy(y_true).type(torch.LongTensor)
Label_true = Data.TensorDataset(x_true, y_true)

label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)
label_true_loader = Data.DataLoader(Label_true, batch_size=100, shuffle=False)

# -------------------------------------------------------------------------------
# Create model
model = ViT(
    image_size=args.patches,
    near_band=args.band_patches,
    num_patches=band,
    num_classes=num_classes,
    channels_band=args.channels_band if args.channels_band > 0 else band,
    dim=64,
    depth=5,
    heads=4,
    mlp_dim=8,
    dropout=0.1,
    emb_dropout=0.1,
    mode=args.mode
)
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)

# -------------------------------------------------------------------------------
# Test or Train
OA2, AA_mean2, Kappa2, AA2 = 0.0, 0.0, 0.0, 0.0

if args.flag_test == 'test':
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

    # output classification maps
    pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(total_pos_true.shape[0]):
        prediction_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i] + 1

    savemat('matrix.mat', {'P': prediction_matrix, 'label': label})

    # Evaluation visualizations
    if not args.no_plot:
        fig_dir = os.path.join(args.output_dir, args.dataset)
        os.makedirs(fig_dir, exist_ok=True)

        plot_prediction_map(prediction_matrix, gt, num_classes, colormap=color_matrix,
                            title=f"{args.dataset} Prediction (OA={OA2:.4f})",
                            save_path=os.path.join(fig_dir, "prediction_map.png"))

        plot_gt_vs_prediction(gt, prediction_matrix, num_classes, colormap=color_matrix,
                               title=f"{args.dataset} Ground Truth vs Prediction",
                               save_path=os.path.join(fig_dir, "gt_vs_prediction.png"))

        plot_confusion_matrix(tar_v, pre_v, num_classes,
                               title=f"{args.dataset} Confusion Matrix",
                               save_path=os.path.join(fig_dir, "confusion_matrix.png"))

        plot_per_class_accuracy(tar_v, pre_v, num_classes,
                                 title=f"{args.dataset} Per-Class Accuracy",
                                 save_path=os.path.join(fig_dir, "per_class_accuracy.png"))

        print("Evaluation visualizations saved to:", fig_dir)

elif args.flag_test == 'train':
    print("start training")
    tic = time.time()
    best_acc = 0.0
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    train_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(args.epoches):
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
              .format(epoch + 1, train_obj, train_acc))

        train_losses.append(float(train_obj))
        train_accs.append(float(train_acc))

        if (epoch % args.test_freq == 0) or (epoch == args.epoches - 1):
            model.eval()
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            if OA2 > best_acc:
                best_acc = OA2
                torch.save(model.state_dict(), args.model_path)
            print("Epoch: {:03d} test_acc: {:.4f}".format(epoch + 1, OA2))
            test_accs.append(float(OA2))
        else:
            test_accs.append(float('nan'))

        scheduler.step()

    toc = time.time()
    print("Running Time: {:.2f}".format(toc - tic))
    print("**************************************************")

    # Training visualizations
    if not args.no_plot:
        fig_dir = os.path.join(args.output_dir, args.dataset)
        os.makedirs(fig_dir, exist_ok=True)

        plot_training_curves(train_losses, train_accs, test_accs,
                              title=f"{args.dataset} Training Curves",
                              save_path=os.path.join(fig_dir, "training_curves.png"))

        # Final evaluation on test set with best model
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
        OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

        pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
        prediction_matrix = np.zeros((height, width), dtype=float)
        for i in range(total_pos_true.shape[0]):
            prediction_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i] + 1

        plot_prediction_map(prediction_matrix, gt, num_classes, colormap=color_matrix,
                            title=f"{args.dataset} Prediction (OA={OA2:.4f})",
                            save_path=os.path.join(fig_dir, "prediction_map.png"))

        plot_gt_vs_prediction(gt, prediction_matrix, num_classes, colormap=color_matrix,
                               title=f"{args.dataset} Ground Truth vs Prediction",
                               save_path=os.path.join(fig_dir, "gt_vs_prediction.png"))

        plot_confusion_matrix(tar_v, pre_v, num_classes,
                               title=f"{args.dataset} Confusion Matrix",
                               save_path=os.path.join(fig_dir, "confusion_matrix.png"))

        plot_per_class_accuracy(tar_v, pre_v, num_classes,
                                 title=f"{args.dataset} Per-Class Accuracy",
                                 save_path=os.path.join(fig_dir, "per_class_accuracy.png"))

        print("Training & evaluation visualizations saved to:", fig_dir)

# -------------------------------------------------------------------------------
# Final output
print("Final result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
print(AA2)
print("**************************************************")

if args.flag_test == 'train':
    print("Model saved to:", args.model_path)

print("Parameter:")
def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))
print_args(vars(args))