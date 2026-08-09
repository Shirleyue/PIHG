'''
test PIHG to generate the metric results.
'''

from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from collections import defaultdict
import torch.nn as nn
from PIL import Image
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../')
sys.path.append(project_root)

from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_tor
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_tor

from cnn_rgat_modules import Phy_CNN_Graph
from SpectrumNet.load_data import SpectrumDataset, SpectrumDatasetField


import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def fig_show_save(epoch, rss_gt, rcv_rss, building, folder_path, current, type='valid'):
    """
    Plot a 1x3 panel: building | recovered RSS | ground-truth RSS.
    - Both RSS maps share the same vmin/vmax for easier comparison.
    - Each subplot gets its own colorbar.
    """
    rss_gt  = np.asarray(rss_gt.detach().cpu())
    rcv_rss = np.asarray(rcv_rss.detach().cpu())
    building = np.asarray(building.detach().cpu())

    # Shared color scale for the two RSS maps
    vmin = min(np.nanmin(rcv_rss), np.nanmin(rss_gt))
    vmax = max(np.nanmax(rcv_rss), np.nanmax(rss_gt))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    ax = axes[0]
    im0 = ax.imshow(building, cmap='gray', interpolation='nearest')
    ax.set_title('Building', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    cbar0 = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    cbar0.ax.tick_params(labelsize=10)

    ax = axes[1]
    im1 = ax.imshow(rcv_rss, vmin=vmin, vmax=vmax, cmap='viridis', interpolation='nearest')
    ax.set_title('Recovered RSS', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=10)

    ax = axes[2]
    im2 = ax.imshow(rss_gt, vmin=vmin, vmax=vmax, cmap='viridis', interpolation='nearest')
    ax.set_title('Ground Truth RSS', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=10)

    save_dir = os.path.join(folder_path, type)
    os.makedirs(save_dir, exist_ok=True)
    save_name = f'epoch_{epoch+1}_panel_{current}.png'
    fig.savefig(
        os.path.join(save_dir, save_name),
        format='png',
        dpi=800,
        pad_inches=0,
        transparent=False
    )
    plt.close(fig)


def evaluate_metrics(target, pred):
    criterion = nn.MSELoss()
    ssim_values = ssim_tor(pred, target)
    psnr_values = psnr_tor(pred, target, data_range=1.0)

    mse_values = criterion(pred, target)
    rmse_values = torch.sqrt(criterion(pred, target))
    nmse_values = criterion(pred, target) / criterion(target, 0 * target)

    rmse_values = rmse_values.cpu().numpy()* target.shape[0]
    ssim_values = ssim_values.cpu().numpy()* target.shape[0]
    psnr_values = psnr_values.cpu().numpy()* target.shape[0]
    mse_values =   mse_values.cpu().numpy()* target.shape[0]
    nmse_values = nmse_values.cpu().numpy()* target.shape[0]

    return rmse_values, ssim_values, psnr_values, mse_values, nmse_values


def print_metrics(metrics, epoch_samples, phase):
    outputs1 = []
    outputs2 = []
    for k in metrics.keys():
        outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs1)))

def calc_loss_test_include_building(pred1, pred2, target, metrics, error="MSE"):

    ssim1 = ssim(pred1, target)
    psnr1 = psnr(pred1, target)

    criterion = nn.MSELoss()
    loss = criterion(pred1, target)
    loss1 = criterion(pred1, target) / criterion(target, 0 * target)
    loss2 = torch.sqrt(criterion(pred1, target))

    metrics['mse'] += loss.data.cpu().numpy() * target.size(0)
    metrics['nmse'] += loss1.data.cpu().numpy() * target.size(0)
    metrics['rmse'] += loss2.data.cpu().numpy() * target.size(0)
    metrics['ssim'] += ssim1.data.cpu().numpy() * target.size(0)
    metrics['psnr'] += psnr1.data.cpu().numpy() * target.size(0)


    return [loss1, loss2]

def calc_loss_test(pred1, pred2, target, metrics, error="MSE", mask=None):
    """
    Compute prediction error metrics over the unmasked positions only.
    Args:
        pred1: first prediction (torch.Tensor)
        target: ground truth (torch.Tensor)
        metrics: dict accumulating the metrics
        error: error type (default "MSE")
        mask: mask matrix (1 = keep, 0 = masked out)
    Returns:
        [nmse_loss, rmse_loss]
    """
    mask = mask.bool() if mask.dtype != torch.bool else mask

    pred1_masked = pred1[mask]
    target_masked = target[mask]

    if pred1_masked.numel() == 0:
        print("pred1_masked is an empty tensor")
        mse_loss = torch.tensor(0.0, device=pred1.device)
        nmse_loss = torch.tensor(0.0, device=pred1.device)
        rmse_loss = torch.tensor(0.0, device=pred1.device)
    else:
        criterion = nn.MSELoss()
        mse_loss = criterion(pred1_masked, target_masked)
        denominator = criterion(target_masked, torch.zeros_like(target_masked))
        nmse_loss = mse_loss / denominator if denominator != 0 else torch.tensor(float('nan'), device=pred1.device)
        rmse_loss = torch.sqrt(mse_loss) if mse_loss >= 0 else torch.tensor(float('nan'), device=pred1.device)

    metrics['mse'] += mse_loss.item() if pred1_masked.numel() > 0 else 0.0
    metrics['nmse'] += nmse_loss.item() if pred1_masked.numel() > 0 and not torch.isnan(nmse_loss) else 0.0
    metrics['rmse'] += rmse_loss.item() if pred1_masked.numel() > 0 and not torch.isnan(rmse_loss) else 0.0

    return [nmse_loss, rmse_loss]



def print_metrics_test(metrics, epoch_samples, error):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format("Test" + " " + error, ", ".join(outputs)))

def test_loss(model, error="MSE", dataset="coarse", folder_path=''):
    # dataset is "coarse" or "fine".
    since = time.time()
    model.eval()  # Set model to evaluate mode
    metrics = defaultdict(float)
    epoch_samples = 0
    if dataset == "coarse":
        for inputs, rss_gt, graph, *_ in DataLoader(Radio_test, batch_size=batch_size, shuffle=False, num_workers=0):

            inputs = inputs.to(device)
            targets = rss_gt.to(device)

            # do not track history if only in train
            with torch.set_grad_enabled(False):
                _, outputs1 = model(inputs, graph)
                outputs2 = outputs1
                [loss1, loss2] = calc_loss_test_include_building(outputs1, outputs2, targets, metrics, error)
                epoch_samples += inputs.size(0)
                building = inputs[:,0:1,:,:]

        print('epoch_samples:', epoch_samples)
    elif dataset == "fine":
        for inputs, targets, samples in DataLoader(Radio_test, batch_size=batch_size, shuffle=False, num_workers=0):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # do not track history if only in train
            with torch.set_grad_enabled(False):
                [outputs1, outputs2] = model(inputs)
                [loss1, loss2] = calc_loss_test(outputs1, outputs2, targets, metrics, error)
                epoch_samples += inputs.size(0)
    print_metrics_test(metrics, epoch_samples, error)
    #  --- save ---  #
    outputs = []
    with open(folder_path + 'test.txt', 'a') as f:
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        f.write(', '.join(outputs))
        f.write('\n')

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == "__main__":

    batch_size = 1
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    isGraph = 1
    scen = 'area'

    data_name = 'SpectrumNet'
    graph_name='disdepthAdj_32'
    rss_name='splitRSS_32'
    noise_std = None


    if data_name == 'SpectrumNet':
        Radio_test = SpectrumDatasetField("./dataset/SpectrumNet/area _test.txt", div_block=True, load_W=False,
                                          few_shot_ratio=1,device=device, noise_std=noise_std)

        print('Radio_test len:', len(Radio_test))


    model_name = 'PIHG'
    train_scen = 'area '
    if isGraph:
        folder_path = 'results/' + data_name + '/' + model_name + '/' + scen + '/'
        train_folder_path = 'results/' + data_name + '/' + model_name + '/' + train_scen + '/' + 'train/'
        pt_path = None

    else:
        folder_path = 'results/' + data_name + '/' + model_name + '/' + scen + '/' + 'pretrain/'
        train_folder_path = 'results/' + data_name + '/' + model_name + '/' + train_scen + '/' + 'pretrain/'
        pt_path = None

    model = Phy_CNN_Graph(in_channels=4,
                    out_channels=2,
                    device=device,
                    graph_in_dims=2,
                    node_type_dim=3,
                    graph_hidden_channels=32,
                    graph_out_channels=1,
                    graph_num_relations=6,
                    num_node_types=3,
                    graph_num_heads=2,
                    pt_path=pt_path,
                    isGraph=isGraph)

    model.to(device)

    model.load_state_dict(torch.load(train_folder_path + 'best_model.pt'), strict=False)
    model.to(device)
    os.makedirs(folder_path, exist_ok=True)
    test_loss(model, error="MSE", folder_path=folder_path)
