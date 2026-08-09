import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import defaultdict
import torch.nn as nn
import time
import os
import copy
import sys
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../')
sys.path.append(project_root)
from SpectrumNet.load_data import SpectrumDataset, SpectrumDatasetField
from cnn_rgat_modules import Phy_CNN_Graph
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from phy_vie_fft import WKernelCache, incident_from_total_fft

def print_metrics(metrics, epoch_samples, phase):
    outputs1 = []
    outputs2 = []
    for k in metrics.keys():
        outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs1)))

def calc_loss(pred, target, metrics):
    loss = CRITERION(pred, target)
    metrics['loss'] += loss.detach().item() * target.size(0)

    return loss


# single shared criterion to avoid recreating it every call
CRITERION = nn.MSELoss()


def move_graph_to_device(graph, device):
    """Move all tensors inside a graph dict to `device` in-place.
    Expects graph to be a dict mapping idx -> block-dict.
    """
    if graph is None:
        return
    for k in list(graph.keys()):
        block = graph[k]
        for name in ('node_obs_axis', 'node_freq', 'node_type_ids', 'edge_index', 'edge_type', 'tx_img', 'building'):
            if name in block and isinstance(block[name], torch.Tensor):
                try:
                    block[name] = block[name].to(device, non_blocking=True)
                except Exception:
                    block[name] = block[name].to(device)


def handle_plateau(optimizer, scheduler, backup_scheduler):
    print("Applying plateau emergency measures")
    old_lr = optimizer.param_groups[0]['lr']
    # Bump the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = min(param_group['lr'] * 2, 1e-3)

    # Reset the schedulers
    if hasattr(backup_scheduler, '_reset'):
        backup_scheduler._reset()
    if hasattr(scheduler, 'last_epoch'):
        scheduler.last_epoch = -1

def batch_incident_from_total_einsum(E_total_complex, W, chi):
    """
    Batched version based on einsum; keeps gradients.
    """
    batch_size, _, Ny, Nx = E_total_complex.shape
    N = Ny * Nx

    E_tot_flat = E_total_complex.reshape(batch_size, N)  # (batch_size, N)

    chi_E_tot = chi * E_tot_flat  # (batch_size, N)

    # Batched matrix product via einsum: bij,bj->bi
    W_chi_E_tot = torch.einsum('bij,bj->bi', W, chi_E_tot)

    E_inc_rec_flat = E_tot_flat + W_chi_E_tot  # (batch_size, N)

    # Reshape and split into two channels, keeping gradients
    E_inc_rec = E_inc_rec_flat.reshape(batch_size, Ny, Nx)
    E_inc_rec_real = E_inc_rec.real.unsqueeze(1)
    E_inc_rec_imag = E_inc_rec.imag.unsqueeze(1)
    E_inc_rec_2channel = torch.cat([E_inc_rec_real, E_inc_rec_imag], dim=1).float()
    return E_inc_rec_2channel


def batch_incident_from_total_bmm(E_tot_bchw, W_bnn, chi_bn):
    """
    E_tot_bchw: (B,1,H,W) complex
    W_bnn:      (B,N,N)   complex   # per-sample W
    chi_bn:     (B,N)     complex
    Returns:    (B,2,H,W) float (real/imag channels)
    """
    B, _, H, W = E_tot_bchw.shape
    N = H * W

    E_tot = E_tot_bchw.view(B, N)           # (B,N) complex
    v = chi_bn * E_tot                      # (B,N) complex

    # y = W @ v, computed as v @ W^T with bmm to avoid a (B,N,N) intermediate
    y = torch.bmm(v.unsqueeze(1), W_bnn.transpose(1,2)).squeeze(1)  # (B,N)

    E_inc = E_tot + y                       # (B,N)
    E_inc = E_inc.view(B, 1, H, W)

    E2 = torch.cat([E_inc.real, E_inc.imag], dim=1).float()
    return E2


def batch_incident_from_total_fft(E_tot_bchw, chi_bn, f_hz, kernel_cache):
    """
    Drop-in replacement for batch_incident_from_total_bmm that does not need the (B,N,N) W.

    W depends only on pairwise distances and the grid is regular => W is a BTTB matrix
    => W @ v is a 2D convolution, computed exactly via circulant embedding + FFT. On a
    128x128 grid the kernel spectrum is only 256x256 complex values (0.5MB), replacing a
    16384x16384 complex128 matrix (4 GB per frequency). See phy_vie_fft.py.
    """
    return incident_from_total_fft(E_tot_bchw, chi_bn, f_hz, kernel_cache)

def batch_incident_from_tot(E_tot_bchw, chi_bn):
    """
    chi_bn * E_tot
    Returns: (B,2,H,W) float
    """
    B, _, H, W = E_tot_bchw.shape
    N = H * W

    E_flat = E_tot_bchw.view(B, N)
    result_flat = chi_bn * E_flat

    real = result_flat.real.view(B, 1, H, W)
    imag = result_flat.imag.view(B, 1, H, W)

    return torch.cat([real, imag], dim=1).float()

def train_model(model, dataloaders, optimizer, scheduler, backup_scheduler, folder_path, device, num_epochs=100):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    plateau_count = 0
    backup_scheduler.just_triggered = False

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        #  ---- training phase ---- #
        phase = 'train'
        model.train()
        metrics = defaultdict(float)
        total_samples = 0

        for inputs, targets, graph, _, _, _ in dataloaders[phase]:
            inputs = inputs.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs1, outputs = model(inputs, graph)
                loss = calc_loss(outputs, targets, metrics)

                loss.backward()
                optimizer.step()
            total_samples += inputs.size(0)
        avg_train_loss = metrics['loss'] / total_samples
        print_metrics(metrics, total_samples, phase)

        with open(folder_path + 'train.txt', 'a') as f:
            f.write(f'epoch:{epoch},{phase} loss:{avg_train_loss}\n')

        #  ---- validation phase ---- #
        phase = 'val'
        model.eval()
        val_metrics = defaultdict(float)
        total_samples = 0
        with torch.no_grad():
            for inputs, targets, graph, _, _, _ in dataloaders[phase]:
                inputs, targets = inputs.float().to(device), targets.to(device)
                move_graph_to_device(graph, device)
                outputs1, outputs = model(inputs, graph)
                loss = calc_loss(outputs, targets, val_metrics)

                total_samples += inputs.size(0)
        avg_val_loss = val_metrics['loss'] / total_samples
        print_metrics(val_metrics, total_samples, phase)

        with open(folder_path + 'train.txt', 'a') as f:
            f.write(f'epoch:{epoch},{phase} loss:{avg_val_loss}\n')

        scheduler.step()

        if avg_val_loss < best_loss:
            print("Saving best model...")
            with open(folder_path + 'result.txt', 'a') as f:
                f.write(f'Epoch {epoch+1}/{num_epochs} | Saving best model...\n' )
            best_loss = avg_val_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, os.path.join(folder_path, 'best_model.pt'))


        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {avg_train_loss:.6f} | '
              f'Val Loss: {avg_val_loss:.6f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        with open(folder_path + 'result.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {avg_train_loss:.6f} | '
              f'Val Loss: {avg_val_loss:.6f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}\n')

    print(f"Best val loss: {best_loss:.6f}")
    model.load_state_dict(best_model_wts)
    return model

def pretrain_model(model, dataloaders, optimizer, scheduler, backup_scheduler, folder_path, device, num_epochs=100,
                   kernel_cache=None, lamda=1.0):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    plateau_count = 0
    backup_scheduler.just_triggered = False

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        start = time.time()

        #  ---- training phase ---- #
        phase = 'train'
        model.train()
        metrics = defaultdict(float)
        total_samples = 0

        # The dataset returns (inputs, rss_map, graph, chi, f_hz, W); W is always an empty
        # tensor (load_W=False) because W @ v in the physics-consistency loss is computed
        # on the fly from the FFT kernel, which only needs f_hz.
        for batch_idx, (inputs, targets, graph, chi, f_hz, _) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).float()
            chi = chi.to(device, non_blocking=True).to(torch.complex64)
            move_graph_to_device(graph, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(phase == 'train'):
                outputs1, outputs2 = model(inputs, graph)
                E_total_complex = outputs1[:, 0:1, :, :] + 1j*outputs1[:, 1:2, :, :]  # (B,1,128,128)
                E_inc = torch.cat([inputs[:, 2:3,:,:], inputs[:, 3:4,:,:]], dim=1)
                E_inc_rec_2channel = batch_incident_from_total_fft(
                    E_total_complex, chi, f_hz, kernel_cache)
                loss1 = calc_loss(E_inc_rec_2channel, E_inc, metrics)
                loss2 = calc_loss(outputs2, targets, metrics)
                loss = loss1 + lamda * loss2
                loss.backward()
                optimizer.step()

            total_samples += inputs.size(0)
        scheduler.step()
        avg_train_loss = metrics['loss'] / total_samples
        print_metrics(metrics, total_samples, phase)

        with open(folder_path + 'train.txt', 'a') as f:
            f.write(f'epoch:{epoch},{phase} loss:{avg_train_loss}\n')

        #  ---- validation phase ---- #
        phase = 'val'
        model.eval()
        val_metrics = defaultdict(float)
        total_samples = 0
        with torch.no_grad():
            for inputs, targets, graph,_, _, _ in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True).float()
                targets = targets.to(device, non_blocking=True).float()
                move_graph_to_device(graph, device)
                _, outputs2 = model(inputs, graph)
                # Validation measures the error between the estimated total field and the
                # ground-truth received signal strength.
                loss = calc_loss(outputs2, targets, val_metrics)
                total_samples += inputs.size(0)
        avg_val_loss = val_metrics['loss'] / total_samples
        print_metrics(val_metrics, total_samples, phase)

        with open(folder_path + 'train.txt', 'a') as f:
            f.write(f'epoch:{epoch},{phase} loss:{avg_val_loss}\n')

        if avg_val_loss < best_loss:
            print("Saving best model...")
            with open(folder_path + 'result.txt', 'a') as f:
                f.write(f'Epoch {epoch+1}/{num_epochs} | Saving best model...\n' )
            best_loss = avg_val_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, os.path.join(folder_path, 'best_model.pt'))
            plateau_count = 0
        else:
            plateau_count += 1

        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {avg_train_loss:.6f} | '
              f'Val Loss: {avg_val_loss:.6f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        with open(folder_path + 'result.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {avg_train_loss:.6f} | '
              f'Val Loss: {avg_val_loss:.6f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}\n')

    print(f"Best val loss: {best_loss:.6f}")
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    batch_size = 1

    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    args = {
        # dataset settings
        'dx': 10.0,
        'dy': 10.0,
        'Nx': 128,  # block size
        'Ny': 128,
        'origin':(0.0, 0.0),
        'padding_mode': 'embed',

        # --- VIE physical discretisation: must match the one used to generate E_inc / chi ---
        # The incident-field directory in the dataset is incField_10.0/, i.e. it was produced
        # by gen_incident_field.py with dx=dy=10.0. The W kernel must use the same spacing,
        # otherwise the physics-consistency loss and the input E_inc live on different grids.
        'phy_dx': 10.0,
        'phy_dy': 10.0,
        'phy_Nx': 128,
        'phy_Ny': 128,
        'f_mhz_lst': [150, 1500, 1700, 3500, 22000],
        'lamda': 0.01,  # weight of the physics-consistency loss
    }
    args = SimpleNamespace(**args)

    # FFT kernel cache for W: 5 frequencies x (256x256) complex64 ~= 2.5MB resident on the
    # GPU, replacing the former dense W of 4GB per frequency. Built once at startup.
    kernel_cache = WKernelCache(
        Ny=args.phy_Ny, Nx=args.phy_Nx,
        dy=args.phy_dy, dx=args.phy_dx,
        cell_area=args.phy_dx * args.phy_dy,
        device=device, dtype=torch.complex64,
    ).prebuild([f * 1e6 for f in args.f_mhz_lst])
    print(f'W FFT kernels cached: {len(kernel_cache)} freqs')

    isGraph = True
    graph_name='disdepthAdj_32'
    rss_name='splitRSS_32'

    scen = 'area '
    print(scen)

    if isGraph:
        stage = 'train'
        Radio_train_spec = SpectrumDatasetField("./dataset/SpectrumNet/area _train.txt", div_block=True, load_W=False, few_shot_ratio=1,device=device)
        Radio_val_spec = SpectrumDatasetField("./dataset/SpectrumNet/area _valid.txt", div_block=True, load_W=False,few_shot_ratio=1,device=device)


    else:
        stage = 'pretrain'
        # Only the pretraining stage needs W to be loaded.
        Radio_train_spec = SpectrumDatasetField("./dataset/SpectrumNet/area _train.txt", div_block=False, load_W=False, few_shot_ratio=1,device=device)
        Radio_val_spec = SpectrumDatasetField("./dataset/SpectrumNet/area _valid.txt", div_block=False, load_W=False, few_shot_ratio=1,device=device)



    print('Radio_train_spec len:', len(Radio_train_spec))
    print('Radio_val_spec len:', len(Radio_val_spec))


    data_name = 'SpectrumNet'


    if data_name == 'SpectrumNet':
        dataloaders = {
            'train': DataLoader(Radio_train_spec, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            'val': DataLoader(Radio_val_spec, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        }

    model_name = 'PIHG' + str(args.lamda)
    folder_path = 'results/' + data_name + '/' + model_name + '/' + scen + '/' + f'{stage}/'
    os.makedirs(folder_path, exist_ok=True)
    if isGraph:
        pt_path = 'results/' + data_name + '/' + model_name + '/'+  scen + '/' + 'pretrain/best_model.pt'
    else:
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
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.1f}M")


    if isGraph:
        encoder_1_params = list(model.encoder_1.parameters())
        encoder_2_params = list(model.encoder_2.parameters())
        graph_params = list(model.graphEn.parameters())

        # ---- freeze the encoders completely ---- #
        for param in model.encoder_1.parameters():
            param.requires_grad = False
        for param in model.encoder_2.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.graphEn.parameters(), lr=1e-3, weight_decay=1e-5)
        # Composite learning-rate schedule
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                # Stage 1: warm-up (5 epochs)
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,  # start from lr*0.01
                    end_factor=1.0,
                    total_iters=5),

                # Stage 2: cosine annealing with warm restarts
                CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=20,          # length of the first cycle
                    T_mult=2,        # cycle-length multiplier
                    eta_min=1e-6,    # minimum learning rate
                )
            ],
            milestones=[5]  # switch to stage 2 after epoch 5
        )

    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    # Fallback scheduler, used when cosine annealing stalls
    backup_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    if isGraph:
        model = train_model(model, dataloaders, optimizer, scheduler, backup_scheduler, folder_path, device, num_epochs=300)
    else:
        model = pretrain_model(model, dataloaders, optimizer, scheduler, backup_scheduler, folder_path, device, num_epochs=300,
                               kernel_cache=kernel_cache, lamda=args.lamda)

    torch.save(model.state_dict(), folder_path + 'model.pt')
