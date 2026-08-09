import numpy as np
from utils import read_files_in_natural_order, _get_height_code
import numpy as np
from scipy.constants import epsilon_0 as EPS0, mu_0 as MU0, pi
from scipy.special import hankel2, j1  # H^{(2)}_n,  J_1
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
from types import SimpleNamespace
from utils import _get_height_code, read_data_path
import random
import torch
from readPng import load_png_data
from readNpz import load_npz_data

# ---------- basic helpers ---------

def wavenumber(f_hz, eps0=EPS0, mu0=MU0):
    omega = 2 * pi * f_hz
    return omega * np.sqrt(eps0 * mu0), omega

def build_grid(Nx, Ny, dx, dy, origin=(0.0, 0.0)):
    """
    Build the 2D coordinates of the pixel centres of a regular grid.
    Returns:
      pos: (N,2) -> (x,y) physical coordinates
      X,Y: (Ny,Nx) coordinate fields
    """
    x0, y0 = origin
    xs = x0 + (np.arange(Nx) + 0.5) * dx
    ys = y0 + (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys, indexing='xy')  # Y: row direction, X: column direction
    pos = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N,2) with (x,y)
    return pos, X, Y

def green_2d_free_space(r, k):
    return -1j / 4.0 * hankel2(0, k * r)

# ---------- (5)(7): traditional incident field E_inc ----------
def incident_field_from_tx_multi(positions, tx_pos, f_hz, clip_radius=None):
    """
    Multi-transmitter version:
      positions: (N,2) physical coordinates (x,y)
      tx_pos: (M,2) or (2,) transmitter physical coordinates (x,y)
    Returns:
      (N,) complex, the superposition of all transmitter contributions
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    tx  = np.asarray(tx_pos, dtype=float)
    if tx.ndim == 1:
        tx = tx[None, :]  # (1,2)

    k, _ = wavenumber(f_hz)

    # r: (N,M)
    diff = pos[:, None, :] - tx[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    if clip_radius is not None:
        r = np.maximum(r, clip_radius)

    G_nm = green_2d_free_space(r, k)  # (N,M)
    return G_nm.sum(axis=1)  # (N,)

# ---------- (3)(8): contrast chi ----------
def contrast_chi_from_er_sigma(er_map, sigma_map, f_hz, eps0=EPS0):
    er = np.asarray(er_map).astype(float).ravel()
    sigma = np.asarray(sigma_map).astype(float).ravel()
    _, omega = wavenumber(f_hz, eps0)
    chi = (er - 1.0) - 1j * sigma / (omega * eps0)
    return chi.astype(np.complex64)

# ---------- (11)(12): coefficient matrix W ----------
def build_W_matrix(positions, f_hz, cell_area):
    k, _ = wavenumber(f_hz)
    a = np.sqrt(cell_area / pi)
    ka = k * a

    # Pairwise distance matrix (N x N); positions must be (N,2)
    R = cdist(positions, positions, metric='euclidean')

    # Off-diagonal terms
    W = 1j * pi * ka / 2.0 * j1(ka) * hankel2(0, k * np.where(R == 0.0, 1.0, R))
    np.fill_diagonal(W, 0.0)

    # Diagonal term
    diag_val = 1j / 2.0 * (pi * ka * hankel2(1, ka) - 2j)
    np.fill_diagonal(W, diag_val)

    return W.astype(np.complex128)


# ---------- (10): recover the incident field from the total field ----------
def incident_from_total(E_tot, W, chi):
    E_tot = np.asarray(E_tot).ravel().astype(np.complex128)
    chi = np.asarray(chi).ravel().astype(np.complex128)
    return E_tot + W.dot(chi * E_tot)


def build_grid_no_center(Nx, Ny, dx=1.0, dy=1.0, origin=(0.0, 0.0)):
    """
    Without the 0.5 centre offset: use the grid points (col*dx, row*dy) directly to
    build the (N,2) physical coordinates (x,y).
    """
    x0, y0 = origin
    cols, rows = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='xy')
    X = x0 + cols * dx
    Y = y0 + rows * dy
    pos = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N,2), (x,y)
    return pos, X, Y

def tx_positions_from_map_no_center(tx_map, dx=1.0, dy=1.0, origin=(0.0, 0.0)):
    """
    Extract the transmitter coordinates from the non-zero pixels of tx_map (without the
    0.5 offset), consistent with build_grid_no_center.
    """
    rows, cols = np.where(tx_map != 0)
    if rows.size == 0:
        raise ValueError("tx_map contains no transmitter positions (no non-zero pixels).")
    x = origin[0] + cols * dx
    y = origin[1] + rows * dy
    return np.stack([x, y], axis=1)  # (M,2)

def compute_incident_field_from_tx(building_map, tx_map, f_hz, E_tot_pred=None,
                                   dx=10.0, dy=10.0, origin=(0.0, 0.0)):
    Ny, Nx = building_map.shape
    cell_area = dx * dy
    a = np.sqrt(cell_area / np.pi)

    # Coordinates without the centre offset
    positions, X, Y = build_grid_no_center(Nx, Ny, dx, dy, origin)
    tx_xy = tx_positions_from_map_no_center(tx_map, dx, dy, origin)

    # Medium parameter maps (example values)
    er_map = np.ones_like(building_map, dtype=float)
    er_map[building_map != 0] = 4.0
    sigma_map = np.zeros_like(er_map, dtype=float)

    # Traditional incident field (multiple transmitters)
    E_inc_trad = incident_field_from_tx_multi(positions, tx_xy, f_hz, clip_radius=a)

    E_inc_trad_2channel = np.stack([
        np.real(E_inc_trad.reshape(Ny, Nx)),
        np.imag(E_inc_trad.reshape(Ny, Nx))
    ], axis=0)

    # chi
    chi = contrast_chi_from_er_sigma(er_map, sigma_map, f_hz)

    return dict(positions=positions, chi=chi,
                E_inc_trad_2channel=E_inc_trad_2channel)
def create_and_save_incident (txt_pth, inc_txt, args):
    """
    Read the txt file, process the data and save the graph structure to .npz files.

    Args:
        txt_pth: path of the text file listing the data paths
        grid: block size
        args: namespace with the parameters (must contain d_th and delta)
    """
    dx = args.dx
    dy = args.dy
    incs_info = []
    with open(txt_pth, 'r', encoding='utf-8') as f:
        imgs_info = f.readlines()
        imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))

    for img_info in imgs_info:  # 0 is the png and 1 is the building
        png_data = load_png_data(img_info[0])
        f_hz = png_data['frequency'] * 1e6
        heigtht_code = _get_height_code(png_data['height'])
        rss_gray = png_data['data'] / 255.0  # convert the image from 0-255 to a 0-1 float


        building_map = load_npz_data(img_info[1])["arrays"]["inBldg_zyx"][heigtht_code]

        tx_img = np.load(img_info[2])
        tx_img[tx_img != 0] = 1

        result = compute_incident_field_from_tx(building_map, tx_img, f_hz,
                                        E_tot_pred=None, dx=dx, dy=dy)

        incident_path = img_info[0].replace('/png/', f'/incField_{dx}/').replace('.png', '.npz')
        os.makedirs(os.path.dirname(incident_path), exist_ok=True)
        np.savez_compressed(
            incident_path,
            axis = result['positions'],
            chi = result['chi'],
            E_inc_trad_2channel = result['E_inc_trad_2channel'],

        )

        inc_field_info = img_info
        inc_field_info.append(incident_path)
        inc_info_str = '\t'.join(inc_field_info) + '\n'
        incs_info.append(inc_info_str)

    with open(inc_txt, 'w', encoding='UTF-8') as f:
        for dat in incs_info:
            f.write(str(dat))


# ---------- (11)(12): coefficient matrix W ----------
def gen_W_matrix_and_Save(folder, positions, f_mhz_lst, cell_area):

    for f_mhz in f_mhz_lst:
        f_hz = f_mhz * 1e6
        W = build_W_matrix(positions, f_hz, cell_area)

        W_path = folder+ f'{f_mhz}.npz'
        os.makedirs(os.path.dirname(W_path), exist_ok=True)
        np.savez_compressed(
                W_path,
                W = W,
            )
        print(f"W saved to {W_path}")

def write_data_path(dataList, fileName, isShuf=False):
    # dataList: a list of data paths and tags,
    # fileName: the txt file name to write path to
    # isShuf: Indicates whether the data set is shuffled
    if isShuf:
        random.shuffle(dataList)
    with open(fileName, 'w', encoding='UTF-8') as f:
        for dat in dataList:
            f.write(str(dat))

if __name__ == "__main__":

    args = {
        # dataset settings
        'dx': 10.0,
        'dy': 10.0,
        'Nx': 128,  # block size
        'Ny': 128,
        'origin':(0.0, 0.0),

    }
    args = SimpleNamespace(**args)

    scen = '/scenario_9'
    txt_pth = './dataset/SpectrumNet/' + scen + ' _data.txt'
    inc_txt = './dataset/SpectrumNet/' + scen + ' _inc_data.txt'  # where the depth paths go

    create_and_save_incident(txt_pth, inc_txt, args)

    # ------------------ split the area data into train / valid / test ---- #
    all_lst = read_data_path(inc_txt)
    group_size = 5   # one group per 5 frequencies
    train_ratio = 0.8
    valid_ratio = 0.1

    total_size = len(all_lst)
    total_groups = len(all_lst) // group_size
    group_indices = np.arange(total_groups)  # group indices [0,1,2,...]
    train_end = int(total_groups * train_ratio)
    valid_end = train_end + int(total_groups * valid_ratio)
    # Original indices of each group
    train_groups = group_indices[:train_end]
    valid_groups = group_indices[train_end:valid_end]
    test_groups = group_indices[valid_end:]

    # Expand the group indices down to element level
    def expand_group_indices(groups):
        indices = []
        for group_id in groups:
            start = group_id * group_size
            indices.extend(range(start, start + group_size))
        return indices


    train_indices = expand_group_indices(train_groups)
    valid_indices = expand_group_indices(valid_groups)
    test_indices = expand_group_indices(test_groups)

    train_list = [all_lst[i] for i in train_indices]
    valid_list = [all_lst[i] for i in valid_indices]
    test_list = [all_lst[i] for i in test_indices]


    print('len(area_dat_lst):', len(all_lst))
    print('len(area_train_lst):', len(train_list))
    print('len(area_valid_lst):', len(valid_list))
    print('len(area_test_lst):', len(test_list))

    write_data_path(all_lst, './dataset/SpectrumNet/' + scen + ' _data.txt', isShuf=False)
    write_data_path(train_list, './dataset/SpectrumNet/' + scen + ' _train.txt', isShuf=False)
    write_data_path(valid_list, './dataset/SpectrumNet/' + scen + ' _valid.txt', isShuf=False)
    write_data_path(test_list, './dataset/SpectrumNet/' + scen + ' _test.txt', isShuf=False)

    print('Done!')
