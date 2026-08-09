import numpy as np
import os
from readPng import load_png_data, parse_png_filename
from readNpz import load_npz_data
from tx_process import load_tx_positions
from utils import _get_height_code, read_data_path

import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib.pyplot as plt
from math import log10
from scipy.sparse import lil_matrix
from numba import jit, prange
from skimage.draw import line
import time
import random
from natsort import natsorted

def bresenham_line(start, end):
    """Compute the straight-line path between two points (Bresenham's algorithm)."""
    x0, y0 = start
    x1, y1 = end
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def calculate_single_depth_map(building_map, tx_pos, f_k, beta):
    """Depth map of a single transmitter."""
    depth_map = np.zeros_like(building_map, dtype=float)
    rows, cols = building_map.shape
    log_fk = beta * log10(f_k)  # constant part
    tx_col, tx_row = tx_pos


    for i in range(rows):
        for j in range(cols):
            if (i, j) == tx_pos:
                depth_map[i, j] = log_fk if building_map[i, j] == 0 else 0
                continue

            path = bresenham_line((tx_row, tx_col), (i, j))
            total_pts = len(path)
            free_pts = sum(1 for (x, y) in path
                        if 0 <= x < rows and 0 <= y < cols
                        and building_map[x, y] == 0)

            T = free_pts / total_pts if total_pts > 0 else 0

            depth_map[i, j] = log_fk * T

    return depth_map

@jit(nopython=True, parallel=True)
def calculate_single_depth_map_numba(building_map, tx_pos, f_k, beta):
    """Accelerated single-transmitter depth map."""
    rows, cols = building_map.shape
    depth_map = np.zeros((rows, cols), dtype=np.float32)
    log_fk = beta * np.log10(f_k)  # np.log10 for numba compatibility

    # Note: tx_pos is in (row, col) order
    tx_col, tx_row = tx_pos


    for i in prange(rows):  # rows
        for j in range(cols):  # columns
            # Buildings and the transmitter position are handled as in the original version
            if (i, j) == (tx_row, tx_col):
                depth_map[i, j] = log_fk if building_map[i, j] == 0 else 0
                continue

            # Bresenham's line algorithm, with the coordinate system fixed up
            # start: (tx_row, tx_col) (row, col)
            # end:   (i, j) (row, col)
            x0, y0 = (tx_row, tx_col)
            x1, y1 = (i,j)

            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            # Walk the path points exactly
            x, y = x0, y0
            total_pts = max(dx, dy) + 1
            free_pts = 0

            for _ in range(total_pts):
                if 0 <= x < rows and 0 <= y < cols and building_map[x, y] == 0:
                    free_pts += 1
                if x == x1 and y == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy


            T = free_pts / total_pts if total_pts > 0 else 0
            depth_map[i, j] = log_fk * T

    return depth_map


@jit(nopython=True, parallel=True)
def calculate_single_depth_dis_map_numba(building_map, tx_pos, f_k, beta, alpha):
    """
    Depth map including the distance term:
    - signal strength = T * (beta * log10(f_k) + alpha * log10(d))
    - T: path transparency (fraction not blocked by buildings)
    - d: Euclidean distance from the node to the transmitter
    """
    rows, cols = building_map.shape
    depth_map = np.zeros((rows, cols), dtype=np.float32)
    log_fk = beta * np.log10(f_k)  # base attenuation term: beta * log10(f_k)
    tx_col, tx_row = tx_pos        # transmitter coordinates (col, row)
    C = 100 + 32.4
    for i in prange(rows):         # rows, in parallel
        for j in range(cols):      # columns
            # The transmitter's own position
            if (i, j) == (tx_row, tx_col):
                depth_map[i, j] = log_fk if building_map[i, j] == 0 else 0
                continue

            # Distance d and the distance-attenuation term log_d
            d = np.sqrt((i - tx_row)**2 + (j - tx_col)**2)
            d = d * 10 / 1000    # distance in km
            log_d = alpha * np.log10(d) if d > 0 else 0  # alpha * log10(d)

            # Path transparency T from Bresenham's algorithm
            x0, y0 = tx_row, tx_col
            x1, y1 = i, j
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            total_pts = max(dx, dy) + 1
            free_pts = 0

            x, y = x0, y0
            for _ in range(total_pts):
                if 0 <= x < rows and 0 <= y < cols and building_map[x, y] == 0:
                    free_pts += 1
                if x == x1 and y == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

            T = free_pts / total_pts if total_pts > 0 else 0
            # Signal-strength formula: T * (log_fk + log_d)
            depth_map[i, j] = T * (C - log_fk - log_d)

    return depth_map


def calculate_multi_depth_map(building_map, tx_positions, f_k=5000, beta=10.0):
    """Combined depth map of several transmitters (sum of all contributions)."""
    combined_depth = np.zeros_like(building_map, dtype=float)
    for tx_pos in tx_positions:
        single_depth = calculate_single_depth_map(building_map, tx_pos, f_k, beta)
        combined_depth += single_depth  # linear superposition
    return combined_depth


def calculate_multi_depth_map_optimized(building_map, tx_positions, f_k=5000, beta=10.0):
    """Combined depth map of several transmitters (sum of all contributions)."""
    combined_depth = np.zeros_like(building_map, dtype=float)
    for tx_pos in tx_positions:
        single_depth = calculate_single_depth_map_numba(building_map, tx_pos, f_k, beta)
        combined_depth += single_depth  # linear superposition
    return combined_depth


def calculate_multi_depth_dis_map_optimized(building_map, tx_positions, f_k, beta, alpha):
    """Combined depth map of several transmitters (sum of all contributions)."""
    combined_depth = np.zeros_like(building_map, dtype=float)
    for tx_pos in tx_positions:
        single_depth = calculate_single_depth_dis_map_numba(building_map, tx_pos, f_k, beta, alpha)
        combined_depth += single_depth  # linear superposition
    return combined_depth


# Read the png and npz, combine them with the frequency information and generate the depth
def gen_radio_depth (txt_pth, depth_txt, beta):
    """xxxx
    """
    with open(txt_pth, 'r', encoding='utf-8') as f:
        imgs_info = f.readlines()
        imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
    depths_info = []
    for img_info in imgs_info:
        png_data = load_png_data(img_info[0])
        f_MHz = png_data['frequency']


        npz_data = load_npz_data(img_info[1])
        heigtht_code = _get_height_code(png_data['height'])
        # Building data of the matching height code
        building_yx = npz_data["arrays"]["inBldg_zyx"][heigtht_code]

        tx_pth = img_info[1].replace('/npz/', '/tx/').replace('bdtr.npz', 'coords.txt')
        tx_pos = load_tx_positions(tx_pth)

        radio_depth = calculate_multi_depth_map_optimized(building_yx, tx_pos, f_MHz, beta)

        save_path = img_info[0].replace('/png/', '/depth/').replace('.png', '.npy')

        folder_path = os.path.dirname(save_path)
        os.makedirs(folder_path, exist_ok=True)
        np.save(save_path, radio_depth)
        # Append the depth-map path to the txt
        depth_info = img_info
        depth_info.append(save_path)
        dpt_info_str = '\t'.join(depth_info) + '\n'
        depths_info.append(dpt_info_str)

    with open(depth_txt, 'w', encoding='UTF-8') as f:
        for dat in depths_info:
            f.write(str(dat))


# Read the png and npz, combine them with the frequency information and generate the depth
def gen_radio_depth_dis (txt_pth, depth_txt, beta, alpha):
    """xxxx
    """
    with open(txt_pth, 'r', encoding='utf-8') as f:
        imgs_info = f.readlines()
        imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
    depths_info = []
    for img_info in imgs_info:
        png_data = load_png_data(img_info[0])
        f_MHz = png_data['frequency']


        npz_data = load_npz_data(img_info[1])
        heigtht_code = _get_height_code(png_data['height'])
        # Building data of the matching height code
        building_yx = npz_data["arrays"]["inBldg_zyx"][heigtht_code]

        tx_pth = img_info[1].replace('/npz/', '/tx/').replace('bdtr.npz', 'coords.txt')
        tx_pos = load_tx_positions(tx_pth)

        radio_depth = calculate_multi_depth_dis_map_optimized(building_yx, tx_pos, f_MHz, beta, alpha)

        save_path = img_info[0].replace('/png/', '/depthdis20/').replace('.png', '.npy')

        folder_path = os.path.dirname(save_path)
        os.makedirs(folder_path, exist_ok=True)
        np.save(save_path, radio_depth)
        # Append the depth-map path to the txt
        depth_info = img_info
        depth_info.append(save_path)
        dpt_info_str = '\t'.join(depth_info) + '\n'
        depths_info.append(dpt_info_str)

    with open(depth_txt, 'w', encoding='UTF-8') as f:
        for dat in depths_info:
            f.write(str(dat))

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
    scen = 'scenario_3'

    # #####    ---------gen radio depth ------------------------  ####
    txt = './dataset/SpectrumNet/' + scen + '_height_0.txt'
    depth_txt = './dataset/SpectrumNet/depth_' + scen + '_height_0.txt'  # where the depth paths go
    gen_radio_depth(txt, depth_txt, beta=10.0)

    # ------------------ split the area data into train / valid / test ---- #
    all_lst = read_data_path(depth_txt)
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
