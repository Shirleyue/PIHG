'''
code by jiangxinyue(shirleyuue@foxmail.com)
date:2025-03-21
'''

import os
import random
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from readPng import parse_png_filename
from readNpz import parse_npz_filename
from utils import read_files_in_natural_order

import re

from tqdm import tqdm
import numpy as np
import sys


# List every file path in a folder
def get_file_names(folder_path):
    file_pths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return file_pths



# Build the parameter dictionary of the NPZ files
def build_npz_dict(npz_file_lists):
    npz_dict = defaultdict(list)
    for npz_file_pth in npz_file_lists:
        npz_file_name = os.path.basename(npz_file_pth)
        npz_params = parse_npz_filename(npz_file_name)
        key = (npz_params["terrain_type"], npz_params["climate_type"], npz_params["map_id"], npz_params["sample_id"])
        npz_dict[key].append(npz_file_pth)
    return npz_dict

# Find the NPZ file matching a given PNG file name
def find_npz_from_png(png_file_name, npz_dict):
    png_params = parse_png_filename(png_file_name)
    key = (png_params["terrain_type"], png_params["climate_type"], png_params["map_id"], png_params["sample_id"])
    return npz_dict.get(key, [])

# Build the list of matching PNG / NPZ files
def get_file_to_Lst(png_pth, npz_pth):
    pathLst = []
    npz_file_lists = get_file_names(npz_pth)
    npz_dict = build_npz_dict(npz_file_lists)

    for item in os.listdir(png_pth):  # '01.Grassland'
        item_path = os.path.join(png_pth, item).replace("\\", "/")  # ./data/01.Grassland/
        for file in os.listdir(item_path):
            npz_pth_Lst = find_npz_from_png(file, npz_dict)  # dictionary lookup
            if npz_pth_Lst:
                filePth = os.path.join(item_path, file).replace("\\", "/")
                # Join the PNG and the NPZ; npz_pth_Lst is a list but only one entry can match
                dlst = f"{filePth}\t{npz_pth_Lst[0]}\n"
                pathLst.append(dlst)
    return pathLst



def find_png_from_npz(npz_file_lists, png_file_lists):

    # Pre-build a lookup dictionary of PNG file names for O(1) access
    png_dict = defaultdict(list)
    for png_path in png_file_lists:
        base_name = os.path.basename(png_path)
        # Extract the shared part, assuming names look like T01C0D0000_n00_fX_ss_zY.png
        key = '_'.join(base_name.split('_')[:2])   # the T01C0D0000_n00 part
        png_dict[key].append(png_path)

    match_list = []
    pattern = re.compile(r"(T\d{2}C\dD\d{4}_n\d{2})_bdtr\.npz")
    for npz_pth in tqdm(npz_file_lists, desc="Processing NPZ files", unit="file"):
        npz_name = os.path.basename(npz_pth)
        match = re.match(pattern, npz_name)
        common_part = match.group(1)  # T01C0D0000_n00
        matched_pngs = png_dict.get(common_part, [])
        if len(matched_pngs) == 15:
            # Extend in one go, rather than appending one by one
            match_list.extend(f"{png}\t{npz_pth}\n" for png in matched_pngs)

    return match_list


def file_filter(dat_lst, dic):
    """
    Filter dat_lst with the conditions given in the dictionary dic.
    :param dat_lst: data list; every element is a tuple (png_path, npz_path)
    :param dic: dictionary of filter conditions, e.g. {"height_id": 1, "frequency_id": 2, "terrain_type": 1}
    :return: the filtered data list
    """
    filtered_dat_lst = []
    for lst in dat_lst:
        png_lst = lst.strip().split('\t')
        png_file_name = png_lst[0].split("/")[-1]
        params = parse_png_filename(png_file_name)
        # Every condition has to hold
        match = True
        for key, value in dic.items():
            if params.get(key) != value:
                match = False
                break

        if match:
            filtered_dat_lst.append(lst)

    return filtered_dat_lst


# Write the labeled path list to txt
def write_data_path(dataList, fileName, isShuf=False):
    # dataList: a list of data paths and tags,
    # fileName: the txt file name to write path to
    # isShuf: Indicates whether the data set is shuffled
    if isShuf:
        random.shuffle(dataList)
    with open(fileName, 'w', encoding='UTF-8') as f:
        for dat in dataList:
            f.write(str(dat))

def read_data_path(fileName):
    """
    Read the stored list of file paths back from a txt file.
    :param fileName: path of the txt file
    :return: list of file paths
    """
    dataList = []
    with open(fileName, 'r', encoding='UTF-8') as f:
        for line in f:
            dataList.append(line)
    return dataList



def div_train_test(dataList, train_ratio, valid_ratio, test_ratio, datst_pth='', isShuf=True, group_size=15, random_seed=42):
    """
    Split the dataset into train / valid / test by group ratio, keeping every group intact.

    Args:
        dataList: the original dataset (list or array)
        train_ratio: training-set ratio (0~1)
        valid_ratio: validation-set ratio (0~1)
        test_ratio: test-set ratio (0~1)
        isShuf: whether to shuffle the group order (default True)
        group_size: number of elements per group (default 15)
        random_seed: random seed (default 42)

    Returns:
        train_list, valid_list, test_list
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "the ratios must sum to 1"
    assert len(dataList) % group_size == 0, f"the total data length must be divisible by {group_size}"

    total_size = len(dataList)
    total_groups = len(dataList) // group_size
    group_indices = np.arange(total_groups)  # group indices [0,1,2,...]

    # Shuffle the group order, reproducibly
    if isShuf:
        np.random.seed(random_seed)
        np.random.shuffle(group_indices)

    # Group boundaries of each split
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

    train_list = [dataList[i] for i in train_indices]
    valid_list = [dataList[i] for i in valid_indices]
    test_list = [dataList[i] for i in test_indices]

    print(f"Total samples: {total_size}")
    print(f"Train set: {len(train_list)} ({(len(train_list)/total_size)*100:.1f}%)")
    print(f"Valid set: {len(valid_list)} ({(len(valid_list)/total_size)*100:.1f}%)")
    print(f"Test set: {len(test_list)} ({(len(test_list)/total_size)*100:.1f}%)")

    write_data_path(train_list, datst_pth + 'train.txt', isShuf=False)
    write_data_path(valid_list, datst_pth + 'valid.txt', isShuf=False)
    write_data_path(test_list, datst_pth + 'test.txt', isShuf=False)

    return train_list, valid_list, test_list


def find_tx_num(txt_path, tx_num):
    with open(txt_path, 'r', encoding='utf-8') as f:
        imgs_info = f.readlines()
        imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
    txs_info = []
    for img_info in imgs_info:
        tx_img = np.load(img_info[2])
        nonzero_positions = np.nonzero(tx_img)
        if len(nonzero_positions[0]) == tx_num:
            tx_info = img_info
            tx_info_str = '\t'.join(tx_info) + '\n'
            txs_info.append(tx_info_str)

    txt_path = txt_path.replace('.txt', f'_{tx_num}tx.txt')
    write_data_path(txs_info, txt_path, isShuf=False)


if __name__ == '__main__':
    train_ratio = 0.7
    valid_ratio = 0.15
    test_ratio = 0.15

    new_imags = find_tx_num("./dataset/SpectrumNet/area _test.txt", 2)
    print('Done!')

    png_pth = '/root/autodl-tmp/SpectrumNet/png/'
    npz_pth = '/root/autodl-tmp/SpectrumNet/npz/'


    #   ----- read all the data and write it to a txt file  ----- #
    dat_lst = get_file_to_Lst(png_pth, npz_pth)
    print('dat_lst len:', len(dat_lst))

    # ---- match the data up, pairing each png with its npz ------- #
    npz_file_lists = get_file_names(npz_pth)
    png_file_lists = [item.split('\t')[0] for item in dat_lst]
    match_lst = find_png_from_npz(npz_file_lists, png_file_lists)
    write_data_path(match_lst, './dataset/matched_dat.txt', isShuf=False)  # 94458


    # ---- filter the data down to dense urban at height 0 ------- #
    data_lst = match_lst
    dic_lst = [ {"terrain_type":11}]

    for idx, terrain_list in enumerate(dic_lst, start=1):
        filtered_dat_lst = file_filter(data_lst, terrain_list)
        scen_idx = terrain_list['terrain_type']
        print(f'scenario_{scen_idx} len: {len(filtered_dat_lst)}')

        scen_lst = filtered_dat_lst
        dic_lst = [{"height_id":0}]
        for idx, height_list in enumerate(dic_lst, start=0):
            filtered_dat_lst = file_filter(scen_lst, height_list)
            imgs_info = list(map(lambda x: x.strip().split('\t'), filtered_dat_lst))

            # ----add txt pth ---- #
            pth_lst = []
            for img_info in imgs_info:
                dir_path = os.path.dirname(img_info[1])  # folder pth: '/root/autodl-tmp/SpectrumNet/npz/'
                file_name = os.path.basename(img_info[1]) # 'T06C2D0076_n01_bdtr.npz'
                tx_pth = dir_path.replace('npz', 'tx_dB') + '/' + file_name.replace('_bdtr.npz', '.npy')
                dlst = f"{img_info[0]}\t{img_info[1]}\t{tx_pth}\n"
                pth_lst.append(dlst)

            height_idx = height_list['height_id']
            print(f'scenario_{scen_idx}__height_{height_idx} len: {len(filtered_dat_lst)}')
            scenario_height_file = f'./dataset/SpectrumNet/scenario_{scen_idx}_height_{height_idx}.txt'
            write_data_path(pth_lst, scenario_height_file, isShuf=False)


    print('Done!')
