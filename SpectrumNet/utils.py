import os
import re
import random
import torch
import numpy as np

def get_all_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def extract_numbers_from_path(path):
    """Extract every number in a path and return them as a list."""
    numbers = []
    for part in path.split(os.sep):
        # Numbers of one path component, e.g. "1_1.txt" -> [1, 1]
        nums_in_part = list(map(int, re.findall(r"\d+", part)))
        numbers.extend(nums_in_part)
    return numbers

def get_sorted_files_and_dirs(root_dir):
    """List every file and directory in natural numeric order."""
    all_items = []

    for root, dirs, files in os.walk(root_dir):
        # Sort the sub-directories and files of the current directory numerically
        dirs.sort(key=lambda d: list(map(int, re.findall(r"\d+", d))))
        files.sort(key=lambda f: list(map(int, re.findall(r"\d+", f))))

        # Add the files of the current directory, with their full path
        for file in files:
            full_path = os.path.join(root, file)
            all_items.append((extract_numbers_from_path(full_path), full_path))

    # Sort all files numerically
    all_items.sort(key=lambda x: x[0])
    sorted_files = [item[1] for item in all_items]
    return sorted_files

def read_files_in_natural_order(root_dir):
    """Read the files in natural numeric order."""
    sorted_files = get_sorted_files_and_dirs(root_dir)
    return sorted_files

def write_data_path(dataList, fileName, isShuf=False):
    # dataList: a list of data paths and tags,
    # fileName: the txt file name to write path to
    # isShuf: Indicates whether the data set is shuffled
    if isShuf:
        random.shuffle(dataList)
    with open(fileName, 'w', encoding='UTF-8') as f:
        for dat in dataList:
            f.write(str(dat)+'\n')

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


def _get_height_code (height):
    """Convert an actual height (metres) into a height-layer code."""
    height_map = {
        1.5: 0,
        30.0: 1,
        200.0: 2
    }
    return height_map.get(height, -1.0)


def get_all_files(folder_path):
    """
    List the paths of every file under a folder.
    Args:
        folder_path: the folder path (e.g. './a')
    Returns:
        list: list of file paths (relative or absolute)
    """
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def ensure_edge_index_2xE(ei):
    # ei: numpy array or tensor, shaped either (E,2) or (2,E)
    arr = np.array(ei)
    if arr.shape[0] == 2:
        return torch.LongTensor(arr)           # (2, E)
    elif arr.shape[1] == 2:
        return torch.LongTensor(arr.T)         # transposed to (2, E)
    else:
        raise ValueError("edge_index shape unexpected: " + str(arr.shape))

def adj_to_edge_index(adj):
    """
    Convert an adjacency matrix (torch.Tensor) into the edge_index format PyG expects.

    Args:
        adj: torch.Tensor, shape = [N, N], a 0/1 matrix describing undirected adjacency

    Returns:
        edge_index: torch.LongTensor, shape = [2, num_edges]
    """
    # Indices of every non-zero edge, symmetric pairs included
    row, col = torch.nonzero(adj, as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)  # shape = [2, num_edges]
    return edge_index

def generate_edge_type(edge_index, node_type_ids):
    """
    Derive the edge types edge_type from the node types and the edges (undirected graphs
    are supported).

    Args:
        edge_index: torch.LongTensor [2, num_edges] -- one (source, target) per edge
        node_type_ids: torch.LongTensor [num_nodes] -- type of each node (0,1,2)

    Returns:
        edge_index: [2, valid_edges]
        edge_type:  [valid_edges]
    """
    src_type = node_type_ids[edge_index[0]]
    tgt_type = node_type_ids[edge_index[1]]

    # Keep undirected edges consistent: the smaller type always comes first
    min_type = torch.minimum(src_type, tgt_type)
    max_type = torch.maximum(src_type, tgt_type)

    # -1 marks an undefined type
    edge_type = torch.full_like(min_type, fill_value=-1)

    # Edge-type mapping
    edge_type[(min_type == 0) & (max_type == 0)] = 0   # 0-0, building-building
    edge_type[(min_type == 0) & (max_type == 1)] = 1   # 0-1, Obs-Rx
    edge_type[(min_type == 0) & (max_type == 2)] = 2   # 0-2, Obs-Tx
    edge_type[(min_type == 1) & (max_type == 1)] = 3   # 1-1, Rx-Rx
    edge_type[(min_type == 1) & (max_type == 2)] = 4   # 1-2, Rx-Tx
    edge_type[(min_type == 2) & (max_type == 2)] = 5   # 2-2, transmitter-transmitter


    # Drop the undefined edges
    valid_mask = edge_type >= 0
    edge_index = edge_index[:, valid_mask]
    edge_type = edge_type[valid_mask]

    return edge_index, edge_type


def _get_frequency_code(frequency_mhz):
    """Convert an actual frequency (MHz) into a frequency code.

    Args:
        frequency_mhz (int): the frequency in MHz

    Returns:
        int: the matching frequency code (0~4), or -1 if there is none
    """
    freq_to_code = {
        150: 0,    # 150 MHz -> 0
        1500: 1,   # 1.5 GHz -> 1
        1700: 2,   # 1.7 GHz -> 2
        3500: 3,   # 3.5 GHz -> 3
        22000: 4   # 22 GHz -> 4
    }
    return freq_to_code.get(frequency_mhz, -1)
