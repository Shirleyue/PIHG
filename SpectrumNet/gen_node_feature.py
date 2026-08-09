
'''
Split the dataset and generate blocks. At least 40 areas worth of data are needed to
produce 640 blocks.

'''
import numpy as np
from scipy.sparse import lil_matrix
import os
import matplotlib.pyplot as plt
import networkx as nx
from typing import List
from types import SimpleNamespace
import sys
from pathlib import Path
# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from SpectrumNet.readPng import load_png_data
from SpectrumNet.readNpz import load_npz_data
from SpectrumNet.utils import read_files_in_natural_order, _get_height_code


def split_matrix_with_overlap(matrix, grid, overlap=None):
    """
    Split a matrix into grid x grid sub-matrices, allowing overlap, and extract the
    contents of each sub-matrix. Sub-matrices never extend past the original matrix.
    :param matrix: original matrix (nested list or NumPy array)
    :param grid: sub-matrix size (grid x grid)
    :param overlap: optional number of overlapping pixels; computed automatically if None
    :return:
        sub_matrices: list with the contents of every sub-matrix
        positions: position of every sub-matrix (x, y, w, h)
        stride_x, stride_y: the strides that were computed
    """

    if isinstance(matrix, np.ndarray):
        h, w = matrix.shape
    else:
        h = len(matrix)
        w = len(matrix[0]) if h > 0 else 0

    # Number of sub-matrices needed (rounded up)
    n_cols = (w + grid - 1) // grid
    n_rows = (h + grid - 1) // grid
    n_blocks = n_rows * n_cols

    if overlap is None:
        # Total extra space that has to be covered
        total_extra_x = max(0, n_cols * grid - w)
        total_extra_y = max(0, n_rows * grid - h)

        # Strides, accounting for the overlap
        stride_x = grid - (total_extra_x // max(1, n_cols - 1)) if n_cols > 1 else grid
        stride_y = grid - (total_extra_y // max(1, n_rows - 1)) if n_rows > 1 else grid

        stride_x = max(1, stride_x)
        stride_y = max(1, stride_y)
    else:
        stride_x = grid - overlap
        stride_y = grid - overlap

    positions = []
    sub_matrices = []

    block_idx = 0
    # Walk over the rows (y direction)
    for row in range(n_rows):
        y = row * stride_y
        # Stay inside the boundary
        if y + grid > h:
            y = max(0, h - grid)

        # Walk over the columns (x direction)
        for col in range(n_cols):
            x = col * stride_x
            if x + grid > w:
                x = max(0, w - grid)

            # Actual sub-matrix size, clipped to the boundary
            actual_w = min(grid, w - x)
            actual_h = min(grid, h - y)

            positions.append((x, y, actual_w, actual_h))

            # Extract the sub-matrix (only elements inside the original matrix)
            if isinstance(matrix, np.ndarray):
                sub_matrix = matrix[y:y+actual_h, x:x+actual_w]
            else:
                # For Python lists
                sub_matrix = []
                for i in range(y, y + actual_h):
                    if i < h:
                        row_data = matrix[i][x:x+actual_w]
                        sub_matrix.append(row_data)

            sub_matrices.append(sub_matrix)
    # Turn the list into a 3D NumPy array
    sub_matrices = np.stack(sub_matrices, axis=0)

    return sub_matrices, positions, stride_x, stride_y

def print_matrix(matrix, title="Matrix"):
    """Print a matrix."""
    print(f"\n{title}:")
    for row in matrix:
        print(' '.join(str(cell).rjust(3) for cell in row))


def build_adjacency_matrix(depth_map, d_th, delta, building_map, exclude_building=False):
    """
    Build the adjacency matrix following equation (20) of RadioGAT.
    :param depth_map: radio depth map (H, W)
    :param building_map: building matrix (H, W), 1 marks a building
    :param d_th: distance threshold (in grid cells)
    :param delta: depth-difference threshold
    :return: sparse adjacency matrix (scipy.sparse.lil_matrix)
    """
    H, W = depth_map.shape
    adj = lil_matrix((H * W, H * W))  # sparse matrix format


    for i in range(H):
        for j in range(W):
            if exclude_building and building_map[i, j] == 1:
                continue  # skip building nodes

            # Convert to a 1D index
            u = i * W + j

            # Search the neighbourhood only, instead of the whole map
            i_min = max(0, i - d_th)
            i_max = min(H, i + d_th + 1)
            j_min = max(0, j - d_th)
            j_max = min(W, j + d_th + 1)

            for x in range(i_min, i_max):
                for y in range(j_min, j_max):
                    # Skip self-connections
                    if x == i and y == j:
                        continue
                    if exclude_building and building_map[x, y] == 1:
                        continue  # skip self-connections and buildings

                    # Distance and depth difference
                    dist = np.sqrt((i - x)**2 + (j - y)**2)
                    depth_diff = np.abs(depth_map[i, j] - depth_map[x, y])

                    # Connection criterion (equation 20)
                    if dist <= d_th and depth_diff <= delta:
                        v = x * W + y
                        adj[u, v] = 1  # undirected graph, symmetric link
                        adj[v, u] = 1
    return adj.toarray()


def matrix_sampling(matrix, num_samples):
    """Efficient sparse sampling: generate unique coordinates directly."""
    h, w = matrix.shape
    # Generate unique linear indices and convert them to 2D coordinates
    indices = np.random.choice(h * w, num_samples, replace=False)
    y = indices // w  # row coordinates
    x = indices % w   # column coordinates
    sparse_matrix = np.zeros_like(matrix)
    sparse_matrix[y, x] = matrix[y, x]
    return sparse_matrix, (y, x)


def get_block_local_coords(y_global, x_global, positions):
    """
    Convert global coordinates into coordinates local to each block and keep the sampled
    values.
    :param y_global: array of row coordinates of the global sample points
    :param x_global: array of column coordinates of the global sample points
    :param positions: list with the global position of every block, each entry (x, y, w, h)
                      note: x is the top-left column, y the top-left row
    :param sparse_matrix: original matrix the values are sampled from
    :return: List[Dict], each entry {block_idx, y_local, x_local, values}
    """
    num_blocks = len(positions)
    block_coords = [[] for _ in range(num_blocks)]
    for block_idx, (x_block, y_block, w_block, h_block) in enumerate(positions):
        # Which global sample points fall inside the current block
        mask = (x_global >= x_block) & (x_global < x_block + w_block) & \
               (y_global >= y_block) & (y_global < y_block + h_block)
        y_local = y_global[mask] - y_block
        x_local = x_global[mask] - x_block
        # Convert to linear indices (row-major order)
        indices = y_local * w_block + x_local

        if len(y_local) > 0:
            block_coords[block_idx]= ({
                'block_idx': block_idx,
                'y_local': y_local,
                'x_local': x_local,
                'node_indices': indices.astype(np.int32),  # node index (0 to h*w-1)
            })
    return block_coords


def create_and_save_node_feat (txt_pth, args):
    """
    Read the txt file, process the data and save the graph structure to .npz files.

    Args:
        txt_pth: path of the text file listing the data paths
        grid: block size
        args: namespace with the parameters (must contain d_th and delta)
    """
    grid = args.grid_size
    with open(txt_pth, 'r', encoding='utf-8') as f:
        imgs_info = f.readlines()
        imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))

    for img_info in imgs_info:  # 0 is the png and 1 is the building
        png_data = load_png_data(img_info[0])
        heigtht_code = _get_height_code(png_data['height'])
        rss_gray = png_data['data'] / 255.0  # convert the image from 0-255 to a 0-1 float
        sub_pngs, _, _, _ = split_matrix_with_overlap(rss_gray, grid, overlap=None)


        building_map = load_npz_data(img_info[1])["arrays"]["inBldg_zyx"][heigtht_code]
        sub_buildings, _, _, _ = split_matrix_with_overlap(building_map, grid, overlap=None)

        tx_img = np.load(img_info[2])
        tx_img[tx_img != 0] = 1
        sub_tximg, _, _, _ = split_matrix_with_overlap(tx_img, grid, overlap=None)

        depth_data = np.load(img_info[3])
        sub_depths, _, _, _ = split_matrix_with_overlap(depth_data, grid, overlap=None)

        node_types = np.ones_like(building_map)  # free space defaults to 1
        rows, cols = np.where(tx_img != 0)
        node_types[rows, cols] = 2  # transmitters are class 2
        rows, cols = np.where(building_map != 0)
        node_types[rows, cols] = 0    # buildings are class 0

        sub_node_types, _, _, _ = split_matrix_with_overlap(node_types, grid, overlap=None)

        for i in range(sub_buildings.shape[0]):
            original_path = img_info[0]

            all_mask = np.ones_like(sub_buildings[i])
            coords = np.where(all_mask)
            rss_values = sub_pngs[i][coords]
            node_types_ids = sub_node_types[i][coords]

            split_path =  original_path[:-4].replace('png', f'splitRSS_{grid}') + f'/{i}.npz'

            os.makedirs(os.path.dirname(split_path), exist_ok=True)

            # Save as a compressed npz file, which suits multiple arrays
            np.savez_compressed(
                split_path,
                axis = coords,
                rss = rss_values,
                rss_map = sub_pngs[i],
                building = sub_buildings[i].astype(np.uint8),
                depth = sub_depths[i],
                tx_img = sub_tximg[i].astype(np.uint8),
                node_types_ids = node_types_ids.astype(np.uint8),
            )


def visualize_graph_from_adj(adj, threshold=0.1, block_idx=0):
    """
    Draw the graph structure described by an adjacency matrix.
    :param adj: adjacency matrix (NxN numpy array)
    :param threshold: edges below this threshold are ignored (useful for sparsification)
    """
    adj_filtered = np.where(adj >= threshold, adj, 0)

    G = nx.from_numpy_array(adj_filtered)

    # Layout options: spring_layout, circular_layout, kamada_kawai_layout
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, node_size=20, with_labels=False, edge_color='gray', node_color='blue', alpha=0.7)
    plt.title(f'Block_{block_idx}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_graph_grid(adj_blocks, num_blocks=64, block_size=16):
    """
    Visualise the graph structure of several blocks in a compact grid, drawn with NetworkX.

    Args:
        adj_blocks: list of adjacency blocks (num_blocks, block_size, block_size)
        num_blocks: total number of blocks (default 64)
        block_size: number of nodes per block (default 16)
    """
    rows = cols = int(np.sqrt(num_blocks))  # 16x16 for 64 blocks

    # Create the canvas with a compact layout
    fig = plt.figure(figsize=(20, 20))
    grid = plt.GridSpec(rows, cols, wspace=0.05, hspace=0.05)

    # Shared drawing parameters
    node_size = max(1, 50 // np.sqrt(block_size))  # scale the node size with the block
    edge_width = 0.5

    for i in range(num_blocks):
        ax = fig.add_subplot(grid[i // cols, i % cols])

        G = nx.from_numpy_array(adj_blocks[i])

        pos = nx.spring_layout(G, seed=42, k=0.3/np.sqrt(block_size))

        nx.draw_networkx_nodes(G, pos, ax=ax,
                             node_size=node_size,
                             node_color='skyblue',
                             alpha=0.7)

        # Only draw links with a positive weight
        edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0]
        nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax,
                              width=edge_width,
                              edge_color='gray',
                              alpha=0.5)

        # Block index
        ax.text(0.05, 0.05, str(i),
               transform=ax.transAxes,
               fontsize=6,
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'Network Structure of All {num_blocks} Blocks', y=0.92, fontsize=16)
    plt.tight_layout()
    plt.show()


def get_all_files(root_dir: str, ext: str = None) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if ext and not filename.endswith(ext):
                continue
            files.append(os.path.join(dirpath, filename).replace("\\", "/")+'\n')
    return files


if __name__ == "__main__":

    args = {
        # dataset settings
        'depth_delta': 4.5,
        'd_th': 1,
        'grid_size': 32,  # block size
        'num_blocks': 16,

    }
    args = SimpleNamespace(**args)

    scen = 'scenario_1'
    txt_pth = './dataset/SpectrumNet/' + scen + ' _data.txt'


    create_and_save_node_feat(txt_pth, args)


    print('Done!')
