
import numpy as np
import os
from scipy.sparse import lil_matrix
from scipy.spatial.distance import cdist
from bresenham import bresenham  # requires: pip install bresenham
from multiprocessing import Pool, cpu_count

from SpectrumNet.utils import read_files_in_natural_order
from SpectrumNet.tx_process import load_tx_positions

# Method 1: build the adjacency matrix from a node-distance threshold.
# Input: the coordinates stored under NodeFeatureAxis40/1/1750/1_1.txt,
# start from an all-zero adj, compute the distances, test them, add the edges,
# then save the adj of every block to a file.


def gen_adj_by_distance_Spectrum(block_axis, dth=4.5):
    """
    Build the adjacency matrix from a node-distance threshold (vectorised version).
    Args:
        block_axis: [2, N_Node] array; row 0 holds the y coordinates, row 1 the x ones
        dth: distance threshold (default 4.5)
    Returns:
        adj: adjacency matrix (undirected, with self-loops), returned as a dense array
    """
    # Convert to an (N, 2) array of (x,y) coordinates
    coords = np.column_stack((block_axis[1], block_axis[0]))

    # Euclidean distance matrix between all nodes
    dist_matrix = cdist(coords, coords, 'euclidean')

    # Adjacency matrix, self-loops included
    adj = (dist_matrix < dth).astype(int)

    return adj


def gen_adj_by_dis_depth_Spectrum(block_axis, dth, block_depth, sigma):
    """Optimised adjacency-matrix generation."""
    num_nodes = block_axis.shape[1]
    adj = lil_matrix((num_nodes, num_nodes), dtype=np.uint8)

    # 1. Euclidean distance between every pair of nodes
    coords = block_axis.T  # reshaped to (N, 2)
    dist_matrix = cdist(coords, coords, 'euclidean')

    # 2. Depth-difference matrix
    y_coords = coords[:, 0].astype(int)
    x_coords = coords[:, 1].astype(int)
    depths = block_depth[y_coords, x_coords]
    depth_diff_matrix = np.abs(depths[:, None] - depths[None, :])

    # 3. Apply both conditions
    mask = (dist_matrix < dth) & (depth_diff_matrix < sigma)
    np.fill_diagonal(mask, 0)

    return mask.astype(np.uint8)



def process_single_item(args):
    """Process a single sample; shaped for multiprocessing."""
    dat_lst, dth, sigma = args
    data = np.load(dat_lst)
    block_axis = data['axis']
    block_depth = data['depth']

    adj = gen_adj_by_dis_depth_Spectrum(block_axis, dth, block_depth, sigma)

    adj_pth = dat_lst.replace('splitRSS', f'disAdj{dth:.1f}')


    os.makedirs(os.path.dirname(adj_pth), exist_ok=True)
    np.savez_compressed(adj_pth, adj=adj.astype(np.uint8))
    return adj_pth


def main():
    dth = 4.5  # distance threshold
    stand_freq = 5750
    stand_cof = 10 * np.log10(stand_freq)  # 37.6
    obs_cof = 0.5  # obstruction coefficient, default 0.1
    sigma = obs_cof*stand_cof  # depth threshold

    folder_path = "/root/autodl-tmp/SpectrumNet/splitRSS_32/06.DenseUrban"

    data_lst = read_files_in_natural_order(folder_path)

    # Process pool over all available cores
    num_workers = cpu_count()
    with Pool(num_workers) as pool:
        args_list = [(dat_lst, dth, sigma) for dat_lst in data_lst]

        # imap returns the results in order; map would block on memory
        results = list(pool.imap(process_single_item, args_list, chunksize=10))

    print(f"Finished: {len(results)} adjacency matrices generated")



if __name__ == "__main__":
    dth = 4.5  # distance threshold
    stand_freq = 5750
    stand_cof = 10 * np.log10(stand_freq)  # 37.6
    obs_cof = 0.5  # obstruction coefficient, default 0.1
    sigma = obs_cof*stand_cof  # depth threshold
    print('Done!')
