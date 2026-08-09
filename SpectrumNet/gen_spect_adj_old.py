
import numpy as np
import os
from utils import read_files_in_natural_order
from scipy.sparse import lil_matrix
from tx_process import load_tx_positions
from scipy.spatial.distance import cdist
from bresenham import bresenham  # requires: pip install bresenham

# Method 1: build the adjacency matrix from a node-distance threshold.
# Input: the coordinates stored under NodeFeatureAxis40/1/1750/1_1.txt,
# start from an all-zero adj, compute the distances, test them, add the edges,
# then save the adj of every block to a file.
def gen_adj_by_distance(block_axis, dth=4.5):

    """
    Build the adjacency matrix from a node-distance threshold.
    Args:
        block_axis: [2, N_Node] array; row 0 holds the y coordinates, row 1 the x ones
        dth: distance threshold (default 4.5)
    Returns:
        adj: adjacency matrix (undirected, with self-loops)
    """
    num_nodes = block_axis.shape[1]
    adj = lil_matrix((num_nodes, num_nodes))  # sparse matrix format

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            # Distance criterion
            if np.sqrt((block_axis[0,i]-block_axis[0,j])**2+(block_axis[1,i]-block_axis[1,j])**2) < dth:
                adj[i, j] = 1
                adj[j, i] = 1  # symmetric, undirected graph
    return adj.toarray().astype(int)


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
    adj[mask] = 1

    return adj.toarray()


def gen_adj_by_dis_depth_building_Spectrum(block_axis, dth, block_depth, sigma, block_building):
    """
    Build the adjacency matrix from distance, depth difference and the absence of a
    building blocking the path.

    Args:
        block_axis: (2, N) array holding the (y, x) coordinates of the nodes
        dth: distance threshold (Euclidean)
        block_depth: depth map of shape (H, W)
        sigma: depth-difference threshold
        block_building: building map of shape (H, W); 1 is a building, 0 is open space

    Returns:
        adjacency matrix (N, N); 1 means connected, 0 means not connected
    """
    num_nodes = block_axis.shape[1]
    adj = lil_matrix((num_nodes, num_nodes))  # sparse matrix format

    for i in range(num_nodes):
        for j in range(i, num_nodes):  # upper triangle, self-loops included
            y_i, x_i = int(block_axis[0,i]), int(block_axis[1,i])
            y_j, x_j = int(block_axis[0,j]), int(block_axis[1,j])

            # Distance and depth difference
            dist = np.sqrt((y_i-y_j)**2 + (x_i-x_j)**2)
            depth_diff = np.abs(block_depth[y_i, x_i] - block_depth[y_j, x_j])

            # Basic criteria
            if dist < dth and depth_diff < sigma:
                # Is a building blocking the line between the two points?
                line_path = list(bresenham(x_i, y_i, x_j, y_j))  # all points on the line
                has_building = any(block_building[y, x] == 1 for x, y in line_path)

                if not has_building:
                    adj[i, j] = 1
                    adj[j, i] = 1  # symmetric, undirected graph

    return adj.toarray().astype(int)


def gen_adj_by_dis_depth_Spectrum_or(block_axis, dth, block_depth, sigma):
    """
    Build the adjacency matrix from distance and depth difference.

    Args:
        block_axis: (2, N) array holding the (y, x) coordinates of the nodes
        dth: distance threshold (Euclidean)
        block_depth: depth map of shape (H, W)
        sigma: depth-difference threshold

    Returns:
        adjacency matrix (N, N); 1 means connected, 0 means not connected
    """
    num_nodes = block_axis.shape[1]
    adj = lil_matrix((num_nodes, num_nodes))  # sparse matrix format

    for i in range(num_nodes):
        for j in range(i, num_nodes):  # upper triangle, self-loops included
            x_i, y_i = block_axis[1,i], block_axis[0,i]
            x_j, y_j = block_axis[1,j], block_axis[0,j]

            dist = np.sqrt((y_i-y_j)**2+(x_i-x_j)**2)
            depth_diff = np.abs(block_depth[y_i, x_i] - block_depth[y_j, x_j])

            if dist < dth or depth_diff<sigma:
                adj[i, j] = 1
                adj[j, i] = 1  # symmetric, undirected graph
    return adj.toarray().astype(int)


def gen_adj_by_building(block_axis, block_building):
    """
    Build the adjacency matrix from building occlusion: an edge is added when no building
    blocks the path.
    Args:
        block_axis: [2, N_Node] array; row 0 holds the y coordinates, row 1 the x ones
        block_building: building map; 1 is a building, 0 is open space
    Returns:
        adj: adjacency matrix (undirected, with self-loops)
    """
    num_nodes = block_axis.shape[1]
    adj = lil_matrix((num_nodes, num_nodes))  # sparse matrix format

    for i in range(num_nodes):
        y_i, x_i = block_axis[0, i], block_axis[1, i]
        for j in range(i, num_nodes):  # upper triangle, self-loops included
            y_j, x_j = block_axis[0, j], block_axis[1, j]
            dx = abs(x_j - x_i)
            dy = abs(y_j - y_i)
            sx = 1 if x_i < x_j else -1
            sy = 1 if y_i < y_j else -1
            err = dx - dy
            # Walk the path points exactly
            x, y = x_i, y_i
            total_pts = max(dx, dy) + 1
            building_cnt = 0
            while True:
                if block_building[y, x] == 1:  # a building occupies this cell
                    building_cnt += 1

                if x == x_j and y == y_j:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

            if building_cnt == 0:
                adj[i, j] = 1  # only add the edge when the path is clear
                adj[j, i] = 1  # symmetric, undirected graph

    return adj.toarray().astype(int)


def are_collinear(xi, yi, xj, yj, tx_x, tx_y, tol=1e-6):
    """
    Test whether the three points (xi,yi), (xj,yj), (tx_x,tx_y) are collinear.
    Args:
        tol: floating-point tolerance, to absorb rounding error
    Returns:
        True (collinear) / False (not collinear)
    """
    area = abs((xj - xi) * (tx_y - yi) - (yj - yi) * (tx_x - xi))
    return area < tol


# Method 3: add an edge when the nodes and the transmitter lie on the same line.
# Input: the coordinates stored under NodeFeatureAxis40/1/1750/1_1.txt plus the
# transmitter positions, and the blocks in the slitStregth40 folder; adj starts all-zero.


def gen_adj_by_transmitters(block_axis, tx_positions):

    """
    Build the adjacency matrix from a node-distance threshold.
    Args:
        block_axis: [2, N_Node] array; row 0 holds the y coordinates, row 1 the x ones
        tx_positions: coordinates given as [y1, x1, y2, x2, y3, x3]
    Returns:
        adj: adjacency matrix (undirected, with self-loops)
    """
    num_nodes = block_axis.shape[1]
    adj = lil_matrix((num_nodes, num_nodes))  # sparse matrix format

    for tx_idx in range(3):  # the three transmitters
        # The raw data comes from matlab, where indices start at 1
        tx_y, tx_x = tx_positions[tx_idx*2: tx_idx*2 + 2] - 1
        for i in range(num_nodes):
            y_i, x_i = block_axis[0, i], block_axis[1, i]
            for j in range(num_nodes):
                y_j, x_j = block_axis[0, j], block_axis[1, j]
                if are_collinear(x_i, y_i, x_j, y_j, tx_x, tx_y, tol=1e-5):
                    adj[i, j] = 1
                    adj[j, i] = 1  # symmetric, undirected graph
    return adj.astype(np.uint8)


def gen_adj_by_trans_Spectrum(block_axis, tx_positions):

    """
    Build the adjacency matrix from nodes being collinear with a transmitter.
    Args:
        block_axis: [2, N_Node] array; row 0 holds the y coordinates, row 1 the x ones
        tx_positions: coordinates given as [(x1,y1),(x2,y2),(x3,y3)]
    Returns:
        adj: adjacency matrix (undirected, with self-loops)
    """
    num_nodes = block_axis.shape[1]
    adj = lil_matrix((num_nodes, num_nodes))  # sparse matrix format
    for tx_idx in range(len(tx_positions)):
        tx_x, tx_y = tx_positions[tx_idx]  # position of transmitter tx_idx
        for i in range(num_nodes):
            y_i, x_i = block_axis[0, i], block_axis[1, i]
            for j in range(i, num_nodes):  # upper triangle, self-loops included
                y_j, x_j = block_axis[0, j], block_axis[1, j]
                # Are the three points collinear?
                if are_collinear(x_i, y_i, x_j, y_j, tx_x, tx_y, tol=1e-4):
                    adj[i, j] = 1
                    adj[j, i] = 1  # symmetric, undirected graph
    return adj.toarray().astype(int)



if __name__ == "__main__":

    dth = 9

    stand_freq = 5750
    stand_cof = 10 * np.log10(stand_freq)
    obs_cof = 0.1  # obstruction coefficient

    sigma = obs_cof*stand_cof  # depth threshold
    freq_all= [1750,2750,3750,4750,5750]

    folder_path = "/root/autodl-tmp/SpectrumNet/splitRSS_32/"
    data_lst = read_files_in_natural_order(folder_path)


    for dat_idx, dat_lst in enumerate(data_lst):

        data = np.load(dat_lst)
        block_axis = data['axis']

        # ------ adjacency matrix from distance and depth ---------- #
        adj_pth = dat_lst.replace('splitRSS', 'disdepthAdj')
        block_depth = data['depth']
        adj = gen_adj_by_dis_depth_Spectrum(block_axis, dth, block_depth, sigma)
        os.makedirs(os.path.dirname(adj_pth), exist_ok=True)
        np.savez_compressed(
            adj_pth,
            adj = adj.astype(np.uint8),
        )


    print('Done!')
