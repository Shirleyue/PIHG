from torch.utils.data import Dataset
import numpy as np
import torch
import random
import re
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
from pathlib import Path






sys.path.append(str(Path(__file__).parent.parent))
from SpectrumNet.readPng import load_png_data
from SpectrumNet.readNpz import load_npz_data
from SpectrumNet.utils import _get_height_code, read_files_in_natural_order, _get_frequency_code, adj_to_edge_index, generate_edge_type


class SpectrumDataset(Dataset):
    def __init__(self, txt_path,
                 div_block=False,
                 graph_name='disdepthAdj_32',
                 rss_name='splitRSS_32',
                 few_shot_ratio=1.0,
                 device=torch.device("cpu"),
                 noise_std = None):
        self.imgs_info = self.get_images(txt_path)
        self.div_block = div_block
        self.rss_name = rss_name
        self.graph_name = graph_name
        self.device = device
        self.noise_std = noise_std

        # Truncate the dataset according to few_shot_ratio
        if 0 < few_shot_ratio < 1.0:
            total_len = len(self.imgs_info)
            self.few_shot_len = int(total_len * few_shot_ratio)
            self.imgs_info = self.imgs_info[:self.few_shot_len]  # keep the first N entries

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
        return imgs_info

    def __getitem__(self, idx):
        img_path = self.imgs_info[idx]
        png_data = load_png_data(img_path[0])
        f_MHz = png_data['frequency']
        freq_idx = _get_frequency_code(f_MHz)
        heigtht_code = _get_height_code(png_data['height'])

        rss_map = png_data['data'].astype(np.float32) / 255.0
        rss_map = torch.from_numpy(rss_map).unsqueeze(0).to(self.device, non_blocking=True)  # [1,H,W]
        if self.noise_std is not None:
            mean = 0.0
            noise = np.random.normal(mean, self.noise_std, rss_map.shape).astype(np.float32) # Gaussian noise, same shape as rss_map, std = noise_std
            rss_map = rss_map + noise
            rss_map = np.clip(rss_map, 0.0, 1.0)

        building_map = load_npz_data(img_path[1])["arrays"]["inBldg_zyx"][heigtht_code]

        tx_img = np.load(img_path[2])
        tx_img[tx_img != 0] = 1

        inputs = torch.stack([
                    torch.from_numpy(building_map).to(self.device, non_blocking=True).float(),
                    torch.from_numpy(tx_img).to(self.device, non_blocking=True).float()
                ], dim=0)  # (2, 128, 128)

        graph = {}
        if self.div_block:
            block_folder = img_path[0][:-4].replace('png', self.rss_name)
            block_lst = read_files_in_natural_order(block_folder)

            for dat_lst in block_lst:
                parts = dat_lst.split('/')
                block_name = parts[-1]  # [0.npz]
                dat_idx = int (block_name.split('.')[0]) # strip '.npz' and convert to int

                data = np.load(dat_lst)
                node_obs_rss = data['rss']
                num_nodes = node_obs_rss.shape[0]
                node_obs_axis = data['axis']
                node_type = data['node_types_ids']
                node_freq = np.full(num_nodes, freq_idx)/5.0

                # Optional: block-local building/tx as tensors, if the graph needs them
                tx_img_blk = data['tx_img']            # [h,w] (numpy)
                building_blk = data['building']        # [h,w] (numpy)

                adj_pth = dat_lst.replace(self.rss_name, self.graph_name)
                adj = np.load(adj_pth)['adj']


                node_obs_rss = torch.as_tensor(node_obs_rss, dtype=torch.float32, device=self.device)
                node_freq     = torch.as_tensor(node_freq,     dtype=torch.float32, device=self.device)
                node_obs_axis = torch.as_tensor(node_obs_axis, dtype=torch.float32, device=self.device)  # [2,N]
                node_type_ids = torch.as_tensor(node_type,     dtype=torch.int,    device=self.device)
                adj           = torch.as_tensor(adj,           dtype=torch.float32, device=self.device)

                tx_img_blk_t   = torch.as_tensor(tx_img_blk,   dtype=torch.float32, device=self.device)
                building_blk_t = torch.as_tensor(building_blk, dtype=torch.float32, device=self.device)


                edge_index = adj_to_edge_index(adj)
                edge_index, edge_type = generate_edge_type(edge_index, node_type_ids)

                graph[dat_idx] = {
                    'node_obs_axis': node_obs_axis,
                    'node_obs_rss': node_obs_rss,
                    'node_freq': node_freq,
                    'node_type_ids': node_type_ids,
                    'edge_index': edge_index,
                    'edge_type': edge_type,

                    'tx_img': tx_img_blk_t,
                    'building': building_blk_t,

                }


        return inputs, rss_map, graph

    def __len__(self):
        return len(self.imgs_info)

class SpectrumDatasetFieldold(Dataset):
    def __init__(self, txt_path,
                 div_block=False,
                 load_W=True,
                 graph_name='disdepthAdj_32',
                 rss_name='splitRSS_32',
                 few_shot_ratio=1.0,
                 device=torch.device("cpu")):
        self.imgs_info = self.get_images(txt_path)
        self.div_block = div_block
        self.rss_name = rss_name
        self.graph_name = graph_name
        self.device = device
        self.load_W = load_W


        # Truncate the dataset according to few_shot_ratio
        if 0 < few_shot_ratio < 1.0:
            total_len = len(self.imgs_info)
            self.few_shot_len = int(total_len * few_shot_ratio)
            self.imgs_info = self.imgs_info[:self.few_shot_len]  # keep the first N entries

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
        return imgs_info

    def __getitem__(self, idx):
        img_path = self.imgs_info[idx]
        png_data = load_png_data(img_path[0])
        f_MHz = png_data['frequency']
        freq_idx = _get_frequency_code(f_MHz)
        heigtht_code = _get_height_code(png_data['height'])

        rss_map = png_data['data'].astype(np.float32) / 255.0
        rss_map = torch.from_numpy(rss_map).unsqueeze(0).float()  # [1,H,W]

        building_map = load_npz_data(img_path[1])["arrays"]["inBldg_zyx"][heigtht_code]

        tx_img = np.load(img_path[2])
        tx_img[tx_img != 0] = 1


        inputs = torch.stack([
            torch.from_numpy(building_map).float(),
            torch.from_numpy(tx_img).float()
        ], dim=0)  # (2, 128, 128)


        E_inc_dic = np.load(img_path[4])
        E_inc_inputs = E_inc_dic['E_inc_trad_2channel']

        E_inc_inputs = torch.from_numpy(E_inc_inputs).float()

        chi = torch.tensor([])  # empty placeholder tensor
        W = torch.tensor([])    # empty placeholder tensor
        if self.load_W:
            chi = E_inc_dic['chi']
            W_pth = '/root/autodl-tmp/SpectrumNet/W/06.DenseUrban/' + f'{f_MHz}.npz'
            W = np.load(W_pth)['W']

            chi = torch.from_numpy(chi)
            W = torch.from_numpy(W)

        inputs = torch.cat([inputs, E_inc_inputs], dim=0)  # (4, 128, 128)
        graph = {}
        if self.div_block:
            block_folder = img_path[0][:-4].replace('png', self.rss_name)
            block_lst = read_files_in_natural_order(block_folder)

            for dat_lst in block_lst:
                parts = dat_lst.split('/')
                block_name = parts[-1]  # [0.npz]
                dat_idx = int (block_name.split('.')[0]) # strip '.npz' and convert to int

                data = np.load(dat_lst)
                node_obs_rss = data['rss']
                num_nodes = node_obs_rss.shape[0]
                node_obs_axis = data['axis']
                node_type = data['node_types_ids']
                node_freq = np.full(num_nodes, freq_idx)/5.0

                # Optional: block-local building/tx as tensors, if the graph needs them
                tx_img_blk = data['tx_img']            # [h,w] (numpy)
                building_blk = data['building']        # [h,w] (numpy)

                adj_pth = dat_lst.replace(self.rss_name, self.graph_name)
                adj = np.load(adj_pth)['adj']





                node_obs_rss = torch.as_tensor(node_obs_rss, dtype=torch.float32)
                node_freq     = torch.as_tensor(node_freq,     dtype=torch.float32)
                node_obs_axis = torch.as_tensor(node_obs_axis, dtype=torch.float32)  # [2,N]
                node_type_ids = torch.as_tensor(node_type,     dtype=torch.int)
                adj           = torch.as_tensor(adj,           dtype=torch.float32)

                tx_img_blk_t   = torch.as_tensor(tx_img_blk,   dtype=torch.float32)
                building_blk_t = torch.as_tensor(building_blk, dtype=torch.float32)

                edge_index = adj_to_edge_index(adj)
                edge_index, edge_type = generate_edge_type(edge_index, node_type_ids)

                graph[dat_idx] = {
                    'node_obs_axis': node_obs_axis,
                    'node_obs_rss': node_obs_rss,
                    'node_freq': node_freq,
                    'node_type_ids': node_type_ids,
                    'edge_index': edge_index,
                    'edge_type': edge_type,

                    'tx_img': tx_img_blk_t,
                    'building': building_blk_t,

                }

        return inputs, rss_map, graph, chi, W

    def __len__(self):
        return len(self.imgs_info)

class SpectrumDatasetField(Dataset):
    def __init__(self, txt_path,
                 div_block=False,
                 load_W=True,
                 graph_name='disdepthAdj_32',
                 rss_name='splitRSS_32',
                 few_shot_ratio=1.0,
                 device=torch.device("cpu"),
                 noise_std=None):
        self.imgs_info = self.get_images(txt_path)
        self.div_block = div_block
        self.rss_name = rss_name
        self.graph_name = graph_name
        self.device = device
        self.load_W = load_W
        self.noise_std = noise_std

        # Truncate the dataset according to few_shot_ratio
        if 0 < few_shot_ratio < 1.0:
            total_len = len(self.imgs_info)
            self.few_shot_len = int(total_len * few_shot_ratio)
            self.imgs_info = self.imgs_info[:self.few_shot_len]  # keep the first N entries

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
        return imgs_info

    def __getitem__(self, idx):
        img_path = self.imgs_info[idx]
        png_data = load_png_data(img_path[0])
        f_MHz = png_data['frequency']
        freq_idx = _get_frequency_code(f_MHz)
        f_hz = f_MHz * 1e6
        heigtht_code = _get_height_code(png_data['height'])

        rss_map = png_data['data'].astype(np.float32) / 255.0
        rss_map = torch.from_numpy(rss_map).unsqueeze(0).float()  # [1,H,W]
        if self.noise_std is not None:
            mean = 0.0
            noise = np.random.normal(mean, self.noise_std, rss_map.shape).astype(np.float32) # Gaussian noise, same shape as rss_map, std = noise_std
            rss_map = rss_map + noise
            rss_map = np.clip(rss_map, 0.0, 1.0)


        building_map = load_npz_data(img_path[1])["arrays"]["inBldg_zyx"][heigtht_code]

        tx_img = np.load(img_path[2])
        tx_img[tx_img != 0] = 1


        inputs = torch.stack([
            torch.from_numpy(building_map).float(),
            torch.from_numpy(tx_img).float()
        ], dim=0)  # (2, 128, 128)


        E_inc_dic = np.load(img_path[4])
        E_inc_inputs = E_inc_dic['E_inc_trad_2channel']

        E_inc_inputs = torch.from_numpy(E_inc_inputs).float()

        chi = E_inc_dic['chi']
        chi = torch.from_numpy(chi)
        W = torch.tensor([])    # empty placeholder tensor
        if self.load_W:
            W_pth = '/root/autodl-tmp/SpectrumNet/W/06.DenseUrban/' + f'{f_MHz}.npz'
            W = np.load(W_pth)['W']

        inputs = torch.cat([inputs, E_inc_inputs], dim=0)  # (4, 128, 128)
        graph = {}
        if self.div_block:
            block_folder = img_path[0][:-4].replace('png', self.rss_name)
            block_lst = read_files_in_natural_order(block_folder)

            for dat_lst in block_lst:
                parts = dat_lst.split('/')
                block_name = parts[-1]  # [0.npz]
                dat_idx = int (block_name.split('.')[0]) # strip '.npz' and convert to int

                data = np.load(dat_lst)
                node_obs_rss = data['rss']
                num_nodes = node_obs_rss.shape[0]
                node_obs_axis = data['axis']
                node_type = data['node_types_ids']
                node_freq = np.full(num_nodes, freq_idx)/5.0

                # Optional: block-local building/tx as tensors, if the graph needs them
                tx_img_blk = data['tx_img']            # [h,w] (numpy)
                building_blk = data['building']        # [h,w] (numpy)

                adj_pth = dat_lst.replace(self.rss_name, self.graph_name)
                adj = np.load(adj_pth)['adj']





                node_obs_rss = torch.as_tensor(node_obs_rss, dtype=torch.float32)
                node_freq     = torch.as_tensor(node_freq,     dtype=torch.float32)
                node_obs_axis = torch.as_tensor(node_obs_axis, dtype=torch.float32)  # [2,N]
                node_type_ids = torch.as_tensor(node_type,     dtype=torch.int)
                adj           = torch.as_tensor(adj,           dtype=torch.float32)

                tx_img_blk_t   = torch.as_tensor(tx_img_blk,   dtype=torch.float32)
                building_blk_t = torch.as_tensor(building_blk, dtype=torch.float32)

                edge_index = adj_to_edge_index(adj)
                edge_index, edge_type = generate_edge_type(edge_index, node_type_ids)

                graph[dat_idx] = {
                    'node_obs_axis': node_obs_axis,
                    'node_obs_rss': node_obs_rss,
                    'node_freq': node_freq,
                    'node_type_ids': node_type_ids,
                    'edge_index': edge_index,
                    'edge_type': edge_type,

                    'tx_img': tx_img_blk_t,
                    'building': building_blk_t,

                }

        return inputs, rss_map, graph, chi, f_hz, W

    def __len__(self):
        return len(self.imgs_info)


class SpectrumDatasetDict(Dataset):
    def __init__(self, txt_path,
                 div_block=False,
                 graph_name='disdepthAdj_32',
                 rss_name='splitRSS_32',
                 few_shot_ratio=1.0,
                 device=torch.device("cpu")):
        self.imgs_info = self.get_images(txt_path)
        self.div_block = div_block
        self.rss_name = rss_name
        self.graph_name = graph_name
        self.device = device

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])

        # Truncate the dataset according to few_shot_ratio
        if 0 < few_shot_ratio < 1.0:
            total_len = len(self.imgs_info)
            self.few_shot_len = int(total_len * few_shot_ratio)
            self.imgs_info = self.imgs_info[:self.few_shot_len]  # keep the first N entries

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
        return imgs_info

    def __getitem__(self, idx):
        img_path = self.imgs_info[idx]
        png_data = load_png_data(img_path[0]) # /root/autodl-tmp/SpectrumNet/png/06.DenseUrban/T06C2D0054_n01_f04_ss_z00.png
        img_name = img_path[0].split('/')[-1]
        f_MHz = png_data['frequency']
        freq_idx = _get_frequency_code(f_MHz)
        heigtht_code = _get_height_code(png_data['height'])

        rss_map = png_data['data'].astype(np.float32) / 255.0
        rss_map = torch.from_numpy(rss_map).unsqueeze(0).to(self.device, non_blocking=True)  # [1,H,W]

        building_map = load_npz_data(img_path[1])["arrays"]["inBldg_zyx"][heigtht_code]

        tx_img = np.load(img_path[2])
        tx_img[tx_img != 0] = 1

        inputs = torch.stack([
                    torch.from_numpy(building_map).to(self.device, non_blocking=True).float(),
                    torch.from_numpy(tx_img).to(self.device, non_blocking=True).float(),
                    torch.from_numpy(building_map).to(self.device, non_blocking=True).float(),
                ], dim=0)  # (2, 128, 128)

        graph = {}
        if self.div_block:
            block_folder = img_path[0][:-4].replace('png', self.rss_name)
            block_lst = read_files_in_natural_order(block_folder)

            for dat_lst in block_lst:
                parts = dat_lst.split('/')
                block_name = parts[-1]  # [0.npz]
                dat_idx = int (block_name.split('.')[0]) # strip '.npz' and convert to int

                data = np.load(dat_lst)
                node_obs_rss = data['rss']
                num_nodes = node_obs_rss.shape[0]
                node_obs_axis = data['axis']
                node_type = data['node_types_ids']
                node_freq = np.full(num_nodes, freq_idx)/5.0

                # Optional: block-local building/tx as tensors, if the graph needs them
                tx_img_blk = data['tx_img']            # [h,w] (numpy)
                building_blk = data['building']        # [h,w] (numpy)

                adj_pth = dat_lst.replace(self.rss_name, self.graph_name)
                adj = np.load(adj_pth)['adj']


                node_obs_rss = torch.as_tensor(node_obs_rss, dtype=torch.float32, device=self.device)
                node_freq     = torch.as_tensor(node_freq,     dtype=torch.float32, device=self.device)
                node_obs_axis = torch.as_tensor(node_obs_axis, dtype=torch.float32, device=self.device)  # [2,N]
                node_type_ids = torch.as_tensor(node_type,     dtype=torch.int,    device=self.device)
                adj           = torch.as_tensor(adj,           dtype=torch.float32, device=self.device)

                tx_img_blk_t   = torch.as_tensor(tx_img_blk,   dtype=torch.float32, device=self.device)
                building_blk_t = torch.as_tensor(building_blk, dtype=torch.float32, device=self.device)


                edge_index = adj_to_edge_index(adj)
                edge_index, edge_type = generate_edge_type(edge_index, node_type_ids)

                graph[dat_idx] = {
                    'node_obs_axis': node_obs_axis,
                    'node_obs_rss': node_obs_rss,
                    'node_freq': node_freq,
                    'node_type_ids': node_type_ids,
                    'edge_index': edge_index,
                    'edge_type': edge_type,

                    'tx_img': tx_img_blk_t,
                    'building': building_blk_t,

                }

        out = {}
        out['image'] = self.transform_compose(rss_map)
        out['cond'] = self.transform_GY(inputs)
        out['img_name'] = img_name

        return out

    def __len__(self):
        return len(self.imgs_info)





if __name__ == "__main__":
    batch_size = 2
    Radio_train_spec = SpectrumDatasetField("./dataset/SpectrumNet/area_train.txt", div_block=True, load_W=False)


    dataloaders = DataLoader(Radio_train_spec, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch_idx, (inputs, rss_map, graph, chi, W) in enumerate(dataloaders):

        batchsize = inputs.shape[0]

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 4))

        axes[0, 0].imshow(inputs[0, 0], cmap='gray')
        axes[0, 0].set_title('building_map')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(inputs[0, 1], cmap='gray')
        axes[0, 1].set_title('tx_map')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(rss_map[0, 0])
        axes[0, 2].set_title('rss_map')
        axes[0, 2].axis('off')

        E_inc_comlex = inputs[0, 2,:,:] + 1j*inputs[0, 3,:,:]


        axes[1, 0].imshow(abs(E_inc_comlex))
        axes[1, 0].set_title('E_inc_abs')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(inputs[0, 2])
        axes[1, 1].set_title('E_inc_real')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(inputs[0, 3])
        axes[1, 2].set_title('E_inc_imag')
        axes[1, 2].axis('off')


        titleStr = 'inc_fig'
        plt.suptitle(titleStr)
        save_name = str(batch_idx) + '_' + titleStr + '.png'
        plt.savefig(save_name, format='png')
        plt.close()

        fig = plt.figure(figsize=(20, 20))
        rows = 4
        cols = 4
        grid = plt.GridSpec(rows, cols, wspace=0.05, hspace=0.05)
        block_num = len(graph)
        for i in range(block_num):

            graph_block = graph[i]
            node_obs_axis = graph_block['node_obs_axis']

            rss_map_block = np.zeros((32, 32))
            rows_coord = node_obs_axis[:, 0].int()  # row coordinates (int)
            cols_coord = node_obs_axis[:, 1].int()  # column coordinates (int)

            rss_map_block[rows_coord, cols_coord] = graph_block['node_type_ids']

            ax = fig.add_subplot(grid[i // cols, i % cols])
            ax.imshow(rss_map_block, cmap='viridis')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            # Block index in the lower-left corner
            ax.text(0.05, 0.05, str(i),
                transform=ax.transAxes,
                color='white',
                fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
        titleStr = 'block_fig'
        save_name = str(batch_idx) + '_' + titleStr + '.png'
        plt.savefig(save_name, format='png')
        plt.close(fig)

        if batch_idx >= 0:
            break

    print('done!')



