
import torch
from torch.nn import Module, Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import torch.nn as nn
from myradioUNet_modules import MyUNet

import numpy as np
from SpectrumNet.load_data import SpectrumDataset
from torch.utils.data import DataLoader

from torch_geometric.nn import RGATConv  # Relational GAT

class GAT(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 80)
        self.conv2 = GATConv(80, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class RGAT_HeteroNode(torch.nn.Module):
    def __init__(self, in_channels,
                 node_type_dim,
                 hidden_channels,
                 out_channels,
                 num_relations,
                 num_node_types,
                 num_heads):
        super().__init__()
        self.node_type_embed = nn.Embedding(num_node_types, node_type_dim)
        nn.init.xavier_uniform_(self.node_type_embed.weight)
        self.rgat1 = RGATConv(in_channels + node_type_dim, hidden_channels, num_relations,
                              heads=num_heads,
                              dropout=0.2)
        self.norm1 = nn.LayerNorm(hidden_channels * num_heads)
        self.rgat2 = RGATConv(hidden_channels * num_heads, out_channels, num_relations,
                              heads=1,
                              dropout=0.2)

    def forward(self, x, node_type_ids, edge_index, edge_type):
        node_type_vec = self.node_type_embed(node_type_ids)
        x = torch.cat([x, node_type_vec], dim=-1)
        x = self.rgat1(x, edge_index, edge_type)
        x = self.norm1(x)
        x = F.leaky_relu(x)
        x = self.rgat2(x, edge_index, edge_type)
        return x

class RGAT_HeteroNode_1_bk(torch.nn.Module):
    def __init__(self, in_channels,
                 node_type_dim,
                 hidden_channels,
                 out_channels,
                 num_relations,
                 num_node_types,
                 num_heads):
        super().__init__()
        self.in_dim = in_channels + node_type_dim
        self.hidden_dim = hidden_channels * num_heads
        self.node_type_embed = nn.Embedding(num_node_types, node_type_dim)
        self.rgat1 = RGATConv(in_channels + node_type_dim, hidden_channels, num_relations,
                              heads=num_heads,
                              dropout=0.2)
        self.norm1 = nn.LayerNorm(hidden_channels * num_heads)
        # Project when the input and output dims differ so the residual can be added
        if self.in_dim != self.hidden_dim:
            self.res_proj1 = nn.Linear(self.in_dim, self.hidden_dim)
        else:
            self.res_proj1 = nn.Identity()
        self.rgat2 = RGATConv(hidden_channels * num_heads, out_channels, num_relations,
                              heads=1,
                              dropout=0.2)
        self.norm2 = nn.LayerNorm(out_channels)

        if self.hidden_dim != out_channels:
            self.res_proj2 = nn.Linear(self.hidden_dim, out_channels)
        else:
            self.res_proj2 = nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.node_type_embed.weight, -0.1, 0.1)  # small-range uniform init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, node_type_ids, edge_index, edge_type):
        node_type_vec = self.node_type_embed(node_type_ids)
        x = torch.cat([x, node_type_vec], dim=-1)
        # --- first RGAT layer + residual ---
        h = self.rgat1(x, edge_index, edge_type)
        h = self.norm1(h)
        h = F.leaky_relu(h, negative_slope=0.2)
        x_res1 = self.res_proj1(x)  # align shapes
        h = h + x_res1  # residual

        # --- second RGAT layer ---
        h2 = self.rgat2(h, edge_index, edge_type)
        out = h2
        return out

class RGAT_HeteroNode_1(torch.nn.Module):
    def __init__(self, in_channels,
                 node_type_dim,
                 hidden_channels,
                 out_channels,
                 num_relations,
                 num_node_types,
                 num_heads):
        super().__init__()
        self.in_dim = in_channels + node_type_dim
        self.hidden_dim = hidden_channels * num_heads
        self.node_type_embed = nn.Embedding(num_node_types, node_type_dim)
        self.rgat1 = RGATConv(in_channels + node_type_dim, hidden_channels, num_relations,
                              heads=num_heads,
                              dropout=0.2)
        self.norm1 = nn.LayerNorm(hidden_channels * num_heads)
        # Project when the input and output dims differ so the residual can be added
        if self.in_dim != self.hidden_dim:
            self.res_proj1 = nn.Linear(self.in_dim, self.hidden_dim)
        else:
            self.res_proj1 = nn.Identity()
        self.rgat2 = RGATConv(hidden_channels * num_heads, out_channels, num_relations,
                              heads=1,
                              dropout=0.2)
        self.norm2 = nn.LayerNorm(out_channels)

        if self.hidden_dim != out_channels:
            self.res_proj2 = nn.Linear(self.hidden_dim, out_channels)
        else:
            self.res_proj2 = nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.node_type_embed.weight, -0.1, 0.1)  # small-range uniform init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, node_type_ids, edge_index, edge_type, return_attention=False):
        node_type_vec = self.node_type_embed(node_type_ids)
        x = torch.cat([x, node_type_vec], dim=-1)
        # --- first RGAT layer + residual ---

        if return_attention:
            h, (_, alpha1) = self.rgat1(
                x, edge_index, edge_type,
                return_attention_weights=True
            )
        else:
            h = self.rgat1(x, edge_index, edge_type)
            alpha1 = None
        h = self.norm1(h)
        h = F.leaky_relu(h, negative_slope=0.2)
        x_res1 = self.res_proj1(x)  # align shapes
        h = h + x_res1  # residual

        # --- second RGAT layer + residual ---
        if return_attention:
            out, (_, alpha2) = self.rgat2(
                h, edge_index, edge_type,
                return_attention_weights=True
            )
        else:
            out = self.rgat2(h, edge_index, edge_type)
            alpha2 = None
        if return_attention:
            return out, {"alpha1": alpha1, "alpha2": alpha2}
        return out


class RGAT_HeteroNode_1_layers(torch.nn.Module):
    def __init__(self, in_channels,
                 node_type_dim,
                 hidden_channels,
                 out_channels,
                 num_relations,
                 num_node_types,
                 num_heads,
                 graph_layers=2):
        super().__init__()
        self.in_dim = in_channels + node_type_dim
        self.hidden_dim = hidden_channels * num_heads
        self.graph_layers = graph_layers
        self.node_type_embed = nn.Embedding(num_node_types, node_type_dim)

        # RGAT layers with their matching normalisation layers
        self.rgat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        # First layer
        if graph_layers >= 1:
            self.rgat_layers.append(
                RGATConv(in_channels + node_type_dim, hidden_channels, num_relations,
                        heads=num_heads, dropout=0.2)
            )
            self.norm_layers.append(nn.LayerNorm(hidden_channels * num_heads))

            # residual projection of the first layer
            if self.in_dim != self.hidden_dim:
                self.res_projs.append(nn.Linear(self.in_dim, self.hidden_dim))
            else:
                self.res_projs.append(nn.Identity())

        # Intermediate layers
        for i in range(1, graph_layers - 1):
            self.rgat_layers.append(
                RGATConv(hidden_channels * num_heads, hidden_channels, num_relations,
                        heads=num_heads, dropout=0.2)
            )
            self.norm_layers.append(nn.LayerNorm(hidden_channels * num_heads))

            # residual projection of the intermediate layers (input and output dims
            # match, so usually no projection is needed)
            self.res_projs.append(nn.Identity())

        # Last layer (when there is more than one)
        if graph_layers > 1:
            self.rgat_layers.append(
                RGATConv(hidden_channels * num_heads, out_channels, num_relations,
                        heads=1, dropout=0.2)
            )
            # the last layer needs no normalisation
            self.norm_layers.append(nn.Identity())

            # residual projection of the last layer
            if hidden_channels * num_heads != out_channels:
                self.res_projs.append(nn.Linear(hidden_channels * num_heads, out_channels))
            else:
                self.res_projs.append(nn.Identity())

        # Special case: a single layer
        elif graph_layers == 1:
            self.rgat_layers.append(
                RGATConv(in_channels + node_type_dim, out_channels, num_relations,
                        heads=1, dropout=0.2)
            )
            # with a single layer no normalisation is needed
            self.norm_layers.append(nn.Identity())

            # residual projection
            if self.in_dim != out_channels:
                self.res_projs.append(nn.Linear(self.in_dim, out_channels))
            else:
                self.res_projs.append(nn.Identity())

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.node_type_embed.weight, -0.1, 0.1)  # small-range uniform init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, node_type_ids, edge_index, edge_type):
        node_type_vec = self.node_type_embed(node_type_ids)
        x = torch.cat([x, node_type_vec], dim=-1)

        h = x
        # Walk through every layer except the last
        for i in range(self.graph_layers - 1):
            h_next = self.rgat_layers[i](h, edge_index, edge_type)
            h_next = self.norm_layers[i](h_next)
            h_next = F.leaky_relu(h_next, negative_slope=0.2)

            # residual connection
            h_res = self.res_projs[i](h)
            h = h_next + h_res

        # Last layer, if any
        if self.graph_layers > 0:
            out = self.rgat_layers[-1](h, edge_index, edge_type)
            # the last layer needs neither activation nor normalisation
        else:
            # without RGAT layers, return the concatenated features directly
            out = h

        return out

class RGAT_HeteroNode_2(torch.nn.Module):
    def __init__(self, in_channels,
                 node_type_dim,
                 hidden_channels,
                 out_channels,
                 num_relations,
                 num_node_types,
                 num_heads):
        super().__init__()
        self.in_dim = in_channels + node_type_dim
        self.hidden_dim = hidden_channels * num_heads
        self.node_type_embed = nn.Embedding(num_node_types, node_type_dim)
        self.rgat1 = RGATConv(in_channels + node_type_dim, hidden_channels, num_relations,
                              heads=num_heads,
                              dropout=0.2)
        self.norm1 = nn.LayerNorm(hidden_channels * num_heads)
        # Project when the input and output dims differ so the residual can be added
        if self.in_dim != self.hidden_dim:
            self.res_proj1 = nn.Linear(self.in_dim, self.hidden_dim)
        else:
            self.res_proj1 = nn.Identity()
        self.mid_linear = nn.Sequential(
            nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.rgat2 = RGATConv(hidden_channels * num_heads, out_channels, num_relations,
                              heads=1,
                              dropout=0.2)
        self.norm2 = nn.LayerNorm(out_channels)

        if self.hidden_dim != out_channels:
            self.res_proj2 = nn.Linear(self.hidden_dim, out_channels)
        else:
            self.res_proj2 = nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.node_type_embed.weight, -0.1, 0.1)  # small-range uniform init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, node_type_ids, edge_index, edge_type):
        node_type_vec = self.node_type_embed(node_type_ids)
        x = torch.cat([x, node_type_vec], dim=-1)
        # --- first RGAT layer + residual ---
        h = self.rgat1(x, edge_index, edge_type)
        h = self.norm1(h)
        h = F.leaky_relu(h, negative_slope=0.2)
        x_res1 = self.res_proj1(x)  # align shapes
        h = h + x_res1  # residual

        # --- mid layer ---
        h = self.mid_linear(h)

        # --- second RGAT layer + residual ---
        h2 = self.rgat2(h, edge_index, edge_type)
        h2 = self.norm2(h2)
        h2 = F.leaky_relu(h2)
        x_res2 = self.res_proj2(h)
        out = h2 + x_res2  # residual
        return out




class CNN_Graph(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 device,
                 graph_in_dims=2,
                 node_type_dim=3,
                 graph_hidden_channels=32,
                 graph_out_channels=1,
                 graph_num_relations=6,
                 num_node_types=3,
                 graph_num_heads=2,
                 pt_path = None,
                 isGraph=False):
        super().__init__()
        self.encoder = MyUNet(in_channels, out_channels)
        self.isGraph = isGraph
        self.device = device
        if isGraph:
            self.graphEn = RGAT_HeteroNode(in_channels=graph_in_dims,
                            node_type_dim=node_type_dim,
                            hidden_channels=graph_hidden_channels,
                            out_channels=graph_out_channels,
                            num_relations=graph_num_relations,
                            num_node_types=num_node_types,
                            num_heads=graph_num_heads)
        self.init_weights()
        if pt_path is not None:
            self.init_from_pt(pt_path)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def init_from_pt(self, path):
        #  --- load the pretrained weights ---- #
        checkpoint = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        model_dict.update(checkpoint)
        self.load_state_dict(model_dict ,strict=False)

    def forward(self, x, graph):
        x = self.encoder(x.float())
        out = x.clone()
        if self.isGraph:  # when the graph module is enabled
            B, C, H, W = x.shape
            grid_size = 32
            num_block_x = W // grid_size
            block_num = len(graph)
            flat_out = out.squeeze().squeeze().view(-1)  # [H*W]
            for block_idx in range(block_num):  # iterate over the n blocks
                graph_block = graph[block_idx]
                node_obs_axis = graph_block['node_obs_axis'].squeeze(0).to(self.device)
                node_freq = graph_block['node_freq'].squeeze(0).to(self.device)
                node_type_ids = graph_block['node_type_ids'].squeeze(0).to(self.device)
                edge_index = graph_block['edge_index'].squeeze(0).to(self.device)
                edge_type = graph_block['edge_type'].squeeze(0).to(self.device)

                # Global offset of the current block
                y_offset = grid_size * (block_idx // num_block_x)
                x_offset = grid_size * (block_idx % num_block_x)

                offset = torch.tensor([y_offset, x_offset],
                                    device=self.device).reshape(2, 1)
                node_obs_axis_global = node_obs_axis + offset
                y_coords = node_obs_axis_global[0].long().clamp(0, H-1)  # global coords inside the whole area
                x_coords = node_obs_axis_global[1].long().clamp(0, W-1)

                linear_indices = y_coords * W + x_coords  # shape: [num_nodes]

                node_obs_rss = out[0, 0, y_coords, x_coords]   # RSS of this block, roughly predicted by the encoder
                node_obs_rss = node_obs_rss.squeeze()
                node_features = torch.stack([node_obs_rss, node_freq], dim=1)

                node_out = self.graphEn(node_features, node_type_ids, edge_index, edge_type)
                node_out = node_out.squeeze(-1)
                flat_out.scatter_(0, linear_indices, node_out.squeeze(-1))  # overwrite instead of accumulating

            out[0, 0] = flat_out.view(H, W)

        return out

class CNN_Graph_1(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 device,
                 graph_in_dims=2,
                 node_type_dim=3,
                 graph_hidden_channels=32,
                 graph_out_channels=1,
                 graph_num_relations=6,
                 num_node_types=3,
                 graph_num_heads=2,
                 pt_path = None,
                 isGraph=False):
        super().__init__()
        self.encoder = MyUNet(in_channels, out_channels)
        self.isGraph = isGraph
        self.device = device
        if isGraph:
            self.graphEn = RGAT_HeteroNode_1(in_channels=graph_in_dims,
                            node_type_dim=node_type_dim,
                            hidden_channels=graph_hidden_channels,
                            out_channels=graph_out_channels,
                            num_relations=graph_num_relations,
                            num_node_types=num_node_types,
                            num_heads=graph_num_heads)
        self.init_weights()
        if pt_path is not None:
            self.init_from_pt(pt_path)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def init_from_pt(self, path):
        #  --- load the pretrained weights ---- #
        checkpoint = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        model_dict.update(checkpoint)
        self.load_state_dict(model_dict ,strict=False)

    def forward(self, x, graph):
        x = self.encoder(x.float())
        out = x.clone()
        if self.isGraph:  # when the graph module is enabled
            B, C, H, W = x.shape
            grid_size = 32
            num_block_x = W // grid_size
            block_num = len(graph)
            flat_out = out.squeeze().squeeze().view(-1)  # [H*W]
            for block_idx in range(block_num):  # iterate over the n blocks
                graph_block = graph[block_idx]
                node_obs_axis = graph_block['node_obs_axis'].squeeze(0).to(self.device)
                node_freq = graph_block['node_freq'].squeeze(0).to(self.device)
                node_type_ids = graph_block['node_type_ids'].squeeze(0).to(self.device)
                edge_index = graph_block['edge_index'].squeeze(0).to(self.device)
                edge_type = graph_block['edge_type'].squeeze(0).to(self.device)

                # Global offset of the current block
                y_offset = grid_size * (block_idx // num_block_x)
                x_offset = grid_size * (block_idx % num_block_x)

                offset = torch.tensor([y_offset, x_offset],
                                    device=self.device).reshape(2, 1)
                node_obs_axis_global = node_obs_axis + offset
                y_coords = node_obs_axis_global[0].long().clamp(0, H-1)  # global coords inside the whole area
                x_coords = node_obs_axis_global[1].long().clamp(0, W-1)

                linear_indices = y_coords * W + x_coords  # shape: [num_nodes]

                node_obs_rss = out[0, 0, y_coords, x_coords]   # RSS of this block, roughly predicted by the encoder
                node_obs_rss = node_obs_rss.squeeze()
                node_features = torch.stack([node_obs_rss, node_freq], dim=1)

                node_out = self.graphEn(node_features, node_type_ids, edge_index, edge_type)
                node_out = node_out.squeeze(-1)
                flat_out.scatter_(0, linear_indices, node_out.squeeze(-1))  # overwrite instead of accumulating

            out[0, 0] = flat_out.view(H, W)

        return out


class INC_CNN_Graph(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 device,
                 graph_in_dims,
                 node_type_dim=3,
                 graph_hidden_channels=32,
                 graph_out_channels=1,
                 graph_num_relations=6,
                 num_node_types=3,
                 graph_num_heads=2,
                 pt_path = None,
                 isGraph=False):
        super().__init__()
        self.encoder = MyUNet(in_channels, out_channels)
        self.isGraph = isGraph
        self.device = device
        if isGraph:
            self.graphEn = RGAT_HeteroNode_1(in_channels=graph_in_dims,
                            node_type_dim=node_type_dim,
                            hidden_channels=graph_hidden_channels,
                            out_channels=graph_out_channels,
                            num_relations=graph_num_relations,
                            num_node_types=num_node_types,
                            num_heads=graph_num_heads)
        self.init_weights()
        if pt_path is not None:
            self.init_from_pt(pt_path)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def init_from_pt(self, path):
        #  --- load the pretrained weights ---- #
        checkpoint = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        model_dict.update(checkpoint)
        self.load_state_dict(model_dict ,strict=False)

    def forward(self, x, graph, return_attention=False):
        x = self.encoder(x.float())
        out = x.clone()
        block_attention = {} if return_attention else None
        if self.isGraph:  # when the graph module is enabled
            B, C, H, W = x.shape
            grid_size = 32
            num_block_x = W // grid_size
            block_num = len(graph)
            base_channel = out[:, 0:1, :, :]  # use the first channel as the base
            flat_out = base_channel.squeeze().squeeze().view(-1)  # [H*W]
            for block_idx in range(block_num):  # iterate over the n blocks
                graph_block = graph[block_idx]
                node_obs_axis = graph_block['node_obs_axis'].squeeze(0).to(self.device)
                node_freq = graph_block['node_freq'].squeeze(0).to(self.device)
                node_type_ids = graph_block['node_type_ids'].squeeze(0).to(self.device)
                edge_index = graph_block['edge_index'].squeeze(0).to(self.device)
                edge_type = graph_block['edge_type'].squeeze(0).to(self.device)

                # Global offset of the current block
                y_offset = grid_size * (block_idx // num_block_x)
                x_offset = grid_size * (block_idx % num_block_x)

                offset = torch.tensor([y_offset, x_offset],
                                    device=self.device).reshape(2, 1)
                node_obs_axis_global = node_obs_axis + offset
                y_coords = node_obs_axis_global[0].long().clamp(0, H-1)  # global coords inside the whole area
                x_coords = node_obs_axis_global[1].long().clamp(0, W-1)

                linear_indices = y_coords * W + x_coords  # shape: [num_nodes]


                node_obs_ch0 = out[0, 0, y_coords, x_coords].squeeze() # channel 0 values
                node_obs_ch1 = out[0, 1, y_coords, x_coords].squeeze()  # channel 1 values
                node_features = torch.stack([node_obs_ch0, node_obs_ch1, node_freq], dim=1)

                if return_attention:
                    node_out, attention_weights = self.graphEn(
                        node_features,
                        node_type_ids,
                        edge_index,
                        edge_type,
                        return_attention=True,
                    )
                    block_attention[block_idx] = attention_weights
                else:
                    node_out = self.graphEn(node_features, node_type_ids, edge_index, edge_type)
                node_out = node_out.squeeze(-1)
                flat_out.scatter_(0, linear_indices, node_out.squeeze(-1))  # overwrite instead of accumulating

            out = flat_out.view(1, 1, H, W)  # final output shape (1, 1, H, W)
        else:
            out_complex = out[:, 0:1, :, :] + 1j*out[:, 1:2, :, :]
            out = torch.abs(out_complex)
        if return_attention:
            return out, block_attention
        return out


class Phy_CNN_Graph(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 device,
                 graph_in_dims,
                 node_type_dim=2,
                 graph_hidden_channels=32,
                 graph_out_channels=1,
                 graph_num_relations=6,
                 num_node_types=3,
                 graph_num_heads=2,
                 pt_path = None,
                 isGraph=False):
        super().__init__()
        self.encoder_1 = MyUNet(in_channels, out_channels)
        self.encoder_2 = MyUNet(1, 1)
        self.isGraph = isGraph
        self.device = device
        if isGraph:
            self.graphEn = RGAT_HeteroNode_1(in_channels=graph_in_dims,
                            node_type_dim=node_type_dim,
                            hidden_channels=graph_hidden_channels,
                            out_channels=graph_out_channels,
                            num_relations=graph_num_relations,
                            num_node_types=num_node_types,
                            num_heads=graph_num_heads)
        self.init_weights()
        if pt_path is not None:
            self.init_from_pt(pt_path)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def init_from_pt(self, path):
        #  --- load the pretrained weights ---- #
        checkpoint = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        model_dict.update(checkpoint)
        self.load_state_dict(model_dict ,strict=False)

    def forward(self, x, graph, return_attention=False, return_e_tot=False):
        x = self.encoder_1(x.float())
        out1 = x.clone()
        x_complex = x[:, 0:1, :, :] + 1j*x[:, 1:2, :, :]
        x = torch.abs(x_complex)
        if return_e_tot:
            return x
        x = self.encoder_2(x)
        out = x.clone()
        block_attention = {} if return_attention else None
        if self.isGraph:  # when the graph module is enabled
            B, C, H, W = x.shape
            grid_size = 32
            num_block_x = W // grid_size
            block_num = len(graph)
            base_channel = out[:, 0:1, :, :]  # use the first channel as the base
            flat_out = base_channel.squeeze().squeeze().view(-1)  # [H*W]
            for block_idx in range(block_num):  # iterate over the n blocks
                graph_block = graph[block_idx]
                node_obs_axis = graph_block['node_obs_axis'].squeeze(0).to(self.device)
                node_freq = graph_block['node_freq'].squeeze(0).to(self.device)
                node_type_ids = graph_block['node_type_ids'].squeeze(0).to(self.device)
                edge_index = graph_block['edge_index'].squeeze(0).to(self.device)
                edge_type = graph_block['edge_type'].squeeze(0).to(self.device)

                # Global offset of the current block
                y_offset = grid_size * (block_idx // num_block_x)
                x_offset = grid_size * (block_idx % num_block_x)

                offset = torch.tensor([y_offset, x_offset],
                                    device=self.device).reshape(2, 1)
                node_obs_axis_global = node_obs_axis + offset
                y_coords = node_obs_axis_global[0].long().clamp(0, H-1)  # global coords inside the whole area
                x_coords = node_obs_axis_global[1].long().clamp(0, W-1)

                linear_indices = y_coords * W + x_coords  # shape: [num_nodes]

                node_obs_ch0 = out[0, 0, y_coords, x_coords].squeeze() # channel 0 values
                node_features = torch.stack([node_obs_ch0, node_freq], dim=1)

                if return_attention:
                    node_out, attention_weights = self.graphEn(
                        node_features,
                        node_type_ids,
                        edge_index,
                        edge_type,
                        return_attention=True,
                    )
                    block_attention[block_idx] = attention_weights
                else:
                    node_out = self.graphEn(node_features, node_type_ids, edge_index, edge_type)
                node_out = node_out.squeeze(-1)
                flat_out.scatter_(0, linear_indices, node_out.squeeze(-1))  # overwrite instead of accumulating

            out = flat_out.view(1, 1, H, W)  # final output shape (1, 1, H, W)
        else:
            out = x
        if return_attention:
            return out, block_attention
        return out1, out



class INC_CNN_Graph_layers(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 device,
                 graph_in_dims,
                 node_type_dim=3,
                 graph_hidden_channels=32,
                 graph_out_channels=1,
                 graph_num_relations=6,
                 num_node_types=3,
                 graph_num_heads=2,
                 pt_path = None,
                 isGraph=False,
                 graph_layers = 2):
        super().__init__()
        self.encoder = MyUNet(in_channels, out_channels)
        self.isGraph = isGraph
        self.device = device
        if isGraph:
            self.graphEn = RGAT_HeteroNode_1_layers(in_channels=graph_in_dims,
                            node_type_dim=node_type_dim,
                            hidden_channels=graph_hidden_channels,
                            out_channels=graph_out_channels,
                            num_relations=graph_num_relations,
                            num_node_types=num_node_types,
                            num_heads=graph_num_heads,
                            graph_layers=graph_layers)
        self.init_weights()
        if pt_path is not None:
            self.init_from_pt(pt_path)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def init_from_pt(self, path):
        #  --- load the pretrained weights ---- #
        checkpoint = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        model_dict.update(checkpoint)
        self.load_state_dict(model_dict ,strict=False)

    def forward(self, x, graph):
        x = self.encoder(x.float())
        out = x.clone()
        if self.isGraph:  # when the graph module is enabled
            B, C, H, W = x.shape
            grid_size = 32
            num_block_x = W // grid_size
            block_num = len(graph)
            base_channel = out[:, 0:1, :, :]  # use the first channel as the base
            flat_out = base_channel.squeeze().squeeze().view(-1)  # [H*W]
            for block_idx in range(block_num):  # iterate over the n blocks
                graph_block = graph[block_idx]
                node_obs_axis = graph_block['node_obs_axis'].squeeze(0).to(self.device)
                node_freq = graph_block['node_freq'].squeeze(0).to(self.device)
                node_type_ids = graph_block['node_type_ids'].squeeze(0).to(self.device)
                edge_index = graph_block['edge_index'].squeeze(0).to(self.device)
                edge_type = graph_block['edge_type'].squeeze(0).to(self.device)

                # Global offset of the current block
                y_offset = grid_size * (block_idx // num_block_x)
                x_offset = grid_size * (block_idx % num_block_x)

                offset = torch.tensor([y_offset, x_offset],
                                    device=self.device).reshape(2, 1)
                node_obs_axis_global = node_obs_axis + offset
                y_coords = node_obs_axis_global[0].long().clamp(0, H-1)  # global coords inside the whole area
                x_coords = node_obs_axis_global[1].long().clamp(0, W-1)

                linear_indices = y_coords * W + x_coords  # shape: [num_nodes]


                node_obs_ch0 = out[0, 0, y_coords, x_coords].squeeze() # channel 0 values
                node_obs_ch1 = out[0, 1, y_coords, x_coords].squeeze()  # channel 1 values
                node_features = torch.stack([node_obs_ch0, node_obs_ch1, node_freq], dim=1)


                node_out = self.graphEn(node_features, node_type_ids, edge_index, edge_type)
                node_out = node_out.squeeze(-1)
                flat_out.scatter_(0, linear_indices, node_out.squeeze(-1))  # overwrite instead of accumulating

            out = flat_out.view(1, 1, H, W)  # final output shape (1, 1, H, W)
        return out


class INC_CNN_Graph_CNN(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 device,
                 graph_in_dims,
                 node_type_dim=3,
                 graph_hidden_channels=32,
                 graph_out_channels=1,
                 graph_num_relations=6,
                 num_node_types=3,
                 graph_num_heads=2,
                 pt_path = None,
                 isGraph=False,
                 isdecoder=True):
        super().__init__()
        self.encoder = MyUNet(in_channels, out_channels)
        self.isGraph = isGraph
        self.device = device
        self.isdecoder = isdecoder
        if isGraph:
            self.graphEn = RGAT_HeteroNode_1(in_channels=graph_in_dims,
                            node_type_dim=node_type_dim,
                            hidden_channels=graph_hidden_channels,
                            out_channels=graph_out_channels,
                            num_relations=graph_num_relations,
                            num_node_types=num_node_types,
                            num_heads=graph_num_heads)
        if isdecoder:
            self.decoder = MyUNet(1, 1)
        self.init_weights()
        if pt_path is not None:
            self.init_from_pt(pt_path)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def init_from_pt(self, path):
        #  --- load the pretrained weights ---- #
        checkpoint = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        model_dict.update(checkpoint)
        self.load_state_dict(model_dict ,strict=False)

    def forward(self, x, graph):
        x = self.encoder(x.float())
        out = x.clone()
        if self.isGraph:  # when the graph module is enabled
            B, C, H, W = x.shape
            grid_size = 32
            num_block_x = W // grid_size
            block_num = len(graph)
            base_channel = out[:, 0:1, :, :]  # use the first channel as the base
            flat_out = base_channel.squeeze().squeeze().view(-1)  # [H*W]
            for block_idx in range(block_num):  # iterate over the n blocks
                graph_block = graph[block_idx]
                node_obs_axis = graph_block['node_obs_axis'].squeeze(0).to(self.device)
                node_freq = graph_block['node_freq'].squeeze(0).to(self.device)
                node_type_ids = graph_block['node_type_ids'].squeeze(0).to(self.device)
                edge_index = graph_block['edge_index'].squeeze(0).to(self.device)
                edge_type = graph_block['edge_type'].squeeze(0).to(self.device)

                # Global offset of the current block
                y_offset = grid_size * (block_idx // num_block_x)
                x_offset = grid_size * (block_idx % num_block_x)

                offset = torch.tensor([y_offset, x_offset],
                                    device=self.device).reshape(2, 1)
                node_obs_axis_global = node_obs_axis + offset
                y_coords = node_obs_axis_global[0].long().clamp(0, H-1)  # global coords inside the whole area
                x_coords = node_obs_axis_global[1].long().clamp(0, W-1)

                linear_indices = y_coords * W + x_coords  # shape: [num_nodes]


                node_obs_ch0 = out[0, 0, y_coords, x_coords].squeeze() # channel 0 values
                node_obs_ch1 = out[0, 1, y_coords, x_coords].squeeze()  # channel 1 values
                node_features = torch.stack([node_obs_ch0, node_obs_ch1, node_freq], dim=1)


                node_out = self.graphEn(node_features, node_type_ids, edge_index, edge_type)
                node_out = node_out.squeeze(-1)
                flat_out.scatter_(0, linear_indices, node_out.squeeze(-1))  # overwrite instead of accumulating

            out = flat_out.view(1, 1, H, W)  # final output shape (1, 1, H, W)
        if self.decoder:
            out = self.decoder(out)
        return out



class INC_CNN_GAT(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 device,
                 graph_in_dims,
                 node_type_dim=3,
                 graph_hidden_channels=32,
                 graph_out_channels=1,
                 graph_num_relations=6,
                 num_node_types=3,
                 graph_num_heads=2,
                 pt_path = None,
                 isGraph=False):
        super().__init__()
        self.encoder = MyUNet(in_channels, out_channels)
        self.isGraph = isGraph
        self.device = device
        if isGraph:
            self.graphEn = GAT(num_node_features=graph_in_dims,
                            num_classes=graph_out_channels)
        self.init_weights()
        if pt_path is not None:
            self.init_from_pt(pt_path)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def init_from_pt(self, path):
        #  --- load the pretrained weights ---- #
        checkpoint = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        model_dict.update(checkpoint)
        self.load_state_dict(model_dict ,strict=False)

    def forward(self, x, graph):
        x = self.encoder(x.float())
        out = x.clone()
        if self.isGraph:  # when the graph module is enabled
            B, C, H, W = x.shape
            grid_size = 32
            num_block_x = W // grid_size
            block_num = len(graph)
            base_channel = out[:, 0:1, :, :]  # use the first channel as the base
            flat_out = base_channel.squeeze().squeeze().view(-1)  # [H*W]
            for block_idx in range(block_num):  # iterate over the n blocks
                graph_block = graph[block_idx]
                node_obs_axis = graph_block['node_obs_axis'].squeeze(0).to(self.device)
                node_freq = graph_block['node_freq'].squeeze(0).to(self.device)
                edge_index = graph_block['edge_index'].squeeze(0).to(self.device)

                # Global offset of the current block
                y_offset = grid_size * (block_idx // num_block_x)
                x_offset = grid_size * (block_idx % num_block_x)

                offset = torch.tensor([y_offset, x_offset],
                                    device=self.device).reshape(2, 1)
                node_obs_axis_global = node_obs_axis + offset
                y_coords = node_obs_axis_global[0].long().clamp(0, H-1)  # global coords inside the whole area
                x_coords = node_obs_axis_global[1].long().clamp(0, W-1)

                linear_indices = y_coords * W + x_coords  # shape: [num_nodes]

                node_obs_ch0 = out[0, 0, y_coords, x_coords].squeeze() # channel 0 values
                node_obs_ch1 = out[0, 1, y_coords, x_coords].squeeze()  # channel 1 values
                node_features = torch.stack([node_obs_ch0, node_obs_ch1, node_freq], dim=1)

                node_out = self.graphEn(node_features,edge_index)

                node_out = node_out.squeeze(-1)
                flat_out.scatter_(0, linear_indices, node_out.squeeze(-1))  # overwrite instead of accumulating

            out = flat_out.view(1, 1, H, W)  # final output shape (1, 1, H, W)
        else:
            out_complex = out[:, 0:1, :, :] + 1j*out[:, 1:2, :, :]
            out = torch.abs(out_complex)
        return out


class INC_CNN_Graph_2(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 device,
                 graph_in_dims,
                 node_type_dim=3,
                 graph_hidden_channels=32,
                 graph_out_channels=1,
                 graph_num_relations=6,
                 num_node_types=3,
                 graph_num_heads=2,
                 pt_path = None,
                 isGraph=False):
        super().__init__()
        self.encoder = MyUNet(in_channels, out_channels)
        self.isGraph = isGraph
        self.device = device
        if isGraph:
            self.graphEn = RGAT_HeteroNode_1(in_channels=graph_in_dims,
                            node_type_dim=node_type_dim,
                            hidden_channels=graph_hidden_channels,
                            out_channels=graph_out_channels,
                            num_relations=graph_num_relations,
                            num_node_types=num_node_types,
                            num_heads=graph_num_heads)
        self.init_weights()
        if pt_path is not None:
            self.init_from_pt(pt_path)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def init_from_pt(self, path):
        #  --- load the pretrained weights ---- #
        checkpoint = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        model_dict.update(checkpoint)
        self.load_state_dict(model_dict ,strict=False)

    def forward(self, x, graph):
        x = self.encoder(x.float())
        out = x.clone()
        if self.isGraph:  # when the graph module is enabled
            B, C, H, W = x.shape
            grid_size = 32
            num_block_x = W // grid_size
            block_num = len(graph)
            flat_out = out.squeeze().squeeze().view(-1)  # [H*W]
            for block_idx in range(block_num):  # iterate over the n blocks
                graph_block = graph[block_idx]
                node_obs_axis = graph_block['node_obs_axis'].squeeze(0).to(self.device)
                node_freq = graph_block['node_freq'].squeeze(0).to(self.device)
                node_type_ids = graph_block['node_type_ids'].squeeze(0).to(self.device)
                edge_index = graph_block['edge_index'].squeeze(0).to(self.device)
                edge_type = graph_block['edge_type'].squeeze(0).to(self.device)

                # Global offset of the current block
                y_offset = grid_size * (block_idx // num_block_x)
                x_offset = grid_size * (block_idx % num_block_x)

                offset = torch.tensor([y_offset, x_offset],
                                    device=self.device).reshape(2, 1)
                node_obs_axis_global = node_obs_axis + offset
                y_coords = node_obs_axis_global[0].long().clamp(0, H-1)  # global coords inside the whole area
                x_coords = node_obs_axis_global[1].long().clamp(0, W-1)

                linear_indices = y_coords * W + x_coords  # shape: [num_nodes]

                node_obs_rss = out[0, 0, y_coords, x_coords]   # RSS of this block, roughly predicted by the encoder
                node_obs_rss = node_obs_rss.squeeze()
                node_features = torch.stack([node_obs_rss, node_freq], dim=1)

                node_out = self.graphEn(node_features, node_type_ids, edge_index, edge_type)
                node_out = node_out.squeeze(-1)
                flat_out.scatter_(0, linear_indices, node_out.squeeze(-1))  # overwrite instead of accumulating

            out[0, 0] = flat_out.view(H, W)

        return out


class CNN_Graph_2(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 device,
                 graph_in_dims=2,
                 node_type_dim=3,
                 graph_hidden_channels=32,
                 graph_out_channels=1,
                 graph_num_relations=6,
                 num_node_types=3,
                 graph_num_heads=2,
                 pt_path = None,
                 isGraph=False):
        super().__init__()
        self.encoder = MyUNet(in_channels, out_channels)
        self.isGraph = isGraph
        self.device = device
        if isGraph:
            self.graphEn = RGAT_HeteroNode_2(in_channels=graph_in_dims,
                            node_type_dim=node_type_dim,
                            hidden_channels=graph_hidden_channels,
                            out_channels=graph_out_channels,
                            num_relations=graph_num_relations,
                            num_node_types=num_node_types,
                            num_heads=graph_num_heads)
        self.init_weights()
        if pt_path is not None:
            self.init_from_pt(pt_path)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def init_from_pt(self, path):
        #  --- load the pretrained weights ---- #
        checkpoint = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        model_dict.update(checkpoint)
        self.load_state_dict(model_dict ,strict=False)

    def forward(self, x, graph):
        x = self.encoder(x.float())
        out = x.clone()
        if self.isGraph:  # when the graph module is enabled
            B, C, H, W = x.shape
            grid_size = 32
            num_block_x = W // grid_size
            block_num = len(graph)
            flat_out = out.squeeze().squeeze().view(-1)  # [H*W]
            for block_idx in range(block_num):  # iterate over the n blocks
                graph_block = graph[block_idx]
                node_obs_axis = graph_block['node_obs_axis'].squeeze(0).to(self.device)
                node_freq = graph_block['node_freq'].squeeze(0).to(self.device)
                node_type_ids = graph_block['node_type_ids'].squeeze(0).to(self.device)
                edge_index = graph_block['edge_index'].squeeze(0).to(self.device)
                edge_type = graph_block['edge_type'].squeeze(0).to(self.device)

                # Global offset of the current block
                y_offset = grid_size * (block_idx // num_block_x)
                x_offset = grid_size * (block_idx % num_block_x)

                offset = torch.tensor([y_offset, x_offset],
                                    device=self.device).reshape(2, 1)
                node_obs_axis_global = node_obs_axis + offset
                y_coords = node_obs_axis_global[0].long().clamp(0, H-1)  # global coords inside the whole area
                x_coords = node_obs_axis_global[1].long().clamp(0, W-1)

                linear_indices = y_coords * W + x_coords  # shape: [num_nodes]

                node_obs_rss = out[0, 0, y_coords, x_coords]   # RSS of this block, roughly predicted by the encoder
                node_obs_rss = node_obs_rss.squeeze()
                node_features = torch.stack([node_obs_rss, node_freq], dim=1)

                node_out = self.graphEn(node_features, node_type_ids, edge_index, edge_type)
                node_out = node_out.squeeze(-1)
                flat_out.scatter_(0, linear_indices, node_out.squeeze(-1))  # overwrite instead of accumulating

            out[0, 0] = flat_out.view(H, W)

        return out


class CNN_Graph_3(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 device,
                 graph_in_dims=2,
                 node_type_dim=3,
                 graph_hidden_channels=32,
                 graph_out_channels=1,
                 graph_num_relations=6,
                 num_node_types=3,
                 graph_num_heads=2,
                 pt_path = None,
                 isGraph=False):
        super().__init__()
        self.encoder = MyUNet(in_channels, out_channels)
        self.isGraph = isGraph
        self.device = device
        if isGraph:
            self.graphEn = RGAT_HeteroNode_2(in_channels=graph_in_dims,
                            node_type_dim=node_type_dim,
                            hidden_channels=graph_hidden_channels,
                            out_channels=graph_out_channels,
                            num_relations=graph_num_relations,
                            num_node_types=num_node_types,
                            num_heads=graph_num_heads)
        self.init_weights()
        if pt_path is not None:
            self.init_from_pt(pt_path)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def init_from_pt(self, path):
        #  --- load the pretrained weights ---- #
        checkpoint = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        model_dict.update(checkpoint)
        self.load_state_dict(model_dict ,strict=False)

    def forward(self, x, graph):
        x = self.encoder(x.float())
        out = x.clone()
        if self.isGraph:  # when the graph module is enabled
            B, C, H, W = x.shape
            grid_size = 32
            num_block_x = W // grid_size
            block_num = len(graph)
            flat_out = out.squeeze().squeeze().view(-1)  # [H*W]
            for block_idx in range(block_num):  # iterate over the n blocks
                graph_block = graph[block_idx]
                node_obs_axis = graph_block['node_obs_axis'].squeeze(0).to(self.device)
                node_freq = graph_block['node_freq'].squeeze(0).to(self.device)

                node_txt = graph_block['tx_img'].squeeze(0).to(self.device)
                node_txt = node_txt[node_obs_axis[0].long(), node_obs_axis[1].long()]

                node_type_ids = graph_block['node_type_ids'].squeeze(0).to(self.device)
                edge_index = graph_block['edge_index'].squeeze(0).to(self.device)
                edge_type = graph_block['edge_type'].squeeze(0).to(self.device)

                # Global offset of the current block
                y_offset = grid_size * (block_idx // num_block_x)
                x_offset = grid_size * (block_idx % num_block_x)

                offset = torch.tensor([y_offset, x_offset],
                                    device=self.device).reshape(2, 1)
                node_obs_axis_global = node_obs_axis + offset
                y_coords = node_obs_axis_global[0].long().clamp(0, H-1)  # global coords inside the whole area
                x_coords = node_obs_axis_global[1].long().clamp(0, W-1)

                linear_indices = y_coords * W + x_coords  # shape: [num_nodes]

                node_obs_rss = out[0, 0, y_coords, x_coords]   # RSS of this block, roughly predicted by the encoder
                node_obs_rss = node_obs_rss.squeeze()
                node_features = torch.stack([node_obs_rss, node_freq, node_txt], dim=1)

                node_out = self.graphEn(node_features, node_type_ids, edge_index, edge_type)
                node_out = node_out.squeeze(-1)
                flat_out.scatter_(0, linear_indices, node_out.squeeze(-1))  # overwrite instead of accumulating

            out[0, 0] = flat_out.view(H, W)

        return out




if __name__ == "__main__":
    conv = RGATConv(in_channels=16, out_channels=8, num_relations=3, heads=2)
    print(list(conv.named_parameters()))  # inspect the parameters that are actually trained
