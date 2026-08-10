
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


if __name__ == "__main__":
    conv = RGATConv(in_channels=16, out_channels=8, num_relations=3, heads=2)
    print(list(conv.named_parameters()))  # inspect the parameters that are actually trained
