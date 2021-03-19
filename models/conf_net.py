"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Part of this code is borrowed from OANet.

Author: Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""

import torch
import torch.nn as nn
from utils.pointnet2_util import gather_nd
from utils.nn_util import get_norm_layer


# Note: Model trained with G-N can be downloaded here:
#   https://drive.google.com/file/d/1bR96NNlFvNtRER4oOAGJlonvYh_cT6yi/view?usp=sharing
# BN_CONFIG = {"class": "GroupNorm", "num_groups": 4}
BN_CONFIG = {"class": "BatchNorm"}


def get_network_input(pc_i, pc_j, flow_ij):
    dist_ji = torch.cdist(pc_j, pc_i + flow_ij)
    snap_dist, best_j = dist_ji.min(dim=-2)
    matched_pc_j = gather_nd(pc_j, best_j)
    matches = torch.cat([pc_i, matched_pc_j, snap_dist.unsqueeze(-1)], dim=-1)
    return matches.transpose(-1, -2).detach()


class PointCN(nn.Module):
    def __init__(self, bn, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shortcut = None
        if out_channels != channels:
            self.shortcut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            get_norm_layer(bn, 2, in_size=channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            get_norm_layer(bn, 2, in_size=out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, channels, W, H)
        # return: (B, out_channels, W, H)
        out = self.conv(x)
        if self.shortcut:
            out = out + self.shortcut(x)
        else:
            out = out + x
        return out


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):
    def __init__(self, bn, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.short_cut = None
        if out_channels != channels:
            self.short_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            get_norm_layer(bn, 2, in_size=channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),  # b*c*n*1
            Transpose(1, 2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
            get_norm_layer(bn, 2, in_size=points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            Transpose(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            get_norm_layer(bn, 2, in_size=out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.short_cut:
            out = out + self.short_cut(x)
        else:
            out = out + x
        return out


# you can use this bottleneck block to prevent from overfiting when your dataset is small
class OAFilterBottleneck(nn.Module):
    def __init__(self, bn, channels, points1, points2, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.short_cut = None
        if out_channels != channels:
            self.short_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            get_norm_layer(bn, 2, in_size=channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),  # b*c*n*1
            Transpose(1, 2))
        self.conv2 = nn.Sequential(
            get_norm_layer(bn, 2, in_size=points1),
            nn.ReLU(),
            nn.Conv2d(points1, points2, kernel_size=1),
            get_norm_layer(bn, 2, in_size=points2),
            nn.ReLU(),
            nn.Conv2d(points2, points1, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            Transpose(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            get_norm_layer(bn, 2, in_size=out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.short_cut:
            out = out + self.short_cut(x)
        else:
            out = out + x
        return out


class DiffPool(nn.Module):
    def __init__(self, bn, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            get_norm_layer(bn, 2, in_size=in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x, return_weight: bool = False):
        """
        :param x: (B, C, N, 1)
        :param return_weight: whether or not to return pool weights.
        :return: (B, C, K, 1), i.e. pooled from N points to K points.
        """
        embed = self.conv(x)                                                    # (B, K, N, 1)
        S = torch.softmax(embed, dim=2).squeeze(3)                              # (B, K, N)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)        # (B, C, K, 1)

        if return_weight:
            return out, S
        else:
            return out


class DiffUnpool(nn.Module):
    def __init__(self, bn, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            get_norm_layer(bn, 2, in_size=in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down, return_weight: bool = False):
        """
        Based on the order of x_up, unpool x_down from K to N.
        :param x_up: (B, C, N, 1)
        :param x_down: (B, C, K, 1)
        :param return_weight: whether or not to return unpool weights.
        :return: (B, C, N, 1)
        """
        embed = self.conv(x_up)                                     # (B, K, N, 1)
        S = torch.softmax(embed, dim=1).squeeze(3)                  # (B, K, N)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)       # (B, C, N, 1)

        if return_weight:
            return out, S
        else:
            return out


class OANBlock(nn.Module):
    def __init__(self, bn, net_channels, input_channel, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)
        l2_nums = clusters

        self.l1_1 = []
        for _ in range(self.layer_num // 2):
            self.l1_1.append(PointCN(bn, channels))

        self.down1 = DiffPool(bn, channels, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num // 2):
            self.l2.append(OAFilter(bn, channels, l2_nums))

        self.up1 = DiffUnpool(bn, channels, l2_nums)

        self.l1_2 = []
        self.l1_2.append(PointCN(bn, 2 * channels, channels))
        for _ in range(self.layer_num // 2 - 1):
            self.l1_2.append(PointCN(bn, channels))

        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)

        self.output = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, data):
        x1_1 = self.conv1(data)
        x1_1 = self.l1_1(x1_1)
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = self.l1_2(torch.cat([x1_1, x_up], dim=1))
        logits = torch.squeeze(torch.squeeze(self.output(out), 3), 1)
        return logits


class ConfNet(nn.Module):
    def __init__(self, in_channel=5, net_channels=128, depth_each_stage=6, n_clusters=64):
        super().__init__()
        self.weights_init = OANBlock(BN_CONFIG, net_channels, in_channel + 2, depth_each_stage, n_clusters)
        self.weights_iter = OANBlock(BN_CONFIG, net_channels, in_channel + 3, depth_each_stage, n_clusters)

    def forward(self, data):
        logits_init = self.weights_init(data.unsqueeze(-1))  # (B, N), (B, K, N)
        logits_iter = self.weights_iter(torch.cat([data,
                                                   torch.relu(torch.tanh(logits_init.detach().unsqueeze(1)))],
                                                  dim=1).unsqueeze(-1))  # (B, N)
        return logits_init, logits_iter
