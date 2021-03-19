"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Author: Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""


import torch
import torch.nn as nn

from utils.nn_util import Seq
from utils.pointnet2_util import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule, gather_nd

BN_CONFIG = {"class": "GroupNorm", "num_groups": 4}


def quantize_flow(xyz: torch.Tensor, flow: torch.Tensor):
    dense_flow = xyz[:, 1].unsqueeze(1) - xyz[:, 0].unsqueeze(2)

    dist_mat = torch.cdist(xyz[:, 0] + flow[:, 0], xyz[:, 1])
    dist_mat = -dist_mat / 0.01

    flow01 = (dense_flow * torch.softmax(dist_mat, dim=-1).unsqueeze(-1)).sum(2)
    flow10 = -(dense_flow * torch.softmax(dist_mat, dim=-2).unsqueeze(-1)).sum(1)

    return torch.stack([flow01, flow10], dim=1)


class PointNet2(nn.Module):
    """
    Hypothesis Generation of the Segmentation Module.
        - Just a PN++ taking in xyz+flow and predicts transformation.
    """
    def __init__(self, bn):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256, radii=[0.1, 0.2],
                nsamples=[64, 64], mlps=[[3, 64, 64, 64], [3, 64, 64, 128]], use_xyz=True,
                bn=bn
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128, radius=0.4,
                nsample=64, mlp=[64 + 128, 128, 128, 256], use_xyz=True,
                bn=bn
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024],
                use_xyz=False,
                npoint=None, radius=None, nsample=None, bn=bn,
            )
        )
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 64], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64 + 128, 256, 128], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 1024, 256, 256], bn=bn))
        self.FC_layer = Seq(64).conv1d(64, bn=bn).conv1d(12, activation=None)

    def forward(self, xyz: torch.Tensor, flow: torch.Tensor):
        """
        :param xyz: (B, N, 3)
        :param flow: (B, N, 3)
        :return: Transformation prediction (B, N, 12)
        """
        l_xyz, l_features = [xyz], [flow.transpose(1, 2).contiguous()]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.FC_layer(l_features[0]).transpose(1, 2)


class MiniPointNets(nn.Module):
    def __init__(self, bn):
        super().__init__()
        self.u_pre_trans = Seq(12).conv2d(16, bn=bn).conv2d(64, bn=bn).conv2d(512, bn=bn)
        self.u_global_trans = Seq(512).conv2d(256, bn=bn).conv2d(256, bn=bn).conv2d(128, bn=bn)
        self.u_post_trans = Seq(512 + 256).conv2d(256, bn=bn).conv2d(64, bn=bn)\
            .conv2d(16, bn=bn).conv2d(1, activation=None)

    def forward(self, xyz: torch.Tensor, flow: torch.Tensor, sub_inds: torch.Tensor, pred_trans: torch.Tensor):
        nsample = sub_inds.size(-1)

        sub_ind0, sub_ind1 = sub_inds[:, 0, ...], sub_inds[:, 1, ...]
        xyz0_down = gather_nd(xyz[:, 0, ...], sub_ind0)
        flow0_down = gather_nd(flow[:, 0, ...], sub_ind0)
        trans0_down = gather_nd(pred_trans[:, 0, ...], sub_ind0)

        xyz1_down = gather_nd(xyz[:, 1, ...], sub_ind1)
        flow1_down = gather_nd(flow[:, 1, ...], sub_ind1)
        trans1_down = gather_nd(pred_trans[:, 1, ...], sub_ind1)

        Rs0 = trans0_down[..., :9].reshape(-1, nsample, 3, 3)
        ts0 = trans0_down[..., 9:].reshape(-1, nsample, 3)

        identity_mat = torch.eye(3, dtype=torch.float32, device=sub_inds.device).reshape(1, 1, 3, 3)

        xyz_ji0 = xyz1_down.unsqueeze(1) - xyz0_down.unsqueeze(2)
        rxji0 = torch.einsum('bnmd,bndk->bnmk', xyz_ji0, Rs0)
        rtsfi0 = torch.einsum('bnmd,bndk->bnmk', flow1_down.unsqueeze(1) + (ts0 + flow0_down).unsqueeze(2),
                              Rs0 + identity_mat)
        res0 = rxji0 - flow1_down.unsqueeze(1) - rtsfi0
        res0 = res0.permute(0, 3, 1, 2).contiguous()

        Rs1 = trans1_down[..., :9].reshape(-1, nsample, 3, 3)
        ts1 = trans1_down[..., 9:].reshape(-1, nsample, 3)
        xyz_ji1 = -xyz_ji0.transpose(1, 2)
        rxji1 = torch.einsum('bnmd,bndk->bnmk', xyz_ji1, Rs1)
        rtsfi1 = torch.einsum('bnmd,bndk->bnmk', flow0_down.unsqueeze(1) + (ts1 + flow1_down).unsqueeze(2),
                              Rs1 + identity_mat)
        res1 = rxji1 - flow0_down.unsqueeze(1) - rtsfi1
        res1 = res1.permute(0, 3, 1, 2).contiguous()
        xyz_info = torch.cat([xyz0_down.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, nsample),
                              xyz1_down.transpose(1, 2).unsqueeze(-2).expand(-1, -1, nsample, -1)], dim=1)
        U = torch.cat([res0, res1.transpose(-1, -2), xyz_info], dim=1)
        U = self.u_pre_trans(U)
        U_global0, _ = U.max(3, keepdim=True)
        U_global0 = self.u_global_trans(U_global0)
        U_global1, _ = U.max(2, keepdim=True)
        U_global1 = self.u_global_trans(U_global1)
        U = torch.cat([U, U_global0.expand(-1, -1, -1, nsample), U_global1.expand(-1, -1, nsample, -1)], dim=1)
        U = self.u_post_trans(U)
        U = U.squeeze(1)
        return U, res0, res1


class MotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trans_net = PointNet2(BN_CONFIG)
        self.verify_net = MiniPointNets(BN_CONFIG)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state'])

    def forward(self, xyz: torch.Tensor, flow: torch.Tensor, sub_inds: torch.Tensor):
        n_batch, _, n_point, _ = xyz.size()
        pred_trans = self.trans_net(xyz.reshape(2 * n_batch, n_point, 3),
                                    flow.reshape(2 * n_batch, n_point, 3)).reshape(n_batch, 2, n_point, -1)
        group_matrix, res0, res1 = self.verify_net(xyz, flow, sub_inds, pred_trans)
        return pred_trans, group_matrix, res0, res1
