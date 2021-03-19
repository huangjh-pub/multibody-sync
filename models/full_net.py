"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Author: Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""


import yaml
import torch
import torch.nn as nn
from pathlib import Path
from models.flow_net import FlowNet
from models.conf_net import ConfNet, get_network_input
from models.mot_net import MotNet
import itertools
from utils import pointnet2_util
from utils.nn_util import break_leading_dim, knead_leading_dims
from utils.sync_util import sync_perm, sync_motion_seg, motion_synchronization_spectral, fit_motion_svd_batch


def compose_dense(comp_dict: dict, n_view: int, identity_filler: torch.Tensor):
    dense_mat = []
    for view_i, view_j in itertools.product(range(n_view), range(n_view)):
        if view_j > view_i:
            mat_ij = comp_dict[(view_i, view_j)]
            dense_mat.append(mat_ij)
        else:
            dense_mat.append(identity_filler)
    dense_mat = torch.stack(dense_mat, dim=1)
    dense_mat = dense_mat.reshape(dense_mat.size(0), n_view, n_view, dense_mat.size(-1), dense_mat.size(-1))
    dense_mat = dense_mat.permute(0, 1, 3, 2, 4).contiguous()
    dense_mat = dense_mat.reshape(-1, n_view * dense_mat.size(-1), n_view * dense_mat.size(-1))
    return dense_mat


def apply_laplacian(p_mat: torch.Tensor, n_view: int, w: torch.Tensor = None):
    """
    :param p_mat: (B, kN, kN)
    :param n_view: number of views
    :param w: (B, k, k)
    """
    n_point = p_mat.size(-1) // n_view
    d = torch.sum(w, dim=-1) if w is not None else None

    P_new = torch.zeros_like(p_mat)
    for view_i, view_j in itertools.product(range(n_view), range(n_view)):
        ia, ib = view_i * n_point, (view_i + 1) * n_point
        ja, jb = view_j * n_point, (view_j + 1) * n_point
        sub_block = p_mat[:, ia:ib, ja:jb]  # (B, N, N)
        if view_i == view_j:
            if w is not None:
                P_new[:, ia:ib, ja:jb] = sub_block * (d[:, view_i] - w[:, view_i, view_j]).reshape(-1, 1, 1)
            else:
                P_new[:, ia:ib, ja:jb] = sub_block * (n_view - 1)
        else:
            if w is not None:
                P_new[:, ia:ib, ja:jb] = -sub_block * w[:, view_i, view_j].reshape(-1, 1, 1)
            else:
                P_new[:, ia:ib, ja:jb] = -sub_block

    return P_new


def symm_flow_to_perm(pc1: torch.Tensor, flow: torch.Tensor, pc2: torch.Tensor, t=0.01):
    dist12 = -torch.cdist(pc1 + flow[:, 0], pc2)
    dist21 = -torch.cdist(pc1, pc2 + flow[:, 1])
    dist = torch.nn.functional.softmax((dist12 + dist21) / t, -1)
    return dist


def perm_to_symm_flow(pc1: torch.Tensor, pc2: torch.Tensor, log_perm: torch.Tensor):
    dense_flow = pc2.unsqueeze(1) - pc1.unsqueeze(2)
    flow01 = (dense_flow * torch.softmax(log_perm, dim=2).unsqueeze(-1)).sum(2)
    flow10 = -(dense_flow * torch.softmax(log_perm, dim=1).unsqueeze(-1)).sum(1)
    return torch.stack([flow01, flow10], dim=1)


def feature_propagation(unknown: torch.Tensor, known: torch.Tensor, known_feats: torch.Tensor,
                        transposed: bool = True):
    """
    :param unknown: (B, N, 3)
    :param known:   (B, n, 3)
    :param known_feats: (B, F, n)
    :param transposed: if false, then will take in known_feats as (B, n, F) and return (B, N, F)
    :return: (B, F, N)
    """
    dist, idx = pointnet2_util.three_nn(unknown.contiguous(), known.contiguous())
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    if not transposed:
        known_feats = known_feats.transpose(-1, -2)
    interpolated_feats = pointnet2_util.three_interpolate(
        known_feats.contiguous(), idx.contiguous(), weight.contiguous()
    )
    if not transposed:
        interpolated_feats = interpolated_feats.transpose(-1, -2)
    return interpolated_feats


class FullNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.nsample_motion = 256
        self.train_s = 6
        self.t = 0.01

        self.flow_net = FlowNet()
        self.conf_net = ConfNet()
        self.mot_net = MotNet()
        self.load_subnet_weights()

    def load_subnet_weights(self):
        with Path(self.args.flow_model).open() as f:
            flow_config = yaml.load(f, Loader=yaml.FullLoader)
        self.flow_net.load_state_dict(torch.load(flow_config["save_path"] + "/best.pth.tar")['model_state'])

        with Path(self.args.conf_model).open() as f:
            conf_config = yaml.load(f, Loader=yaml.FullLoader)
        self.conf_net.load_state_dict(torch.load(conf_config["save_path"] + "/best.pth.tar")['model_state'])

        with Path(self.args.mot_model).open() as f:
            mot_config = yaml.load(f, Loader=yaml.FullLoader)
        self.mot_net.load_state_dict(torch.load(mot_config["save_path"] + "/best.pth.tar")['model_state'])

    def forward(self, xyz):
        """
        :param xyz: (B, K, N, 3)
        """
        n_batch, n_view, n_point, _ = xyz.size()

        # Sub-Sample for motion synchronization.
        xyz_gathered = xyz.reshape(n_batch * n_view, n_point, 3).contiguous()
        sub_inds = pointnet2_util.furthest_point_sample(xyz_gathered, self.nsample_motion).long()
        xyz_down = pointnet2_util.gather_nd(xyz_gathered, sub_inds)
        sub_inds = break_leading_dim([n_batch, n_view], sub_inds)

        # Infer Flow & Pair-wise flow weight
        perm_dict = {}
        weight_bin_dict = {}
        raw_flow_dict = {}

        for view_i in range(n_view):
            for view_j in range(view_i + 1, n_view):
                pc_i = xyz[:, view_i].contiguous()
                pc_j = xyz[:, view_j].contiguous()

                # PD-Flow
                flow_ij, _, _, _, _ = self.flow_net.forward(pc_i, pc_j, pc_i, pc_j)
                flow_ji, _, _, _, _ = self.flow_net.forward(pc_j, pc_i, pc_j, pc_i)

                flow_ij = flow_ij[0].transpose(-1, -2)
                raw_flow_dict[(view_i, view_j)] = flow_ij
                flow_ji = flow_ji[0].transpose(-1, -2)
                raw_flow_dict[(view_j, view_i)] = flow_ji

                flow_ij = torch.stack([flow_ij, flow_ji], dim=1)

                _, weight_ij = self.conf_net(get_network_input(xyz[:, view_i], xyz[:, view_j], flow_ij[:, 0]))
                _, weight_ji = self.conf_net(get_network_input(xyz[:, view_j], xyz[:, view_i], flow_ij[:, 1]))
                weight_ij = torch.stack([weight_ij, weight_ji], dim=1).sigmoid()
                weight_bin_score = torch.mean(weight_ij, -1, keepdim=True)
                weight_bin_score = torch.mean(weight_bin_score, -2, keepdim=True)

                perm_ij = symm_flow_to_perm(xyz[:, view_i], flow_ij, xyz[:, view_j])
                perm_dict[(view_i, view_j)] = perm_ij
                weight_bin_dict[(view_i, view_j)] = weight_bin_score

        # Synchronize permutation and get refined flow
        perm_dense = compose_dense(perm_dict, n_view, torch.eye(n_point).cuda().unsqueeze(0).repeat(n_batch, 1, 1))
        weight_dense = compose_dense(weight_bin_dict, n_view, torch.zeros_like(weight_bin_dict[(0, 1)]))
        weight_dense = weight_dense + weight_dense.transpose(-1, -2)

        perm_dense = apply_laplacian(perm_dense, n_view, weight_dense)
        perm_dense = sync_perm(perm_dense, n_point, 1.0e-4)

        # Infer Segmentation.
        flow_dict = {}
        motion_dict = {}

        for view_i in range(n_view):
            for view_j in range(view_i + 1, n_view):
                perm_ij = perm_dense[:, view_i * n_point: (view_i + 1) * n_point,
                                     view_j * n_point: (view_j + 1) * n_point]  # (B, N, N)
                flow_ij = perm_to_symm_flow(xyz[:, view_i], xyz[:, view_j], perm_ij / self.t / self.t)
                flow_dict[(view_i, view_j)] = flow_ij

                _, motion_ij, _, _ = self.mot_net(xyz[:, [view_i, view_j]], flow_ij, sub_inds[:, [view_i, view_j]])
                motion_ij = motion_ij.sigmoid()
                motion_ij_values = motion_ij.view(motion_ij.size(0), -1)

                motion_ij_scale = torch.mean(motion_ij_values, dim=-1, keepdim=True).unsqueeze(-1)
                motion_ij = motion_ij / motion_ij_scale
                motion_dict[(view_i, view_j)] = motion_ij

        motion_dense = compose_dense(motion_dict, n_view, torch.zeros_like(motion_dict[(0, 1)]))
        motion_absolute = sync_motion_seg(motion_dense, force_d=self.train_s, t=0.0)
        motion_absolute = torch.softmax(motion_absolute / self.t, dim=-1)
        motion_absolute = feature_propagation(xyz_gathered, xyz_down,
            motion_absolute.reshape(n_batch * n_view, self.nsample_motion, self.train_s).transpose(-1, -2)).reshape(
            n_batch, n_view, self.train_s, -1).permute(0, 2, 1, 3).contiguous()
        motion_absolute = knead_leading_dims(2, motion_absolute)

        # Get pairwise transformations.
        R_list = []
        t_list = []
        for view_i in range(n_view):
            R_sub_list = []
            t_sub_list = []
            for view_j in range(n_view):
                if view_i < view_j:
                    flow_ij = flow_dict[(view_i, view_j)][:, 0]
                elif view_i > view_j:
                    flow_ij = flow_dict[(view_j, view_i)][:, 1]
                else:
                    R_sub_list.append(torch.eye(3).cuda().unsqueeze(0).repeat(n_batch * self.train_s, 1, 1))
                    t_sub_list.append(torch.zeros(n_batch * self.train_s, 3).cuda())
                    continue
                flow_ij = flow_ij.unsqueeze(1).repeat(1, self.train_s, 1, 1)
                flow_ij = knead_leading_dims(2, flow_ij)
                xyz_i = xyz[:, view_i].unsqueeze(1).repeat(1, self.train_s, 1, 1)
                xyz_i = knead_leading_dims(2, xyz_i)
                R_ij, t_ij = fit_motion_svd_batch(xyz_i, xyz_i + flow_ij)
                t_ij.clamp_(-2.0, 2.0)
                R_sub_list.append(R_ij)
                t_sub_list.append(t_ij)
            R_list.append(R_sub_list)
            t_list.append(t_sub_list)

        # Synchronize the transformations and get rigid flow.
        motion_absolute = break_leading_dim([n_batch, self.train_s], motion_absolute)
        R_sync, t_sync = motion_synchronization_spectral(R_list, t_list)
        R_sync = R_sync.reshape(n_batch, self.train_s, n_view, 3, 3)
        t_sync = t_sync.reshape(n_batch, self.train_s, n_view, 3)
        R_sync_inv = R_sync.transpose(-1, -2)
        t_sync_inv = -torch.einsum('bskij,bskj->bski', R_sync_inv, t_sync)

        rigid_flows = {}
        for view_i in range(n_view):
            for view_j in range(n_view):
                if view_i == view_j:
                    continue
                R_ji = torch.einsum('bsij,bsjm->bsim', R_sync[:, :, view_j], R_sync_inv[:, :, view_i])
                t_ji = torch.einsum('bsij,bsj->bsi', R_sync[:, :, view_j], t_sync_inv[:, :, view_i]) + \
                       t_sync[:, :, view_j]
                xyz_j = torch.einsum('bsij,bnj->bsni', R_ji, xyz[:, view_i]) + t_ji.unsqueeze(-2)
                flow_ij = torch.einsum('bsni,bsn->bni', xyz_j, motion_absolute[:, :, view_i]) - xyz[:, view_i]
                rigid_flows[(view_i, view_j)] = flow_ij

        motion_absolute = motion_absolute.permute(0, 2, 3, 1).contiguous()
        return motion_absolute, raw_flow_dict, rigid_flows
