"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Author: Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""


import yaml
import tqdm
import argparse
import open3d as o3d
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from utils.nn_util import break_leading_dim, knead_leading_dims
from torch.utils.data import DataLoader
from utils import pointnet2_util
from models.flow_net import FlowNet
from models.conf_net import ConfNet, get_network_input
from models.mot_net import MotNet
from models.full_net import compose_dense, apply_laplacian, perm_to_symm_flow, feature_propagation
from utils.sync_util import sync_perm, sync_motion_seg, motion_synchronization_spectral, fit_motion_svd_batch

from dataset import MultibodyDataset, DatasetSpec as ds


# Borrowed from PointGroup
COLOR20 = np.array(
    [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
     [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
     [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
     [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])


def build_pointcloud(pc, cid: np.ndarray = None):
    assert pc.shape[1] == 3 and len(pc.shape) == 2, f"Point cloud is of size {pc.shape} and cannot be displayed!"
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    if cid is not None:
        assert cid.shape[0] == pc.shape[0], f"Point and color id must have same size {cid.shape[0]}, {pc.shape[0]}"
        assert cid.ndim == 1, f"color id must be of size (N,) currently ndim = {cid.ndim}"
        point_cloud.colors = o3d.utility.Vector3dVector(COLOR20[cid % COLOR20.shape[0]] / 255.)

    return point_cloud


def binarize_motion(mat: torch.Tensor):
    n_batch, _, K, N = mat.size()
    while True:
        mat_bin = torch.zeros_like(mat)
        amax_ind = mat.argmax(dim=1, keepdim=True)
        mat_bin.scatter_(dim=1, index=amax_ind, value=1.)
        point_count = torch.sum(mat_bin, dim=-1)
        valid_Bs = torch.all(knead_leading_dims(2, point_count) > 2, dim=-1)
        if torch.all(valid_Bs):
            break
        mat = knead_leading_dims(2, mat)[valid_Bs]
        mat = mat.view(n_batch, -1, K, N)
    return mat_bin


def remove_motion_outliers(motion_absolute, xyz):
    """
    :param motion_absolute: (B, [s, K, N)
    :param xyz: (B, [K, N)
    """
    NB_CNT = 10
    NB_THRES = 4

    from sklearn.neighbors import NearestNeighbors
    import scipy.stats

    motion_absolute, xyz = motion_absolute[0], xyz[0]
    new_motion = torch.zeros_like(motion_absolute)
    segm = torch.argmax(motion_absolute, dim=0)  # (K, N)
    n_view = xyz.size(0)
    n_point = xyz.size(1)

    for view_i in range(n_view):
        xyz_i = xyz[view_i].cpu().numpy()
        segm_i = segm[view_i].cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=NB_CNT).fit(xyz_i)
        nb_inds = nbrs.kneighbors(xyz_i, return_distance=False)  # (N, 10)
        nb_segm = segm_i[nb_inds]  # (N, 10)
        gcount = np.sum(nb_segm == segm_i[:, np.newaxis], axis=-1)  # (N, )
        gsegm = scipy.stats.mode(nb_segm, axis=1).mode[:, 0]  # (N, )
        segm_i[gcount < NB_THRES] = gsegm[gcount < NB_THRES]
        new_motion[segm_i, view_i, np.arange(n_point)] = 1

    return new_motion


def perform_icp(xyz, v_base: int, R_init, t_init, segm):
    xyz, R_init, t_init, segm = xyz[0], R_init[0], t_init[0], segm[0]
    segm = segm.bool()
    n_view = xyz.size(0)
    n_point = xyz.size(1)
    n_body = R_init.size(0)
    all_Rs = []
    all_ts = []
    for body_i in range(n_body):
        for view_i in range(n_view):
            R_vi = R_init[body_i, view_i].cpu().numpy()
            t_vi = t_init[body_i, view_i].cpu().numpy()
            T_vi = np.identity(4)
            T_vi[:3, :3] = R_vi
            T_vi[:3, 3] = t_vi
            xyz_v = build_pointcloud(xyz[v_base, segm[body_i, v_base]].cpu().numpy())
            xyz_i = build_pointcloud(xyz[view_i, segm[body_i, view_i]].cpu().numpy())
            new_T_vi = o3d.pipelines.registration.registration_icp(
                xyz_i, xyz_v, 0.25, T_vi,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
            new_T_vi = new_T_vi.transformation
            all_Rs.append(new_T_vi[:3, :3])
            all_ts.append(new_T_vi[:3, 3])
    all_Rs = torch.from_numpy(np.stack(all_Rs, axis=0)).float().cuda().reshape(1, n_body, n_view, 3, 3)
    all_ts = torch.from_numpy(np.stack(all_ts, axis=0)).float().cuda().reshape(1, n_body, n_view, 3)
    return all_Rs, all_ts


def check_connectivity(mat: torch.Tensor):
    mat = mat + torch.eye(mat.size(1), device=mat.device, dtype=mat.dtype).unsqueeze(0)
    now_mat = mat
    for vi in range(1, mat.size(1)):
        now_mat = torch.bmm(now_mat, mat.transpose(-1, -2))
    return torch.all(torch.all(now_mat > 0, dim=-1), dim=-1)


def edge_prune(w: torch.Tensor, th_list: list):
    for weight_th in th_list:
        weight_mask = (w > weight_th).float()
        if torch.all(check_connectivity(weight_mask)):
            w = w * weight_mask  # (B, K, K)
            break
    return w


class TestTimeFullNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_iter = 4
        self.rigid_n_iter = 1
        self.t = 0.01
        self.nsample_motion = 256
        self.mcut_thres = args.alpha
        assert self.rigid_n_iter <= self.n_iter

        self.flow_net = FlowNet()
        self.conf_net = ConfNet()
        self.mot_net = MotNet()

    def symm_flow_to_perm(self, pc1: torch.Tensor, flow: torch.Tensor, pc2: torch.Tensor, weight: torch.Tensor):
        n_point = pc1.size(1)
        dist12 = -torch.cdist(pc1 + flow[:, 0], pc2)
        dist21 = -torch.cdist(pc1, pc2 + flow[:, 1])
        weight_mat = torch.stack([weight[:, 0].unsqueeze(-1).repeat(1, 1, n_point),
                                  weight[:, 1].unsqueeze(-2).repeat(1, n_point, 1)])
        weight_mat /= torch.sum(weight_mat, dim=0, keepdim=True)
        dist = torch.nn.functional.softmax((dist12 * weight_mat[0] + dist21 * weight_mat[1]) / self.t,
                                           -1)
        return dist

    def forward(self, xyz):
        """
        :param xyz: (1, K, N, 3)
        """
        n_batch, n_view, n_point, _ = xyz.size()
        assert n_batch == 1, "Test time algorithm only supports batch size 1"

        # Sub-Sample for motion synchronization.
        xyz_gathered = xyz.reshape(n_batch * n_view, n_point, 3).contiguous()
        sub_inds = pointnet2_util.furthest_point_sample(xyz_gathered, self.nsample_motion).long()
        xyz_down = pointnet2_util.gather_nd(xyz_gathered, sub_inds)
        sub_inds = break_leading_dim([n_batch, n_view], sub_inds)

        flow_init = None
        xyz_transformed = xyz
        motion_absolute = None

        for iter_i in range(self.n_iter):

            # Infer Flow & Pair-wise flow weight
            perm_dict = {}
            weight_bin_dict = {}

            for view_i in range(n_view):
                for view_j in range(view_i + 1, n_view):
                    pc_i = xyz_transformed[:, view_i]
                    pc_j = xyz_transformed[:, view_j]

                    # PD-Flow
                    flow_ij, _, _, _, _ = self.flow_net.forward(pc_i, pc_j, pc_i, pc_j)
                    flow_ji, _, _, _, _ = self.flow_net.forward(pc_j, pc_i, pc_j, pc_i)
                    flow_ij = flow_ij[0].transpose(-1, -2)
                    flow_ji = flow_ji[0].transpose(-1, -2)

                    if flow_init is not None:
                        flow_init_i_jjt = feature_propagation(pc_i + flow_ij, pc_j, flow_init[:, view_j], False)
                        flow_ij = flow_ij + flow_init[:, view_i] - flow_init_i_jjt
                        flow_init_j_iit = feature_propagation(pc_j + flow_ji, pc_i, flow_init[:, view_i], False)
                        flow_ji = flow_ji + flow_init[:, view_j] - flow_init_j_iit
                    flow_ij = torch.stack([flow_ij, flow_ji], dim=1)

                    _, weight_ij = self.conf_net(get_network_input(xyz[:, view_i], xyz[:, view_j], flow_ij[:, 0]))
                    _, weight_ji = self.conf_net(get_network_input(xyz[:, view_j], xyz[:, view_i], flow_ij[:, 1]))
                    weight_ij = torch.stack([weight_ij, weight_ji], dim=1)
                    weight_ij.sigmoid_()
                    weight_bin_score = torch.sum(weight_ij > 0.5, dim=-1).float() / n_point
                    weight_bin_score = torch.mean(weight_bin_score, dim=-1)
                    weight_bin_score = weight_bin_score.reshape(n_batch, 1, 1).float()

                    perm_ij = self.symm_flow_to_perm(xyz[:, view_i], flow_ij, xyz[:, view_j], weight_ij)
                    perm_dict[(view_i, view_j)] = perm_ij
                    weight_bin_dict[(view_i, view_j)] = weight_bin_score

            # Synchronize permutation and get refined flow
            perm_dense = compose_dense(perm_dict, n_view, torch.eye(n_point).cuda().unsqueeze(0).repeat(n_batch, 1, 1))
            weight_dense = compose_dense(weight_bin_dict, n_view, torch.zeros_like(weight_bin_dict[(0, 1)]))
            weight_dense = weight_dense + weight_dense.transpose(-1, -2)

            # Determine the weight based on connectivity
            weight_dense = edge_prune(weight_dense, [0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

            perm_dense = apply_laplacian(perm_dense, n_view, weight_dense)
            perm_dense = sync_perm(perm_dense, n_point, 1.0e-4)

            # Infer Segmentation.
            flow_dict = {}
            motion_dict = {}
            sync_weight_dict = {}
            motion_canonical_scale = None

            for view_i in range(n_view):
                for view_j in range(view_i + 1, n_view):
                    perm_ij = perm_dense[:, view_i * n_point: (view_i + 1) * n_point,
                              view_j * n_point: (view_j + 1) * n_point]  # (B, N, N)
                    flow_ij = perm_to_symm_flow(xyz[:, view_i], xyz[:, view_j],
                                                     perm_ij / (self.t ** 2))  # (B, 2, N, 3)
                    # Cache flow:
                    flow_dict[(view_i, view_j)] = flow_ij
                    _, motion_ij, _, _ = self.mot_net(xyz[:, [view_i, view_j]], flow_ij, sub_inds[:, [view_i, view_j]])
                    motion_ij = motion_ij.sigmoid()

                    motion_ij_values = motion_ij.view(motion_ij.size(0), -1)
                    motion_ij_scale = torch.mean(motion_ij_values, dim=-1, keepdim=True).unsqueeze(-1)
                    if view_i == 0 and view_j == 1:
                        motion_canonical_scale = motion_ij_scale
                    else:
                        motion_ij = motion_ij / motion_ij_scale * motion_canonical_scale
                    motion_ij = torch.clamp(motion_ij, 0.0, 1.0)

                    motion_dict[(view_i, view_j)] = motion_ij

                    # Re-evaluate the flow weights, to be used in transformation estimation.
                    _, weight_ij = self.conf_net(get_network_input(xyz[:, view_i], xyz[:, view_j], flow_ij[:, 0]))
                    _, weight_ji = self.conf_net(get_network_input(xyz[:, view_j], xyz[:, view_i], flow_ij[:, 1]))
                    weight_ij = torch.stack([weight_ij, weight_ji], dim=1)
                    sync_weight_dict[(view_i, view_j)] = weight_ij.sigmoid()

            # Pairwise motion synchronization
            motion_dense = compose_dense(motion_dict, n_view, torch.zeros_like(motion_dict[(0, 1)]))
            motion_absolute = sync_motion_seg(motion_dense, t=0.0, cut_thres=self.mcut_thres)
            sync_s = motion_absolute.size(-1)
            motion_absolute /= motion_absolute.sum(-1, keepdim=True)
            motion_absolute = feature_propagation(
                xyz_gathered, xyz_down,
                motion_absolute.reshape(n_batch * n_view, self.nsample_motion, sync_s).transpose(-1, -2)).reshape(
                n_batch, n_view, sync_s, -1).permute(0, 2, 1, 3)

            # Binarize segmentation at test time.
            motion_absolute = binarize_motion(motion_absolute)
            sync_s = motion_absolute.size(1)
            motion_absolute = remove_motion_outliers(motion_absolute, xyz)
            tmp_s = 1 if iter_i < self.rigid_n_iter else sync_s

            motion_absolute = motion_absolute.view(n_batch * sync_s, n_view, n_point)
            R_list = []
            t_list = []
            w_list = []

            for view_i in range(n_view):
                R_sub_list = []
                t_sub_list = []
                w_sub_list = []
                for view_j in range(n_view):
                    if view_i < view_j:
                        flow_ij = flow_dict[(view_i, view_j)][:, 0]
                        weight_ij = sync_weight_dict[(view_i, view_j)][:, 0]
                    elif view_i > view_j:
                        flow_ij = flow_dict[(view_j, view_i)][:, 1]
                        weight_ij = sync_weight_dict[(view_j, view_i)][:, 1]
                    else:
                        R_sub_list.append(torch.eye(3).cuda().unsqueeze(0).repeat(n_batch * tmp_s, 1, 1))
                        t_sub_list.append(torch.zeros(n_batch * tmp_s, 3).cuda())
                        w_sub_list.append(torch.zeros(n_batch * tmp_s, ).cuda())
                        continue
                    flow_ij = flow_ij.unsqueeze(1).repeat(1, tmp_s, 1, 1)
                    flow_ij = knead_leading_dims(2, flow_ij)
                    weight_ij = weight_ij.unsqueeze(1).repeat(1, tmp_s, 1)
                    weight_ij = knead_leading_dims(2, weight_ij)
                    xyz_i = xyz[:, view_i].unsqueeze(1).repeat(1, tmp_s, 1, 1)
                    xyz_i = knead_leading_dims(2, xyz_i)

                    if tmp_s == 1:
                        R_ij, t_ij = fit_motion_svd_batch(xyz_i,
                                                          xyz_i + flow_ij, weight_ij)
                        w_ij = torch.mean(weight_ij, -1)
                    else:
                        local_weight = motion_absolute[:, view_i] * weight_ij
                        R_ij, t_ij = fit_motion_svd_batch(xyz_i,
                                                          xyz_i + flow_ij,
                                                          local_weight)
                        t_ij.clamp_(-2.0, 2.0)
                        w_ij = torch.sum(local_weight, -1) / torch.sum(motion_absolute[:, view_i], -1)
                    R_sub_list.append(R_ij)
                    t_sub_list.append(t_ij)
                    w_sub_list.append(w_ij)
                R_list.append(R_sub_list)
                t_list.append(t_sub_list)
                w_list.append(w_sub_list)

            motion_absolute = break_leading_dim([n_batch, sync_s], motion_absolute)

            trans_global_weight = torch.stack([torch.stack(sl, -1) for sl in w_list], 1)
            trans_global_weight = (trans_global_weight + trans_global_weight.transpose(-1, -2)) / 2.
            trans_global_weight = edge_prune(trans_global_weight, [0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

            # This motion sync part seems not contribute much to the accuracy at test time.
            # So we disable it by default. You can always re-enable it.
            # R_sync, t_sync = motion_synchronization_spectral(R_list, t_list,
            #                                                  trans_global_weight,
            #                                                  fallback_on_error=True)
            R_sync = torch.stack(R_list[0], dim=1)
            t_sync = torch.stack(t_list[0], dim=1)

            R_sync = R_sync.reshape(n_batch, tmp_s, n_view, 3, 3)
            t_sync = t_sync.reshape(n_batch, tmp_s, n_view, 3)

            # Enforce Gauge-freedom to be aligned with the first-view. (s.t. all K == 0 will be identity)
            # So the transformed point cloud do not have weird shapes, which will otherwise make flow bad.
            v_base = n_view - 1
            R_base = R_sync[:, :, v_base].transpose(-1, -2)
            t_base = -torch.einsum('bsij,bsj->bsi', R_base, t_sync[:, :, v_base])
            R_sync_cal = torch.einsum('bskij,bsjm->bskim', R_sync, R_base)
            t_sync_cal = torch.einsum('bskij,bsj->bski', R_sync, t_base) + t_sync
            R_sync, t_sync = R_sync_cal, t_sync_cal

            R_sync = R_sync.transpose(-1, -2)
            t_sync = -torch.einsum('bskij,bskj->bski', R_sync, t_sync)

            # Ignore some outliers.
            t_sync[t_sync > 1.0] = 0.0

            if iter_i >= self.rigid_n_iter:
                R_sync, t_sync = perform_icp(xyz, v_base, R_sync, t_sync, motion_absolute)

            # Apply the motion to the points, so that for next iteration
            # Flow will be made easier.
            xyz_transformed = torch.einsum('bskij,bknj->bskni', R_sync, xyz) + \
                              t_sync.unsqueeze(-2)
            if tmp_s == 1:
                xyz_transformed = xyz_transformed.squeeze(1)
            else:
                xyz_transformed = torch.einsum('bskni,bskn->bkni', xyz_transformed, motion_absolute)

            xyz_transformed = xyz_transformed.contiguous()
            flow_init = xyz_transformed - xyz

        motion_absolute = motion_absolute.permute(0, 2, 3, 1)
        return motion_absolute, xyz_transformed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')

    # Read parameters
    args = parser.parse_args()
    with Path(args.config).open() as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    assert args.type == "full", "Your config file must be of type 'full'."

    # Modify the following to test on your own data.
    test_set = MultibodyDataset(args.test_base_folder, [ds.PC, ds.SEGM, ds.FULL_FLOW], 'test', None)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    weight_path = args.save_path + "/best.pth.tar"
    model = TestTimeFullNet(args)
    model.load_state_dict(torch.load(weight_path)['model_state'])
    model.cuda().eval()

    all_ious = []
    with tqdm.tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='test') as tbar:
        for i, data in tbar:
            with torch.no_grad():
                inputs, segm, full_flow = data
                inputs = inputs.cuda()
                segm = segm.cuda()
                pd_segm, _ = model(inputs)
                segmented_pcds = []

                for view_i in range(inputs.size(1)):
                    segmented_pcds.append(build_pointcloud(inputs[0, view_i].cpu().numpy(),
                                                           pd_segm.argmax(-1)[0, view_i].cpu().numpy()))
                    segmented_pcds[-1].translate([view_i, 0.0, 0.0])
                o3d.visualization.draw_geometries(segmented_pcds)

