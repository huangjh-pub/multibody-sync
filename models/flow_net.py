"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Author: Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""


import torch.nn as nn
import torch
from utils.pointconv_util import PointConvD, PointWarping, UpsampleFlow, PointConvFlow
from utils.pointconv_util import SceneFlowEstimatorPointConv
from utils.nn_util import Seq

BN_CONFIG = {"class": "GroupNorm", "num_groups": 4}


class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()

        flow_nei = 32
        feat_nei = 16

        n_feat_ijs = [128, 256, 512]
        n_feat_layers = [64, 128, 256, 512]
        n_feat_updeconv = [64, 64, 128]
        n_cv_feats = [64, 128, 256]

        scene_flow_channels = [256, 256]        # Inter-points
        scene_flow_mlp = [256, 128]              # Self-points

        self.n_points = [512, 128, 32, 8]

        self.init_fc = Seq(3).conv1d(n_feat_layers[0], leaky=True).conv1d(n_feat_layers[0], leaky=True)

        # warping
        self.warping = PointWarping()

        # upsample
        self.upsample = UpsampleFlow()

        # Layers
        self.feat_ijs = nn.ModuleList([
            Seq(n_feat_layers[0]).conv1d(n_feat_ijs[0], leaky=True),
            Seq(n_feat_layers[1]).conv1d(n_feat_layers[1], leaky=True).conv1d(n_feat_ijs[1], leaky=True),
            Seq(n_feat_layers[2]).conv1d(n_feat_layers[2], leaky=True).conv1d(n_feat_ijs[2], leaky=True)
        ])

        self.subsample_layers = nn.ModuleList([
            PointConvD(self.n_points[1], feat_nei, n_feat_ijs[0] + 3, n_feat_layers[1]),
            PointConvD(self.n_points[2], feat_nei, n_feat_ijs[1] + 3, n_feat_layers[2]),
            PointConvD(self.n_points[3], feat_nei, n_feat_ijs[2] + 3, n_feat_layers[3]),
        ])

        self.upsample_deconv = nn.ModuleList([
            Seq(n_feat_layers[1]).conv1d(n_feat_updeconv[0], leaky=True),
            Seq(n_feat_layers[2]).conv1d(n_feat_updeconv[1], leaky=True),
            Seq(n_feat_layers[3]).conv1d(n_feat_updeconv[2], leaky=True)
        ])

        self.cv_layers = nn.ModuleList([
            PointConvFlow(flow_nei, n_feat_layers[0] * 2 + n_feat_updeconv[0] * 2 + 3, [n_cv_feats[0], n_cv_feats[0]]),
            PointConvFlow(flow_nei, n_feat_layers[1] * 2 + n_feat_updeconv[1] * 2 + 3, [n_cv_feats[1], n_cv_feats[1]]),
            PointConvFlow(flow_nei, n_feat_layers[2] * 2 + n_feat_updeconv[2] * 2 + 3, [n_cv_feats[2], n_cv_feats[2]]),
        ])

        self.flow_layers = nn.ModuleList([
            SceneFlowEstimatorPointConv(n_feat_layers[0] + scene_flow_mlp[-1], n_cv_feats[0], bn=BN_CONFIG,
                                        channels=scene_flow_channels, mlp=scene_flow_mlp),
            SceneFlowEstimatorPointConv(n_feat_layers[1] + scene_flow_mlp[-1], n_cv_feats[1], bn=BN_CONFIG,
                                        channels=scene_flow_channels, mlp=scene_flow_mlp),
            SceneFlowEstimatorPointConv(n_feat_layers[2], n_cv_feats[2], flow_ch=0, bn=BN_CONFIG,
                                        channels=scene_flow_channels, mlp=scene_flow_mlp),
        ])

    def forward_feature(self, xyz, color):
        """
        :param xyz: (B, N, 3)
        :param color: point feature (B, N, 3)
        :return: concat_features, layer_features, point_pyramids, point_fps
        """
        pc_l0 = xyz
        color = color.transpose(-1, -2)
        pc_l = [pc_l0.transpose(-1, -2)]
        feat_l = [self.init_fc(color)]
        fps_l = [None]

        # Go deep and extract features.
        for layer_id in range(len(self.n_points) - 1):
            feat_ij = self.feat_ijs[layer_id](feat_l[-1])
            pc_new, feat_new, fps_new = self.subsample_layers[layer_id](pc_l[-1], feat_ij)
            pc_l.append(pc_new)
            feat_l.append(feat_new)
            fps_l.append(torch.gather(fps_l[-1], dim=-1, index=fps_new.long()) if fps_l[-1] is not None else fps_new)

        # Concat layer L's feature with layer (L+1)'s to increase field of view.
        c_feat_l = []
        for layer_id in range(len(self.n_points) - 2, -1, -1):
            feat_ji = self.upsample(pc_l[layer_id], pc_l[layer_id + 1], feat_l[layer_id + 1])
            feat_ji = self.upsample_deconv[layer_id](feat_ji)
            c_feat = torch.cat([feat_l[layer_id], feat_ji], dim=1)
            c_feat_l.insert(0, c_feat)

        # We do not need information about the last layer, drop them.
        pc_l, feat_l, fps_l = pc_l[:-1], feat_l[:-1], fps_l[:-1]

        return c_feat_l, feat_l, pc_l, fps_l

    def forward_flow(self, layer_id: int, concat_feat1, concat_feat2,
                     layer_feat1, hier_pc1, hier_pc2, info_dict: dict):
        """
        :param layer_id: Layer ID
        :param concat_feat1: list of layers of concat_feat
        :param concat_feat2:
        :param layer_feat1:  list of layers of layer_feat
        :param hier_pc1:     LoD pyramid of point clouds
        :param hier_pc2:
        :param info_dict:    last iteration's information
        :return: flow (B, 3, n), new_info_dict
        """
        # Given PC1, warped-PC2, the assembled features of the two clouds.
        # Build a correlation and assembles into feature vector for each point in PC1.
        cost_volume = self.cv_layers[layer_id](hier_pc1[layer_id],
                                               info_dict["pc_warped"],
                                               concat_feat1[layer_id],
                                               concat_feat2[layer_id])
        # This part is just convolution of the arguments.
        feat, flow = self.flow_layers[layer_id](hier_pc1[layer_id],
                                                info_dict["new_feat"], cost_volume, info_dict["up_flow"])
        if layer_id > 0:
            up_flow = self.upsample(hier_pc1[layer_id - 1], hier_pc1[layer_id], flow)
            pc_warped = self.warping(hier_pc1[layer_id - 1], hier_pc2[layer_id - 1], up_flow)
            feat_up = self.upsample(hier_pc1[layer_id - 1], hier_pc1[layer_id], feat)
            new_feat = torch.cat([layer_feat1[layer_id - 1], feat_up], dim=1)
        else:
            pc_warped, new_feat, up_flow = None, None, None

        return flow, {
            "pc_warped": pc_warped,
            "new_feat": new_feat,
            "up_flow": up_flow
        }

    def forward(self, xyz1, xyz2, color1, color2):
        """
        :param xyz1: (B, N, 3)
        :param xyz2: (B, N, 3)
        :param color1: point feature (B, N, 3)
        :param color2: point feature (B, N, 3)
        :return:
          - all_flows: list of (B, 3, N)
          - point_fps0: list of (B, N)
          - point_fps1: list of (B, N)
          - point_pyramids0: list of (B, 3, N)
          - point_pyramids1: list of (B, 3, N)
          - coarse_perm: (B, n1_coarse, n2_coarse)
        """

        concat_features = []
        layer_features = []
        point_pyramids = []
        point_fps = []

        # Building Feature Pyramid.
        for (xyz, color) in [(xyz1, color1), (xyz2, color2)]:
            c_feat_l, feat_l, pc_l, fps_l = self.forward_feature(xyz, color)
            concat_features.append(c_feat_l)
            layer_features.append(feat_l)
            point_pyramids.append(pc_l)
            point_fps.append(fps_l)

        # Build Cost Volume across two clouds and infer flow of different scales.
        iter_info_dict = {
            "pc_warped": point_pyramids[1][-1],
            "new_feat": layer_features[0][-1],
            "up_flow": None
        }
        all_flows = []
        for layer_id in range(len(self.n_points) - 2, -1, -1):
            flow, iter_info_dict = self.forward_flow(layer_id, concat_features[0], concat_features[1],
                                                     layer_features[0], point_pyramids[0], point_pyramids[1],
                                                     iter_info_dict)
            all_flows.insert(0, flow)

        return all_flows, point_fps[0], point_fps[1], point_pyramids[0], point_pyramids[1]
