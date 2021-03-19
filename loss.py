"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Author: Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""


import torch
from torch import nn
import numpy as np
from utils.pointnet2_util import gather_nd
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class MultiScaleFlowLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred_flows, gt_flow, fps_idxs):
        num_scale = len(pred_flows)
        offset = len(fps_idxs) - num_scale

        gt_flows = [gt_flow]
        for i in range(1, len(fps_idxs)):
            fps_idx = fps_idxs[i]
            sub_gt_flow = gather_nd(gt_flows[0], fps_idx)
            gt_flows.append(sub_gt_flow)

        total_loss = torch.zeros(1).cuda()
        for i in range(num_scale):
            diff_flow = pred_flows[i].permute(0, 2, 1) - gt_flows[i + offset]
            total_loss += self.alpha[i] * torch.norm(diff_flow, dim=2).sum(dim=1).mean()

        return total_loss


class MultiwayFlowLoss(nn.Module):
    def __init__(self, n_view):
        super().__init__()
        self.n_view = n_view

    def forward(self, gt_full_flow, pd_flow_dict, **kwargs):
        loss = []
        for view_i in range(self.n_view):
            for view_j in range(self.n_view):
                if view_i == view_j:
                    continue
                gt_flow = gt_full_flow[:, view_i * self.n_view + view_j]
                pd_flow = pd_flow_dict[(view_i, view_j)]
                loss.append(torch.norm(gt_flow - pd_flow, dim=2).sum(dim=1).mean())
        return sum(loss) / len(loss)


class MultiwayRFlowLoss(MultiwayFlowLoss):
    def __init__(self, n_view):
        super().__init__(n_view)

    def forward(self, gt_full_flow, pd_rflow_dict, **kwargs):
        return super().forward(gt_full_flow, pd_rflow_dict)


class MotTransLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, trans_res0: torch.Tensor, trans_res1: torch.Tensor,
                group_sub_idx: torch.Tensor, segm: torch.Tensor, **kwargs):
        trans_res0 = (trans_res0 ** 2).sum(1)
        trans_res1 = (trans_res1 ** 2).sum(1)

        segm_down = torch.gather(segm, dim=-1, index=group_sub_idx)
        motion_mask0 = ((segm_down[:, 0, :].unsqueeze(2) -
                         segm_down[:, 1, :].unsqueeze(1)) == 0).float()
        motion_mask1 = motion_mask0.transpose(1, 2)
        motion_mask0 /= (motion_mask0.sum(2, keepdim=True) + 1e-8)
        motion_mask1 /= (motion_mask1.sum(2, keepdim=True) + 1e-8)

        trans_res0 *= motion_mask0
        trans_res1 *= motion_mask1
        return trans_res0.sum() / motion_mask0.sum() + trans_res1.sum() / motion_mask1.sum()


class MotGroupLoss(nn.Module):
    """
    Supervise the support matrix used for motion segmentation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_support: torch.Tensor, group_sub_idx: torch.Tensor, segm: torch.Tensor,
                view_pair: tuple = None, **kwargs):
        if view_pair is None:
            view_pair = (0, 1)

        vi, vj = view_pair
        segm_down = torch.gather(segm, dim=-1, index=group_sub_idx)
        motion_mask = ((segm_down[:, vi, :].unsqueeze(2) -
                        segm_down[:, vj, :].unsqueeze(1)) == 0).float()

        group_loss = F.binary_cross_entropy_with_logits(pred_support, motion_mask, reduction='mean')
        return group_loss


class FlowConfLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: list, is_pos: torch.Tensor):
        """
        :param logits: ([B, N])
        :param is_pos: (B, N) 0-1 values.
        :return: scalar loss
        """
        is_pos = is_pos.type(logits[0].type())
        is_neg = 1. - is_pos
        c = 2 * is_pos - 1.         # positive is 1, negative is -1
        num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
        num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0

        all_loss = []
        for ind_logits in logits:
            classif_losses = -torch.log(torch.sigmoid(c * ind_logits) + np.finfo(float).eps.item())
            classif_loss_p = torch.sum(classif_losses * is_pos, dim=1)
            classif_loss_n = torch.sum(classif_losses * is_neg, dim=1)
            classif_loss = torch.mean(classif_loss_p * 0.5 / num_pos + classif_loss_n * 0.5 / num_neg)
            all_loss.append(classif_loss)

        ind_logits = ind_logits.detach()
        precision = torch.mean(
            torch.sum((ind_logits > 0).type(is_pos.type()) * is_pos, dim=1) /
            torch.sum((ind_logits > 0).type(is_pos.type()), dim=1)
        )
        recall = torch.mean(
            torch.sum((ind_logits > 0).type(is_pos.type()) * is_pos, dim=1) /
            torch.sum(is_pos, dim=1)
        )

        return {
            "loss": sum(all_loss),
            "precision": precision.item(), "recall": recall.item()
        }


class IoULoss(nn.Module):

    def __init__(self, use_softmax=False):
        super().__init__()
        self.use_softmax = use_softmax

    @staticmethod
    def batch_hungarian_matching(gt_segm: torch.Tensor, pd_segm: torch.Tensor, iou: bool = True):
        """
        Get the matching based on IoU score of the Confusion Matrix.
            - Restriction: s must be larger/equal to all gt.
        :param gt_segm (B, N), this N should be n_view * n_point, also segmentation should start from 0.
        :param pd_segm (B, N, s), where s should be in the form of scores.
        :param iou: whether the confusion is based on IoU or simple accuracy.
        :return: (B, s, 2), Only the first n_gt_segms are valid mapping from gt to pd.
                 (B, s) gt mask
        """
        assert gt_segm.min() == 0

        n_batch, n_data, s = pd_segm.size()

        n_gt_segms = torch.max(gt_segm, dim=1).values + 1
        gt_segm = torch.eye(s, dtype=pd_segm.dtype, device=pd_segm.device)[gt_segm]

        matching_score = torch.einsum('bng,bnp->bgp', gt_segm, pd_segm)
        if iou:
            union_score = torch.sum(gt_segm, dim=1).unsqueeze(-1) + \
                          torch.sum(pd_segm, dim=1, keepdim=True) - matching_score
            matching_score = matching_score / (union_score + 1e-8)

        matching_idx = torch.ones((n_batch, s, 2), dtype=torch.long)
        valid_idx = torch.zeros((n_batch, s)).float()
        for batch_id, n_gt_segm in enumerate(n_gt_segms):
            assert n_gt_segm <= s
            row_ind, col_ind = linear_sum_assignment(matching_score[batch_id, :n_gt_segm, :].cpu().numpy(),
                                                     maximize=True)
            assert row_ind.size == n_gt_segm
            matching_idx[batch_id, :n_gt_segm, 0] = torch.from_numpy(row_ind)
            matching_idx[batch_id, :n_gt_segm, 1] = torch.from_numpy(col_ind)
            valid_idx[batch_id, :n_gt_segm] = 1

        matching_idx = matching_idx.to(pd_segm.device)
        valid_idx = valid_idx.to(pd_segm.device)

        return matching_idx, gt_segm, valid_idx

    def forward(self, pd_segm: torch.Tensor, segm: torch.Tensor, **kwargs):
        """
        :param segm:    (B, ...), starting from 1.
        :param pd_segm: (B, ..., s)
        :return: (B,) meanIoU
        """
        n_batch = pd_segm.size(0)
        num_classes = pd_segm.size(-1)

        gt_segm = segm.reshape(n_batch, -1)
        pd_segm = pd_segm.reshape(n_batch, -1, num_classes)

        if self.use_softmax:
            pd_segm = torch.softmax(pd_segm, dim=-1)

        n_data = gt_segm.size(-1)

        matching_idx, gt_segm, valid_idx = \
            self.batch_hungarian_matching(gt_segm.detach() - 1, pd_segm.detach())
        gt_gathered = torch.gather(gt_segm, dim=-1,
                                   index=matching_idx[..., 0].unsqueeze(1).repeat(1, n_data, 1))
        pd_gathered = torch.gather(pd_segm, dim=-1,
                                   index=matching_idx[..., 1].unsqueeze(1).repeat(1, n_data, 1))

        matching_score = (pd_gathered * gt_gathered).sum(dim=1)
        union_score = pd_gathered.sum(dim=1) + gt_gathered.sum(dim=1) - matching_score
        iou = matching_score / (union_score + 1e-8)
        matching_mask = (valid_idx > 0.0).float()
        assert not matching_mask.requires_grad
        iou = (iou * matching_mask).sum(-1) / matching_mask.sum(-1)
        iou = torch.mean(iou)

        return -iou


class CombinedLoss(nn.Module):
    def __init__(self, names: list, loss: list, weights: list):
        super().__init__()
        self.names = names
        self.loss = loss
        self.weights = weights
        assert len(names) == len(loss) == len(weights)

    def forward(self, **kwargs):
        loss_dict = {}
        loss_arr = []
        for nm, ls, w in zip(self.names, self.loss, self.weights):
            loss_res = ls(**kwargs)
            this_loss = None
            if isinstance(loss_res, torch.Tensor):
                this_loss = loss_res * w
            elif isinstance(loss_res, dict):
                if "loss" in loss_res.keys():
                    this_loss = loss_res["loss"] * w
                    del loss_res["loss"]
                loss_dict.update(loss_res)
            else:
                raise NotImplementedError
            if this_loss is not None:
                loss_arr.append(this_loss)
                loss_dict[nm] = this_loss.detach().cpu().numpy()
        loss_dict['sum'] = sum(loss_arr)
        return loss_dict

