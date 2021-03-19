"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Author: Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""


import yaml
import argparse
from pathlib import Path
from torch import optim
from dataset import MultibodyDataset, DatasetSpec as ds
from torch.optim.lr_scheduler import LambdaLR
from utils.pytorch_util import BNMomentumScheduler, Trainer
from torch.utils.data import DataLoader
from loss import *


def lr_curve(it):
    return max(
        args.lr_decay ** (int(it * args.batch_size / args.decay_step)),
        args.lr_clip / args.lr,
    )


def bn_curve(it):
    if args.decay_step == -1:
        return args.bn_momentum
    else:
        return max(
            args.bn_momentum
            * args.bn_decay ** (int(it * args.batch_size / args.decay_step)),
            1e-2,
        )


def get_flow_train_specs():
    criterion = MultiScaleFlowLoss(alpha=[0.04, 0.08, 0.16])

    def forward_fn(network, data, is_eval=False):
        with torch.set_grad_enabled(not is_eval):
            inputs, flow = data
            inputs = inputs.cuda()
            flow = flow.cuda()

            pos1, pos2 = inputs[:, 0, ...].contiguous(), inputs[:, 1, ...].contiguous()
            pred_flows, fps_pc1_idxs, fps_pc2_idxs, draw_pc1, draw_pc2 = network(pos1, pos2, pos1, pos2)
            loss = criterion(pred_flows, flow, fps_pc1_idxs)
            loss_dict = {'loss': loss.item()}

        return loss, loss_dict

    return forward_fn, [ds.PC, ds.FLOW], \
        [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3], [3, 0], [3, 1], [3, 2]]


def get_mot_train_specs():
    from utils.pointnet2_util import furthest_point_sample
    criterion = CombinedLoss(names=['trans', 'group'], loss=[MotTransLoss(), MotGroupLoss()], weights=[1.0, 1.0])

    def forward_fn(network, data, is_eval=False):
        with torch.set_grad_enabled(not is_eval):
            inputs, flow, segm = data
            inputs = inputs.cuda()
            segm = segm.cuda()

            dense_flow = inputs[:, 1].unsqueeze(1) - inputs[:, 0].unsqueeze(2)
            dist_mat = -torch.cdist(inputs[:, 0] + flow.cuda(), inputs[:, 1]) / 0.01
            flow01 = (dense_flow * torch.softmax(dist_mat, dim=-1).unsqueeze(-1)).sum(2)
            flow10 = -(dense_flow * torch.softmax(dist_mat, dim=-2).unsqueeze(-1)).sum(1)
            flow = torch.stack([flow01, flow10], dim=1)

            sub_ind0 = furthest_point_sample(inputs[:, 0, ...].contiguous(), 256).long()
            sub_ind1 = furthest_point_sample(inputs[:, 1, ...].contiguous(), 256).long()
            sub_inds = torch.stack([sub_ind0, sub_ind1], dim=1)

            pd_trans, pd_group_matrix, res0, res1 = network(inputs, flow, sub_inds)
            loss = criterion(segm=segm,
                             pred_support=pd_group_matrix,
                             group_sub_idx=sub_inds,
                             trans_res0=res0,
                             trans_res1=res1)

            loss_dict = {}
            loss_dict.update(loss)
            loss = loss['sum']
            del loss_dict['sum']
            loss_dict["loss"] = loss.item()

        return loss, loss_dict

    return forward_fn, [ds.PC, ds.FLOW, ds.SEGM], [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]


def get_conf_train_specs():
    from models.conf_net import get_network_input
    from models.flow_net import FlowNet

    with Path(args.flow_model).open() as yf:
        flow_config = yaml.load(yf, Loader=yaml.FullLoader)

    flow_net_weight_path = flow_config["save_path"] + "/best.pth.tar"
    flow_network = FlowNet()
    flow_network.load_state_dict(torch.load(flow_net_weight_path)['model_state'])
    flow_network.cuda().eval()

    criterion = FlowConfLoss()

    def forward_fn(network, data, is_eval=False):
        with torch.set_grad_enabled(not is_eval):
            inputs, flow = data
            inputs = inputs.cuda()
            flow = flow.cuda()

            pc_i, pc_j = inputs[:, 0].contiguous(), inputs[:, 1].contiguous()
            with torch.no_grad():
                flow_ij, _, _, _, _ = flow_network.forward(pc_i, pc_j, pc_i, pc_j)
            flow_ij = flow_ij[0].transpose(-1, -2)  # (B, N, 3)
            is_pos = torch.sum((flow_ij - flow) ** 2, dim=-1) < (0.1 ** 2)
            matches = get_network_input(pc_i, pc_j, flow_ij)

            logits_init, logits_iter = network(matches)
            loss = criterion(logits=[logits_init, logits_iter], is_pos=is_pos)

            loss_dict = {}
            loss_dict.update(loss)
            loss = loss['loss']
            loss_dict["loss"] = loss.item()

        return loss, loss_dict

    return forward_fn, [ds.PC, ds.FLOW], \
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 0], [2, 1], [3, 1], [3, 2]]


def get_full_train_specs():
    criterion = CombinedLoss(names=['flow', 'rflow', 'iou'],
                             loss=[MultiwayFlowLoss(4), MultiwayRFlowLoss(4), IoULoss()],
                             weights=[0.05, 0.01, 1.0])

    def forward_fn(network, data, is_eval=False):
        with torch.set_grad_enabled(not is_eval):
            inputs, segm, full_flow = data
            inputs = inputs.cuda()

            pd_segm, pd_raw_flow, pd_rigid_flow = network(inputs)

            loss = criterion(segm=segm.cuda(),
                             pd_segm=pd_segm,
                             gt_full_flow=full_flow.cuda(),
                             pd_flow_dict=pd_raw_flow,
                             pd_rflow_dict=pd_rigid_flow)

            loss_dict = {}
            loss_dict.update(loss)
            loss = loss['sum']
            del loss_dict['sum']
            loss_dict["loss"] = loss.item()

        return loss, loss_dict

    return forward_fn, [ds.PC, ds.SEGM, ds.FULL_FLOW], None


def train():
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_curve)
    bnm_scheduler = BNMomentumScheduler(model, bn_lambda=bn_curve)

    train_set = MultibodyDataset(args.train_base_folder, model_data_spec, 'train', model_view_sel)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    val_set = MultibodyDataset(args.val_base_folder, model_data_spec, 'val', model_view_sel)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    trainer = Trainer(
        model, model_fn, optimizer,
        exp_base=args.save_path,
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler
    )
    trainer.train(args.epochs, train_loader, val_loader)


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')

    # Read parameters
    args = parser.parse_args()
    with Path(args.config).open() as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Dispatch training.
    if args.type == "flow":
        from models.flow_net import FlowNet

        model = FlowNet().cuda()
        model_fn, model_data_spec, model_view_sel = get_flow_train_specs()
    elif args.type == "mot":
        from models.mot_net import MotNet

        model = MotNet().cuda()
        model_fn, model_data_spec, model_view_sel = get_mot_train_specs()

    elif args.type == "conf":
        from models.conf_net import ConfNet
        model = ConfNet().cuda()
        model_fn, model_data_spec, model_view_sel = get_conf_train_specs()

    elif args.type == "full":
        from models.full_net import FullNet
        model = FullNet(args).cuda()
        model_fn, model_data_spec, model_view_sel = get_full_train_specs()

    else:
        raise NotImplementedError

    # Perform training
    train()
