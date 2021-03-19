"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Author: Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""


import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from utils.motion_util import Isometry


class DatasetSpec:
    PC = 1
    SEGM = 2
    FLOW = 3
    FULL_FLOW = 4


def compute_flow(base_pc: np.ndarray, base_segms: np.ndarray, base_cam: Isometry, base_motions: list,
                 dest_cam: Isometry, dest_motions: list):
    n_parts = len(base_motions)
    final_pc = np.empty_like(base_pc)
    for part_id in range(n_parts):
        part_mask = np.where(base_segms == (part_id + 1))[0]
        part_pc = (dest_cam.inv().dot(dest_motions[part_id]).dot(
            base_motions[part_id].inv()).dot(base_cam)) @ base_pc[part_mask]
        final_pc[part_mask] = part_pc
    return final_pc - base_pc


class MultibodyDataset(Dataset):
    def __init__(self, base_folder, spec, split='train', view_sel=None):
        """
        :param base_folder: a data root folder containing `data' folder and `meta.json'.
        :param split: split name to be loaded.
        """
        self.base_folder = Path(base_folder)
        with (self.base_folder / "meta.json").open() as f:
            self.meta = json.load(f)
        self.split = split
        self.data_ids = self.meta[split]
        self.spec = spec
        if view_sel is None:
            view_sel = [None]
        self.view_sel = view_sel

    def __len__(self):
        return len(self.data_ids) * len(self.view_sel)

    def __getitem__(self, data_id):
        idx, view_sel_idx = data_id // len(self.view_sel), data_id % len(self.view_sel)
        pcs, segms, trans_dict = self._get_item(idx)
        n_parts = len(trans_dict) - 1
        n_views = pcs.shape[0]

        view_sel = self.view_sel[view_sel_idx]
        if view_sel is None:
            view_sel = list(range(n_views))

        def get_view_motions(view_id):
            return [Isometry.from_matrix(trans_dict[t][view_id]) for t in range(1, n_parts + 1)]

        ret_vals = {}

        if DatasetSpec.PC in self.spec:
            ret_vals[DatasetSpec.PC] = pcs[view_sel, ...]

        if DatasetSpec.SEGM in self.spec:
            ret_vals[DatasetSpec.SEGM] = segms[view_sel, ...]

        if DatasetSpec.FLOW in self.spec:
            assert len(view_sel) == 2
            view0_id, view1_id = view_sel
            flow12 = compute_flow(pcs[view0_id], segms[view0_id],
                                  Isometry.from_matrix(trans_dict['cam'][view0_id]), get_view_motions(view0_id),
                                  Isometry.from_matrix(trans_dict['cam'][view1_id]), get_view_motions(view1_id))
            ret_vals[DatasetSpec.FLOW] = flow12

        if DatasetSpec.FULL_FLOW in self.spec:
            all_flows = []
            for view_i in view_sel:
                for view_j in view_sel:
                    flow_ij = compute_flow(pcs[view_i], segms[view_i],
                                           Isometry.from_matrix(trans_dict['cam'][view_i]), get_view_motions(view_i),
                                           Isometry.from_matrix(trans_dict['cam'][view_j]), get_view_motions(view_j))
                    all_flows.append(flow_ij)
            all_flows = np.stack(all_flows)
            ret_vals[DatasetSpec.FULL_FLOW] = all_flows

        return [ret_vals[k] for k in self.spec]

    def _get_item(self, idx):
        data_path = self.base_folder / "data" / ("%06d.npz" % self.data_ids[idx])
        datum = np.load(data_path, allow_pickle=True)

        raw_pc = datum['pc'].astype(np.float32)
        raw_segm = datum['segm']
        raw_trans = datum['trans'].item()

        return raw_pc, raw_segm, raw_trans
