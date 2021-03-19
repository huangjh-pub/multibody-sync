"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Author: Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""


import torch
import numpy as np


def fit_motion_svd_batch(pts0: torch.Tensor, pts1: torch.Tensor, weight: torch.Tensor = None):
    assert pts0.size(1) == pts1.size(1)
    n_batch, n_point, _ = pts0.size()

    if weight is None:
        pts0_mean = torch.mean(pts0, dim=1, keepdim=True)   # (B, 1, 3)
        pts1_mean = torch.mean(pts1, dim=1, keepdim=True)   # (B, 1, 3)
    else:
        pts0_mean = torch.einsum('bnd,bn->bd', pts0, weight) / torch.sum(weight, dim=1, keepdim=True)   # (B, 3)
        pts0_mean.unsqueeze_(1)
        pts1_mean = torch.einsum('bnd,bn->bd', pts1, weight) / torch.sum(weight, dim=1, keepdim=True)
        pts1_mean.unsqueeze_(1)

    pts0_centered = pts0 - pts0_mean
    pts1_centered = pts1 - pts1_mean

    if weight is None:
        S = torch.bmm(pts0_centered.transpose(1, 2), pts1_centered)
    else:
        S = pts0_centered.transpose(1, 2).bmm(torch.diag_embed(weight).bmm(pts1_centered))

    # If weight is not well-defined, S will be ill-posed.
    # We just return an identity matrix.
    valid_batches = ~torch.isnan(S).any(dim=1).any(dim=1)
    R_base = torch.eye(3, device=pts0.device).unsqueeze(0).repeat(n_batch, 1, 1)
    t_base = torch.ones((n_batch, 3), device=pts0.device) * 10.0

    if valid_batches.any():
        S = S[valid_batches, ...]
        u, s, v = torch.svd(S, some=False, compute_uv=True)
        R = torch.bmm(v, u.transpose(1, 2))
        det = torch.det(R)

        diag = torch.ones_like(S[..., 0], requires_grad=False)
        diag[:, 2] = det
        R = v.bmm(torch.diag_embed(diag).bmm(u.transpose(1, 2)))

        pts0_mean, pts1_mean = pts0_mean[valid_batches], pts1_mean[valid_batches]
        t = pts1_mean.squeeze(1) - torch.bmm(R, pts0_mean.transpose(1, 2)).squeeze(2)

        R_base[valid_batches] = R
        t_base[valid_batches] = t

    return R_base, t_base


def motion_synchronization_spectral(R, t, c: torch.Tensor = None, fallback_on_error: bool = False):
    n_view = len(R)
    n_batch = R[0][0].size(0)
    assert len(R) == n_view and len(R[0]) == n_view
    assert len(t) == n_view and len(t[0]) == n_view

    if c is None:
        c = torch.ones((n_batch, n_view, n_view), device=R[0][0].device)
        c -= torch.eye(n_view, device=c.device).unsqueeze(0).repeat(n_batch, 1, 1)

    # Rotation Sync. (R_{ij} = R_{i0} * R_{j0}^T)
    L = []
    c_rowsum = (torch.sum(c, dim=-1) - torch.diagonal(c, dim1=-2, dim2=-1))
    for view_i in range(n_view):
        L_row = []
        for view_j in range(n_view):
            if view_i == view_j:
                L_row.append(
                    torch.eye(3, dtype=c.dtype, device=c.device).unsqueeze(0).
                    repeat(n_batch, 1, 1) * c_rowsum[:, view_i].unsqueeze(1).unsqueeze(2)
                )
            else:
                L_row.append(
                    -c[:, view_i, view_j].unsqueeze(1).unsqueeze(2) *
                    R[view_i][view_j].transpose(-1, -2)
                )
        L_row = torch.stack(L_row, dim=-2)
        L.append(L_row)
    L = torch.stack(L, dim=1)
    L = L.reshape((-1, n_view * 3, n_view * 3))

    # Solve for 3 smallest eigen vectors (using SVD)
    try:
        e, V = torch.symeig(L, eigenvectors=True)
    except RuntimeError:
        if fallback_on_error:
            final_R = torch.stack(R[0], dim=1)
            final_t = torch.stack(t[0], dim=1)
            return final_R, final_t
        else:
            raise

    V = V[..., :3].reshape(n_batch, n_view, 3, 3)
    detV = torch.det(V.detach().contiguous()).sum(dim=1)
    Vlc = V[..., -1] * detV.sign().unsqueeze(1).unsqueeze(2)
    V = torch.cat([V[..., :-1], Vlc.unsqueeze(-1)], dim=-1)

    V = V.reshape(n_batch * n_view, 3, 3)
    u, s, v = torch.svd(V, some=False, compute_uv=True)
    R_optimal = torch.bmm(u, v.transpose(1, 2)).reshape(n_batch, n_view, 3, 3)

    # Translation Sync.
    b_elements = []
    for view_i in range(n_view):
        b_row_elements = []
        for view_j in range(n_view):
            b_row_elements.append(
                -c[:, view_i, view_j].unsqueeze(1) *
                torch.einsum('bnm,bn->bm', R[view_i][view_j], t[view_i][view_j])
            )
        b_row_elements = sum(b_row_elements)
        b_elements.append(b_row_elements)
    b_elements = torch.stack(b_elements, dim=1)
    b_elements = b_elements.reshape((n_batch, n_view * 3))
    t_optimal, _ = torch.solve(b_elements.unsqueeze(-1), L)
    t_optimal = t_optimal.reshape(n_batch, n_view, 3)
    return R_optimal, t_optimal


def sync_perm(lap: torch.Tensor, d: int, t: float = -np.inf):
    e, V = torch.symeig(lap, eigenvectors=True)
    V = V[..., :d]
    Zd = V.bmm(V.transpose(-1, -2))
    if t > 1e-5:
        Zd[Zd < t] = 0.0
    return Zd


def sync_motion_seg(z_mat: torch.Tensor, force_d: int = -1, t: float = -np.inf, cut_thres: float = 0.1):
    e, V = torch.symeig(z_mat, eigenvectors=True)
    if force_d != -1:
        d = force_d
    else:
        assert z_mat.size(0) == 1
        e_leading = e[:, -10:]
        total_est_points = torch.sum(e_leading, -1)
        e_th = total_est_points * cut_thres
        e_count = torch.sum(e.detach() > e_th, dim=1)
        d = e_count.max().item()

    V = V[..., -d:]  # (B, M, d)
    e = e[..., -d:]  # (B, d)
    V = V * e.sqrt().unsqueeze(1)
    v_sign = V.detach().sum(dim=1, keepdim=True).sign()
    V = V * v_sign
    if t > -1e5:
        V[V < t] = 0.0

    return V
