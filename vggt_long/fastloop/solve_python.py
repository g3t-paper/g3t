import torch
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
# from typing import Tuple, List

def solve_sparse(A: csc_matrix, b: np.ndarray, freen: int) -> np.ndarray:
    """Solve linear system A * delta = b, supports submatrix solving"""
    if freen < 0:
        return spsolve(A, b)
    else:
        A_sub = A[:freen, :freen].tocsc()
        b_sub = b[:freen]
        delta_sub = spsolve(A_sub, b_sub)
        delta = np.zeros_like(b)
        delta[:freen] = delta_sub
        return delta

def solve_system_py(
    J_Ginv_i: torch.Tensor,
    J_Ginv_j: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    res: torch.Tensor,
    ep: float,
    lm: float,
    freen: int,
    dof: int = 7
) -> torch.Tensor:
    # Ensure all tensors are on CPU
    device = res.device
    J_Ginv_i = J_Ginv_i.cpu()
    J_Ginv_j = J_Ginv_j.cpu()
    ii = ii.cpu()
    jj = jj.cpu()
    res = res.clone().cpu()

    r = res.size(0)  # Number of edges
    n = max(ii.max().item(), jj.max().item()) + 1  # Number of nodes

    res_vec = res.view(-1).numpy().astype(np.float64)

    rows, cols, data = [], [], []
    ii_np = ii.numpy()
    jj_np = jj.numpy()
    J_Ginv_i_np = J_Ginv_i.numpy()
    J_Ginv_j_np = J_Ginv_j.numpy()

    for x in range(r):
        i = ii_np[x]
        j = jj_np[x]
        if i == j:
            raise ValueError("Self-edges are not allowed")

        for k in range(dof):
            for l in range(dof):  # noqa
                row_idx = x * dof + k
                col_idx_i = i * dof + l
                val_i = J_Ginv_i_np[x, k, l]
                rows.append(row_idx)
                cols.append(col_idx_i)
                data.append(val_i)

                col_idx_j = j * dof + l
                val_j = J_Ginv_j_np[x, k, l]
                rows.append(row_idx)
                cols.append(col_idx_j)
                data.append(val_j)

    J = coo_matrix((data, (rows, cols)), shape=(r * dof, n * dof)).tocsc()

    b_vec = - J.T @ res_vec

    A_mat = J.T @ J

    diag = A_mat.diagonal()
    new_diag = diag * (1.0 + lm) + ep
    A_mat.setdiag(new_diag)

    freen_total = freen * dof
    delta = solve_sparse(A_mat.tocsc(), b_vec, freen_total)

    delta_tensor = torch.from_numpy(delta.astype(np.float32)).view(n, dof).to(device)
    return delta_tensor
