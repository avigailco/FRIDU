import numpy as np
import torch
import scipy.sparse
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
import os
import matplotlib.pyplot as plt


def compute_landmark_energy(C12, Psi1, Psi2, landmark_pairs, str_type=""):
    """
    Computes the landmark correspondence term:

        E^m(C12, P) = sum_{(i,j) in P} || Psi1[i] @ C12 - Psi2[j] ||^2

    Parameters
    ----------
    C12 : torch.Tensor of shape (k1, k2)
        The functional map matrix.
    Psi1 : torch.Tensor of shape (n1, k1)
        Basis (or descriptor) matrix for the source shape.
    Psi2 : torch.Tensor of shape (n2, k2)
        Basis (or descriptor) matrix for the target shape.
    landmark_pairs : list of two tensors representing landmarks pairs,
                    the first representing index on the source and the second representing index on the target
    Returns
    -------
    float
        The sum of squared differences for the given landmark pairs.
    """
    # Ensure indices are long tensors for indexing
    rows, cols = landmark_pairs[0].long(), landmark_pairs[1].long()
    # Batch computation: Extract indexed rows from Psi1 and Psi2
    Psi1_selected = Psi1[rows, :]  # Shape: (m, k)
    Psi2_selected = Psi2[cols, :]  # Shape: (m, k)
    # Compute transformed source basis
    transformed_source = Psi1_selected @ C12  # Shape: (m, k)

    # visualize
    # pair_idx = 10
    # v1 = transformed_source[0, pair_idx].cpu().numpy()
    # v2 = Psi2_selected[0, pair_idx].cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.plot(v1, color='red', label='v1')
    # plt.plot(v2, color='blue', label='v2')
    # diff = np.sum(np.abs(v1 - v2))  # np.sum((v1 - v2)**2)
    # plt.xlabel("eigenvector idx")
    # plt.ylabel("embedding")
    # plt.title(f"{str_type}, pair ({rows[0,pair_idx]}, {cols[0,pair_idx]}), diff={diff:.4f}")
    # plt.legend()
    # plt.show()

    # Compute squared differences
    # energy = torch.sum(torch.sum((transformed_source - Psi2_selected)**2, dim=2), dim=1)
    diff = transformed_source - Psi2_selected
    energy_verts = torch.norm(diff, dim=2, p='fro').squeeze(0)**2
    energy = torch.norm(diff, p="fro") ** 2
    energy /= rows.shape[1]
    return energy.item(), energy_verts


def compute_pointwise_map(C12, Psi1, Psi2, Psi2_pinv=None):
    """
    Computes the pointwise map P from the functional map C using the equation:
        P12 = Psi1 * C12 * pinv(Psi2)

    Parameters
    ----------
    C12 : torch.Tensor of shape (k1, k2)
        The functional map matrix from source to target.
    Psi1 : torch.Tensor of shape (n1, k1)
        Basis matrix for the source shape.
    Psi2 : torch.Tensor of shape (n2, k2)
        Basis matrix for the target shape.

    Returns
    -------
    P12 : torch.Tensor of shape (n1, n2)
        The reconstructed pointwise map.
    """
    # Compute the pseudo-inverse of Psi2 using torch.linalg.pinv
    if Psi2_pinv is None:
        Psi2_pinv = torch.linalg.pinv(Psi2)  # Shape (k2, n2)

    # Compute P12 using the formula P12 = Psi1 * C12 * Psi2_pinv
    P12 = Psi1 @ C12 @ Psi2_pinv  # Shape (n1, n2)
    return P12


def compute_vertex_to_vertex_map_batched(C12, Phi1, Phi2, batch_size=1000):
    """
    Compute a vertex-to-vertex map from a functional map C12 using batched nearest-neighbor search in spectral space.

    Args:
        C12 (torch.Tensor): Functional map (k1 x k2).
        Phi1 (torch.Tensor): Basis functions for shape M1 (n1 x k1).
        Phi2 (torch.Tensor): Basis functions for shape M2 (n2 x k2).
        batch_size (int): Number of vertices to process at once to reduce memory usage.

    Returns:
        torch.Tensor: Vertex-to-vertex map (size n1) where each index in M1 maps to an index in M2.
        torch.sparse_coo_tensor: Sparse P12 matrix mapping vertices in M1 to M2.
    """
    device = Phi1.device  # Ensure tensors are on the same device
    projected_Phi1 = Phi1 @ C12  # (n1 x k)

    n1, n2 = Phi1.shape[0], Phi2.shape[0]
    vertex_map = torch.zeros(n1, dtype=torch.long, device=device)  # Output mapping

    # Process in batches to avoid OOM errors
    for i in range(0, n1, batch_size):
        batch = projected_Phi1[i: i + batch_size]  # (batch_size x k)

        # Compute pairwise distances for batch only
        dists = torch.cdist(batch, Phi2)  # (batch_size, n2) - Efficient L2 distance in PyTorch
        vertex_map[i: i + batch_size] = torch.argmin(dists, dim=1)  # Nearest neighbor index

    # Construct sparse P12 matrix
    values = torch.ones(n1, dtype=torch.float32, device=device)
    rows = torch.arange(n1, dtype=torch.int64, device=device)
    cols = vertex_map
    P12 = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, size=(n1, n2), dtype=torch.float32)

    return vertex_map, P12


def compute_vertex_to_vertex_map(C12, Phi1, Phi2):
    """
    Compute a vertex-to-vertex map from a functional map C21 using nearest-neighbor search in spectral space.
    Args:
        C21 (torch.Tensor): Functional map (k x k).
        Phi1 (torch.Tensor): Basis functions for shape M1 (n1 x k).
        Phi2 (torch.Tensor): Basis functions for shape M2 (n2 x k).

    Returns:
        torch.Tensor: Vertex-to-vertex map (size n1) where each index in M1 maps to an index in M2.
    """
    # Step 1: Project functional map onto spectral basis
    projected_Phi1 = Phi1 @ C12  # (n1 x k) @ (k x k) → (n1 x k)

    # Step 2: Compute pairwise Euclidean distances between projected_Phi1 and Phi2
    # Expanding dimensions to compute (n1 x n2) distance matrix efficiently
    Phi1_exp = projected_Phi1.unsqueeze(1)  # (n1, 1, k)
    Phi2_exp = Phi2.unsqueeze(0)  # (1, n2, k)
    # Compute squared Euclidean distance ||Phi1_exp - Phi2_exp||^2
    dist_matrix = torch.sum((Phi1_exp - Phi2_exp) ** 2, dim=2)  # (n1, n2)
    # Step 3: Find nearest neighbor in Phi2 for each vertex in Phi1
    # vertex_map[i] gives the index in M2 corresponding to vertex i in M1
    vertex_map = torch.argmin(dist_matrix, dim=1)  # (n1,)

    n1, n2 = Phi1.shape[0], Phi2.shape[0]
    values = torch.ones(n2, dtype=torch.float32, device=Phi1.device)
    rows = torch.arange(n2, dtype=torch.float32, device=Phi1.device)
    cols = vertex_map
    P12 = torch.sparse_coo_tensor((rows, cols), values, C12.shape, dtype=torch.float32)
    return vertex_map, P12


def pairwise_euclidean_distances_chunked(x, chunk_size=10000):
    """
    Computes pairwise Euclidean distances in chunks to reduce memory usage.

    Args:
        x (torch.Tensor): Tensor of shape (N, D) representing N points in D-dimensional space.
        chunk_size (int): Number of rows to process at a time.

    Returns:
        torch.Tensor: Pairwise distance matrix of shape (N, N).
    """
    N = x.shape[0]
    pairwise_distances = torch.zeros((N, N), device=x.device)  # Allocate output matrix

    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        pairwise_distances[i:end_i] = torch.cdist(x[i:end_i], x, p=2)  # Compute chunk

    return pairwise_distances


def get_sorted_cdf(errors):
    sorted_errors = np.sort(errors.cpu().numpy())
    cdf_y = np.linspace(0, 1, len(sorted_errors))  # normalized to [0,1]
    return sorted_errors, cdf_y


def get_pair_cdf(P21_gt, P21_pred, P21_computed, vertices1):
    gt_idx = P21_gt.coalesce().indices()[1]
    pred_idx = P21_pred.coalesce().indices()[1]
    computed_idx = P21_computed.coalesce().indices()[1]

    pairwise_dists = pairwise_euclidean_distances_chunked(vertices1)
    dists_pred = pairwise_dists[gt_idx, pred_idx]
    dists_computed = pairwise_dists[gt_idx, computed_idx]

    # vertices: Tensor of shape (n_vertices, 3)
    min_corner = vertices1.min(dim=0).values  # shape (3,)
    max_corner = vertices1.max(dim=0).values  # shape (3,)
    diagonal = torch.norm(max_corner - min_corner, p=2)

    dists_pred /= diagonal
    dists_computed /= diagonal

    # return get_sorted_cdf(dists_pred), get_sorted_cdf(dists_computed)
    return dists_pred, dists_computed


def plot_euclidean_err_comparison(P21_gt, P21_pred, P21_computed, vertices1):
    gt_idx, pred_idx, computed_idx = P21_gt.coalesce().indices()[1], P21_pred.coalesce().indices()[1], \
    P21_computed.coalesce().indices()[1]
    pairwise_dists = pairwise_euclidean_distances_chunked(vertices1)
    dists_pred = pairwise_dists[gt_idx, pred_idx]
    sorted_errors = np.sort(dists_pred.cpu().numpy())  # Sort errors in ascending order
    y_axis = np.linspace(0, 100, len(sorted_errors))  # cumulative percentage of correspondences
    plt.plot(sorted_errors, y_axis, label='refined', color='g', linestyle='-')

    dists_computed = pairwise_dists[gt_idx, computed_idx]
    sorted_errors = np.sort(dists_computed.cpu().numpy())  # Sort errors in ascending order
    y_axis = np.linspace(0, 100, len(sorted_errors))  # cumulative percentage of correspondences
    plt.plot(sorted_errors, y_axis, label='initial', color='r', linestyle='-')

    plt.legend()
    plt.grid(True)
    # plt.title("Cumulative Correspondence Plot")
    plt.ylabel("% Correspondences")
    plt.xlabel("Euclidean Error")
    plt.show()


def pairwise_euclidean_distances(x):
    """
    Computes pairwise Euclidean distances using PyTorch.

    Args:
        x (torch.Tensor): Tensor of shape (N, D) representing N points in D-dimensional space.

    Returns:
        torch.Tensor: Pairwise distance matrix of shape (N, N).
    """
    # Compute squared norms for each row
    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # Shape: (N, 1)

    # Use broadcasting to compute pairwise distances efficiently
    distances = x_norm - 2 * torch.mm(x, x.T) + x_norm.T  # (N, N) pairwise squared distances

    # Ensure non-negative distances (numerical stability)
    distances = torch.sqrt(torch.clamp(distances, min=0.0))

    return distances


def plot_err_graph_over_dataset(refined_err_lst, init_err_lst, x_label_str, save_path=None):
    # Step 1: Compute global max error across all tensors
    all_max = max(
        max([x.max().item() for x in refined_err_lst]),
        max([x.max().item() for x in init_err_lst])
    )
    x_uniform = np.linspace(0, all_max, 100)

    # Step 2: Compute CDFs directly from PyTorch tensors
    def compute_cdf(errors, x_uniform):
        return np.array([(errors <= t).float().mean().item() for t in x_uniform])

    refined_cdfs = [compute_cdf(errors, x_uniform) for errors in refined_err_lst]
    init_cdfs = [compute_cdf(errors, x_uniform) for errors in init_err_lst]
    refined_cdfs = np.vstack(refined_cdfs)
    init_cdfs = np.vstack(init_cdfs)
    mean_refined = refined_cdfs.mean(axis=0)
    std_refined = refined_cdfs.std(axis=0)
    mean_init = init_cdfs.mean(axis=0)
    std_init = init_cdfs.std(axis=0)

    plt.plot(x_uniform, mean_refined * 100, label='refined', color='g')
    plt.fill_between(x_uniform, (mean_refined - std_refined) * 100, (mean_refined + std_refined) * 100, color='g', alpha=0.2)

    plt.plot(x_uniform, mean_init * 100, label='initial', color='r')
    plt.fill_between(x_uniform, (mean_init - std_init) * 100, (mean_init + std_init) * 100, color='r', alpha=0.2)

    plt.xlabel(x_label_str)
    plt.ylabel("% Correspondences")
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        # plt.title("Average Cumulative Correspondence Curve")
        figure_save_path = os.path.join(save_path, "err_graph_figure.png")
        plt.savefig(figure_save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def compute_point2point_guidance(C21, mass2, psi1, psi2, vertex_map=None):
    """
    Computes the guidance loss:
        Energy = || psi_2 @ C21 - P21 @ psi1 ||_mass2^2 (||v||_mass^2 = v^T @ mass @ v)

    Parameters
    ----------
    C21 : torch.Tensor of shape (1, 1, k2, k1)
        The functional map matrix.
    mass2 : torch.Tensor of shape (1, n2, n2)
        The mass matrix for the source shape.
    psi1 : torch.Tensor of shape (1, n1, k1)
        Basis (or descriptor) matrix for the source shape.
    psi2 : torch.Tensor of shape (1, n2, k2)
        Basis (or descriptor) matrix for the target shape.

    Returns
    -------
    torch.Tensor
        The guidance loss.
    """
    C21 = C21.to(psi2.dtype)
    # extract vertex to vertex mapping
    if vertex_map is None:
        with torch.no_grad():
            vertex_map, _ = compute_vertex_to_vertex_map_batched(C21[0, 0], psi2[0], psi1[0])
        # _, P21 = compute_vertex_to_vertex_map_batched(C21[0, 0].to(psi2.dtype), psi2[0], psi1[0]).coalesce()
    expr1 = psi1[:, vertex_map, :]  # equivalent to P21 @ psi1. Shape: (1, n2, k1)
    expr2 = torch.matmul(psi2, C21.squeeze(0))  # Shape: (1, n2, k1)
    diff = expr1 - expr2  # Shape: (1, n2, k1)
    energy = torch.trace(torch.matmul(diff.transpose(1, 2), torch.bmm(mass2, diff)).squeeze(0))

    return energy


def frobenius_loss(a, b, w=None, minval=None, maxval=None):
    assert a.dim() == b.dim() == 3
    loss = (a - b) ** 2
    if w is not None:
        assert w.dim() == 3
        loss = loss * w
    loss = torch.sum(loss, axis=(1, 2))
    if minval is not None:
        loss = torch.clamp(loss, min=minval)
    if maxval is not None:
        loss = torch.clamp(loss, max=maxval)
    return torch.mean(loss)


def orthogonality_s_loss(fmap01):
    fmap01 = fmap01[0]
    B, K1, K0 = fmap01.shape

    I0 = torch.eye(K0).to(fmap01)
    # W0 = torch.ones_like(I0) * (float(K0) / K0**2)
    # W0.fill_diagonal_(float(K0**2 - K0) / K0**2)

    I0 = torch.unsqueeze(I0, dim=0)
    # W0 = torch.unsqueeze(W0, dim=0)

    loss = frobenius_loss(torch.transpose(fmap01, 1, 2) @ fmap01, I0, w=None)
    return loss


def _get_mask(evals1, evals2, resolvant_gamma):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1 / scaling_factor, evals2 / scaling_factor
    evals_gamma1 = (evals1 ** resolvant_gamma)[None, :]
    evals_gamma2 = (evals2 ** resolvant_gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()


def get_mask(evals1, evals2, resolvant_gamma):
    masks = []
    for bs in range(evals1.shape[0]):
        masks.append(_get_mask(evals1[bs], evals2[bs], resolvant_gamma))
    return torch.stack(masks, dim=0)


def lap_bij_resolvant(C, evals1, evals2):
    k = C.shape[-1]
    evals1[evals1 < 0] = 0 # first eigenvalue, due to numerical issues
    evals2[evals2 < 0] = 0
    mask = get_mask(evals1, evals2, 0.5)  # (B, K2_init, K1_init)
    loss = torch.linalg.norm(C * mask) ** 2 / (k * k)
    # self.cfg.loss.w_lap_bij_resolvant * 1e2 *
    return loss


def check_functional_map_identity(C12, C21):
    """
    Computes deviation of C12 @ C21 from identity.

    Parameters
    ----------
    C12 : torch.Tensor of shape (k1, k2)
        Functional map from source to target.
    C21 : torch.Tensor of shape (k2, k1)
        Functional map from target to source.

    Returns
    -------
    torch.Tensor
        Frobenius norm of deviation from identity.
    """
    k1, k2 = C12.shape
    I_k1 = torch.eye(k1, device=C12.device)
    I_k2 = torch.eye(k2, device=C12.device)

    # Compute errors
    error_12 = torch.norm(C12 @ C21 - I_k1, p="fro") / torch.norm(I_k1, p="fro")
    error_21 = torch.norm(C21 @ C12 - I_k2, p="fro") / torch.norm(I_k2, p="fro")

    print(f"||C12 @ C21 - I||_F / ||I||_F = {error_12.item():.6f}")
    print(f"||C21 @ C12 - I||_F / ||I||_F = {error_21.item():.6f}")

    return error_12, error_21


if __name__ == "__main__":
    # Example usage (toy data):
    k1, k2 = 10, 15  # Number of basis functions
    n1, n2 = 100, 120  # Number of vertices in source and target
    # Random functional map and basis matrices for demonstration
    C12 = torch.randn(k1, k2)
    Psi1 = torch.randn(n1, k1)
    Psi2 = torch.randn(n2, k2)
    # Compute the pointwise map P12
    P12 = compute_pointwise_map(C12, Psi1, Psi2)
    print("Computed pointwise map P12 shape:", P12.shape)  # Should be (n1, n2)
