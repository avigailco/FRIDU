from scipy.sparse import coo_matrix
from scipy.io import loadmat
import os
import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
from pyFM.spectral import mesh_p2p_to_FM
from pyFM.functional import FunctionalMapping
import re
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
from utils.noff_files import load_noff, noff_to_TriMesh
from utils.get_fmap_cmap import create_custom_colormap

cmap = create_custom_colormap('default')
# -------------------------------------------------------
# data path
base_path = os.getcwd()

# TODO: change to accurate path of offs, gt_matches and landmarks (if used)
mesh_path = os.path.join(base_path, "..", "data", "TACO", "offs")  # NOFF, includes also normals
gt_matches_path = os.path.join(base_path, "..", "data", "TACO", "gt_matches")
landmarks_path = os.path.join(base_path, "..", "data", "TACO", "landmarks")

mesh_name = 'horse'  # {'cat', 'centaur', 'david', 'dog', 'gorilla', 'horse', 'michael', 'victoria', 'wolf'}
# fmap parameters
use_landmarks = False
k1, k2 = 128, 128
# WKS parameters
n_descr = 100
subsample_step = 5
# -------------------------------------------------------


def compute_fmap_wks(mesh1, mesh2, mesh_k1, mesh_k2, landmarks=None, icp_iterations=0):
    process_params = {
        'n_ev': (mesh_k1, mesh_k2),  # Number of eigenvalues on source and target
        'k_process': max(mesh_k1, mesh_k2),
        'descr_type': 'WKS',  # WKS or HKS
        'n_descr': n_descr,
        'subsample_step': subsample_step,  # In order not to use too many descriptors
        'landmarks': landmarks
    }
    model = FunctionalMapping(mesh1, mesh2)
    model.preprocess(**process_params, verbose=True)
    fit_params = {
        'w_descr': 1e0,
        'w_lap': 1e-2,
        'w_dcomm': 1e-1,
        'w_orient': 0
    }
    model.fit(**fit_params, verbose=True)
    cmap_12 = model.FM
    if icp_iterations != 0:
        model.icp_refine(nit=icp_iterations)
        cmap_12 = model._FM_icp
    # plt.imshow(cmap_12, cmap=cmap)
    # plt.colorbar()
    # plt.show()
    return cmap_12


def plot_fmaps(fmap_gt, fmap_computed, fmap_gt_opp, fmap_computed_opp, computed_str, title):
    n_cols = 2  # number of columns per row
    # compute shared normalization for the top row (2->1)
    vmin_top = min(fmap_gt.min(), fmap_computed.min())
    vmax_top = max(fmap_gt.max(), fmap_computed.max())
    # compute shared normalization for the bottom row (1->2)
    vmin_bot = min(fmap_gt_opp.min(), fmap_computed_opp.min())
    vmax_bot = max(fmap_gt_opp.max(), fmap_computed_opp.max())

    # create a figure with 2 rows; each row corresponds to one "direction"
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
    row_titles = [r"$1\rightarrow 2$", r"$2\rightarrow 1$"]
    col_titles = ["GT", f"Computed{computed_str}"]
    # ---- Top row: "1->2" ----
    im1 = axes[0, 0].imshow(fmap_gt, cmap=cmap, vmin=vmin_top, vmax=vmax_top)
    im2 = axes[0, 1].imshow(fmap_computed, cmap=cmap, vmin=vmin_top, vmax=vmax_top)
    row1_mappable = im1
    # add a shared colorbar for the top row
    fig.colorbar(row1_mappable, ax=axes[0, :], orientation='vertical')
    # add a left-side row label for the top row
    axes[0, 0].set_ylabel(row_titles[0], fontsize=10, rotation=0, labelpad=20, va='center')
    # ---- Bottom row: "2->1" ----
    im4 = axes[1, 0].imshow(fmap_gt_opp, cmap=cmap, vmin=vmin_bot, vmax=vmax_bot)
    im5 = axes[1, 1].imshow(fmap_computed_opp, cmap=cmap, vmin=vmin_bot, vmax=vmax_bot)
    row2_mappable = im4
    # add a shared colorbar for the bottom row
    fig.colorbar(row2_mappable, ax=axes[1, :], orientation='vertical')
    # add a left-side row label for the bottom row
    axes[1, 0].set_ylabel(row_titles[1], fontsize=10, rotation=0, labelpad=20, va='center')

    # --- add column labels above the top row ---
    # for each column, compute the center of the subplot axes and add a text label
    for j, col_title in enumerate(col_titles):
        pos = axes[0, j].get_position()
        x = pos.x0 + pos.width / 2
        y = pos.y1 + 0.01  # slight offset above the subplot
        fig.text(x, y, col_title, ha='center', va='bottom', fontsize=12)
    # add a common title for the whole figure and adjust layout
    plt.suptitle(title, fontsize=16)
    # plt.show()


def visualize_meshes_function_mapping(source_mesh, target_mesh, fmap):
    ps.init()
    # Register the surface meshes
    align = [-50, 0, 0]
    ps_mesh1 = ps.register_surface_mesh("Mesh 1", source_mesh.vertices, source_mesh.faces)
    ps_mesh2 = ps.register_surface_mesh("Mesh 2", target_mesh.vertices + align, target_mesh.faces)

    f1 = source_mesh.eigenvectors[:, 20]
    f2 = target_mesh.unproject(fmap @ source_mesh.project(f1, k=k1))
    ps_mesh1.add_scalar_quantity("Function 1", f1, enabled=True)
    ps_mesh2.add_scalar_quantity("Function 2", f2, enabled=True)

    ps.show()


def generate_dataset():
    # generate the metadata folder for the chosen mesh (in TACO)
    output_path = os.path.join(base_path, "..", "data_processed")
    mt_path = os.path.join(output_path, f"{mesh_name}_pairs") + use_landmarks*"_landmarks"
    os.makedirs(mt_path, exist_ok=True)

    # iterate over the relevant gt matches and generate the metadata for FRIDU training
    for file_name in os.listdir(gt_matches_path):
        mesh_name_i = re.split(r'(\d)', file_name, maxsplit=1)[0]
        if mesh_name_i != mesh_name:
            continue
        save_folder = os.path.join(mt_path, f"{file_name.split('.')[0]}")
        if os.path.exists(save_folder):
            continue
        os.mkdir(save_folder)
        # **************************************************************************************
        # infer corresponding source and target meshes (file name pattern is target_source.mat)
        # **************************************************************************************
        split1 = file_name.split("_")
        target = split1[0]
        source = split1[1].split('.')[0]
        source_mesh = noff_to_TriMesh(os.path.join(mesh_path, f'{source}.off'), normalize_mesh=False)
        target_mesh = noff_to_TriMesh(os.path.join(mesh_path, f'{target}.off'), normalize_mesh=False)
        # **************************************************************************************
        # compute correspondence between original pair
        # **************************************************************************************
        # T is (n2,), hence \P_21 \in (n2,n1), source -> target
        p2p_gt = np.asarray(loadmat(os.path.join(gt_matches_path, f"{target}_{source}.mat"))['Pi']).flatten() - 1  # indices range correction
        # compute C21 \in (n2,n1)
        source_mesh.process(k=max(k1, k2))
        target_mesh.process(k=max(k1, k2))
        fmap_gt = mesh_p2p_to_FM(p2p_gt, source_mesh, target_mesh, dims=(k1, k2), subsample=None)
        # visualize_meshes_function_mapping(source_mesh, target_mesh, fmap_gt)
        # **************************************************************************************
        # compute correspondence in the opposite direction (naive approach)
        # **************************************************************************************
        # \P_12 in (n1,n2), target -> source, (not mandatory for the training itself!)
        shape = (source_mesh.n_vertices, target_mesh.n_vertices)
        rows, cols, values = p2p_gt, np.arange(p2p_gt.shape[0]), np.ones(p2p_gt.shape[0])
        p2p_gt_opposite = coo_matrix((values, (rows, cols)), shape=shape)
        # normalize rows to ensure sum of each row is 1
        row_sums = np.array(p2p_gt_opposite.sum(axis=1)).flatten()  # Sum per row
        row_sums[row_sums == 0] = 1  # avoid division by zero
        p2p_gt_opposite.data /= row_sums[p2p_gt_opposite.row]
        # compute C12 \in (n1,n2)
        fmap_gt_opposite = mesh_p2p_to_FM(p2p_gt_opposite, target_mesh, source_mesh, dims=(k2, k1), subsample=None)
        # **************************************************************************************
        if use_landmarks:
            landmarks_src = np.load(os.path.join(landmarks_path, mesh_name_i, source, "landmarks.npy"))
            landmarks_target = np.load(os.path.join(landmarks_path, mesh_name_i, target, "landmarks.npy"))
            landmarks = np.stack((landmarks_src, landmarks_target)).T
            landmarks_opp = np.stack((landmarks_target, landmarks_src)).T
        else:
            landmarks, landmarks_opp = None, None
        # **************************************************************************************
        # compute fmap (k2,k1) based on WKS descriptors
        # **************************************************************************************
        fmap_computed = compute_fmap_wks(source_mesh, target_mesh, k1, k2, landmarks=landmarks)
        # **************************************************************************************
        # compute fmap (k1,k2) based on WKS descriptors
        # **************************************************************************************
        # fmap_computed_opposite = fmap_gt_opposite  # placeholder, accelerate calculation in case not needed for training
        fmap_computed_opposite = compute_fmap_wks(target_mesh, source_mesh, k2, k1, landmarks=landmarks_opp)
        # **************************************************************************************
        # save data
        # **************************************************************************************
        data_dict = {"p2p": p2p_gt,
                     "fmap": fmap_gt,
                     "fmap_computed": fmap_computed,    # initial fmap
                     "p2p_opposite": p2p_gt_opposite,
                     "fmap_opposite": fmap_gt_opposite,     # gt (naive) fmap in the opposite direction
                     "fmap_computed_opposite": fmap_computed_opposite,
                     'source_basis': source_mesh.eigenvectors[:, :k1],
                     'target_basis': target_mesh.eigenvectors[:, :k2],
                     'source_eigvals': source_mesh.eigenvalues[:k1],
                     'target_eigvals': target_mesh.eigenvalues[:k2],
                     'source_vertices': source_mesh.vertices,
                     'target_vertices': target_mesh.vertices,
                     'source_faces': source_mesh.faces,
                     'target_faces': target_mesh.faces,

                     'source_L_data': source_mesh.W.data,   # cotangent
                     'source_L_indices': source_mesh.W.indices,
                     'source_L_indptr': source_mesh.W.indptr,
                     'source_L_shape': source_mesh.W.shape,

                     'target_L_data': target_mesh.W.data,
                     'target_L_indices': target_mesh.W.indices,
                     'target_L_indptr': target_mesh.W.indptr,
                     'target_L_shape': target_mesh.W.shape,

                     'source_mass_vec': source_mesh.A.data,  # mass
                     'target_mass_vec': target_mesh.A.data
                     }
        save_path = os.path.join(save_folder, f"{file_name.split('.')[0]}.npz")
        np.savez(save_path, **data_dict)
        # loaded_data = np.load(save_path)
        # **************************************************************************************
        # visualize fmaps
        # **************************************************************************************
        computed_str = ""  # f" (n_desc={n_descr}, samp_step={subsample_step})"
        plot_fmaps(fmap_gt, fmap_computed, fmap_gt_opposite, fmap_computed_opposite,
                   computed_str, title=f"Source: {source} (k1={k1}), Target: {target} (k2={k2})")
        plt.savefig(os.path.join(save_folder, "fmap.png"), format="png", dpi=300)
        # **************************************************************************************


if __name__ == "__main__":
    generate_dataset()
