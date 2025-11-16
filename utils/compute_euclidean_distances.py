import os, sys
from scipy.spatial.distance import cdist
import numpy as np
from noff_files import noff_to_TriMesh

# -----------------------------------------------------------------------------------
# data path
base_path = os.getcwd()
mesh_path = os.path.join(base_path, "..", "..", "data", "TACO", "offs")  # NOFF, includes also normals


def compute_euclidean_distances(figure_name="michael"):
    geodesic_cache_dir = "../euclidean_distances"
    os.makedirs(geodesic_cache_dir, exist_ok=True)
    # -----------------------------------------------------------------------------------
    for file_name in os.listdir(mesh_path):
        curr_figure_name = file_name.split('.')[0]
        if not curr_figure_name.startswith(figure_name):
            continue
        # save_folder = os.path.join(geodesic_cache_dir, f"{curr_figure_name}.npy")
        save_folder = os.path.join(geodesic_cache_dir, f"{curr_figure_name}.npz")
        if os.path.exists(save_folder):
            continue
        tri_mesh = noff_to_TriMesh(os.path.join(mesh_path, file_name))

        # Compute pairwise Euclidean distances between vertices
        vertices = tri_mesh.vertices  # Assuming tri_mesh has a `.vertices` attribute as an (N, 3) array
        pairwise_distances = cdist(vertices, vertices, metric='euclidean')

        # Save the computed distances
        pairwise_distances = pairwise_distances.astype(np.float32)
        np.savez_compressed(save_folder, distances=pairwise_distances)
        # np.save(save_folder, pairwise_distances)
        print(f"Saved Euclidean distances for {curr_figure_name} to {save_folder}")


if __name__ == "__main__":
    compute_euclidean_distances()
