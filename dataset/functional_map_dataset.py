import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import scipy.sparse as sp

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
from utils.utils import create_diagonal_sparse_tensor


def load_csr_to_torch(data, name, device):
    # Retrieve the sparse matrix components
    name_data = data[f'{name}_data']
    name_indices = data[f'{name}_indices']
    name_indptr = data[f'{name}_indptr']
    name_shape = tuple(data[f'{name}_shape'])  # Ensure it's a tuple
    # Reconstruct the SciPy sparse matrix
    source_L_csr = sp.csr_matrix((name_data, name_indices, name_indptr), shape=name_shape)
    # Convert SciPy CSR to COO format (needed for PyTorch)
    source_L_coo = source_L_csr.tocoo()
    # Convert to PyTorch sparse tensor
    values = torch.tensor(source_L_coo.data, dtype=torch.float32, device=device)
    indices = torch.tensor(np.vstack((source_L_coo.row, source_L_coo.col)), dtype=torch.long, device=device)
    name_torch = torch.sparse_coo_tensor(indices, values, torch.Size(name_shape))
    # Ensure the tensor is in the correct format
    name_torch = name_torch.coalesce()
    return name_torch


class FunctionalMapDataset(Dataset):
    def __init__(self, data_folder, k1, k2, device=torch.device("cpu"), train=True, train_split=0,
                 with_bases=True, extended_data=False, external_test=False, external_test_data=None):
        """
        Initializes a dataset of precomputed Functional Maps stored as `.npz` files.
        Supports training/testing splits, external test sets.

        Args:
            data_folder (str): Path to the folder containing subfolders with `.npz` files.
            k1 (int): Number of basis vectors for the source mesh.
            k2 (int): Number of basis vectors for the target mesh.
            device (torch.device, optional): Device to load tensors onto. Default: CPU.
            train (bool, optional): Whether to load Train or Test data. Default: True.
            train_split (float, optional): Fraction [0,1] to split dataset for training. Default: 0.
            with_bases (bool, optional): If True, includes basis matrices in each sample.
            extended_data (bool, optional): If True, load extended geometric data. Default: False.
            external_test (bool, optional): If True, load test data from `external_test_data`. Default: False.
            external_test_data (str, optional): Path to external test folder. Used if `external_test=True`.
        """
        self.device = device
        self.k1, self.k2 = k1, k2
        self.with_bases = with_bases
        self.extended_data = extended_data

        # Get list of files
        if external_test and not train:
            file_list = [os.path.join(external_test_data, f, f + ".npz") for f in os.listdir(external_test_data)]
            if ("michael" in external_test_data) and ("michael" in data_folder):
                n_train = int(len(file_list) * train_split)
                file_list = file_list[n_train:]
        else:
            file_list = [os.path.join(data_folder, f, f + ".npz") for f in os.listdir(data_folder)]
            if train_split != 0:
                n_train = int(len(file_list) * train_split)
                file_list = file_list[:n_train] if train else file_list[n_train:]
        self.file_list = file_list
        self.data = self._load_data_parallel()

    def _load_data_parallel(self):
        """Loads dataset files in parallel using ThreadPoolExecutor."""
        data_list = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            data_list = list(executor.map(self._load_file, self.file_list))

        return data_list

    def _load_file(self, file_path):
        """Loads a single `.npz` file efficiently."""
        data = np.load(file_path, allow_pickle=True, mmap_mode='r')

        # Extract components and truncate basis
        fmap_gt = torch.tensor(data["fmap"][:self.k2, :self.k1], dtype=torch.float32, device=self.device)
        fmap_computed = torch.tensor(data["fmap_computed"][:self.k2, :self.k1], dtype=torch.float32, device=self.device)
        source_basis = torch.tensor(data["source_basis"][:, :self.k1], dtype=torch.float32, device=self.device)
        target_basis = torch.tensor(data["target_basis"][:, :self.k2], dtype=torch.float32, device=self.device)
        if not self.extended_data:
            return (source_basis, target_basis, fmap_gt, fmap_computed)

        source_eigvals = torch.tensor(data["source_eigvals"][:self.k1], dtype=torch.float32, device=self.device)
        target_eigvals = torch.tensor(data["target_eigvals"][:self.k2], dtype=torch.float32, device=self.device)
        # convert p2p vec to pytorch sparse tensor
        p2p = torch.tensor(data["p2p"].astype(np.int32), dtype=torch.int32, device=self.device)
        p2p_pairs = (torch.arange(p2p.shape[0], device=self.device), p2p)

        # convert scipy coo to pytorch sparse tensor
        p2p_opposite_mat_sci = data["p2p_opposite"].item().tocoo()
        values = torch.tensor(p2p_opposite_mat_sci.data, dtype=torch.float32, device=self.device)  # Nonzero values (all 1s)
        indices = torch.tensor(np.vstack((p2p_opposite_mat_sci.row, p2p_opposite_mat_sci.col)), dtype=torch.int64, device=self.device)  # Row, col indices
        p2p_opposite_mat = torch.sparse_coo_tensor(indices, values, p2p_opposite_mat_sci.shape, dtype=torch.float32)

        p2p_opposite_pairs = (torch.tensor(p2p_opposite_mat_sci.row, device=self.device),
                              torch.tensor(p2p_opposite_mat_sci.col, device=self.device))

        fmap_opposite = torch.tensor(data["fmap_opposite"][:self.k1, :self.k2], dtype=torch.float32, device=self.device)
        fmap_computed_opposite = torch.tensor(data["fmap_computed_opposite"][:self.k1, :self.k2], dtype=torch.float32, device=self.device)

        x1 = torch.tensor(data["source_vertices"], dtype=torch.float32, device=self.device)
        f1 = torch.tensor(data["source_faces"], dtype=torch.int32, device=self.device)

        x2 = torch.tensor(data["target_vertices"], dtype=torch.float32, device=self.device)
        f2 = torch.tensor(data["target_faces"], dtype=torch.int32, device=self.device)

        mass1_vec = torch.tensor(data["source_mass_vec"], dtype=torch.float32, device=self.device).squeeze(0)
        mass1 = create_diagonal_sparse_tensor(mass1_vec, self.device)

        mass2_vec = torch.tensor(data["target_mass_vec"], dtype=torch.float32, device=self.device).squeeze(0)
        mass2 = create_diagonal_sparse_tensor(mass2_vec, self.device)

        L1 = load_csr_to_torch(data, "source_L", self.device)
        L2 = load_csr_to_torch(data, "target_L", self.device)

        return (source_basis, target_basis, source_eigvals, target_eigvals,
                fmap_gt, fmap_computed, fmap_opposite, fmap_computed_opposite,
                p2p_pairs, p2p_opposite_mat,
                x1, x2, f1, f2, mass1, mass2, L1, L2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.extended_data:
            source_basis, target_basis, source_eigvals, target_eigvals, fmap_gt, fmap_computed, fmap_opposite, fmap_computed_opposite, p2p, p2p_opposite_mat, x1, x2, f1, f2, mass1, mass2, L1, L2 = self.data[idx]
            if self.with_bases:
                # return source_basis, target_basis, source_eigvals, target_eigvals, fmap_gt, fmap_computed, fmap_opposite, fmap_computed_opposite, p2p, p2p_opposite_mat, x1, x2, f1, f2, mass1, mass2, L1, L2
                return source_basis, target_basis, source_eigvals, target_eigvals, fmap_gt, fmap_computed, p2p, x1, x2, f1, f2, mass1, mass2, L1, L2
            else:   # for batch_size != 1 should have data of the same size for all examples
                return fmap_gt, fmap_computed
        else:
            source_basis, target_basis, fmap_gt, fmap_computed = self.data[idx]
            if self.with_bases:
                return source_basis, target_basis, fmap_gt, fmap_computed
            else:
                return fmap_gt, fmap_computed


if __name__ == "__main__":
    k1, k2 = 50, 50
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_folder = "../data_processed/david_pairs"
    dataset = FunctionalMapDataset(data_folder, k1, k2, device, train=False, train_split=0.85, with_bases=False, extended_data=False)
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for i, (fmap_gt, fmap_computed) in enumerate(dataloader):
        print(f"Batch {i + 1}")
        print(f"Fmap GT Shape: {fmap_gt.shape}")
        print(f"Fmap Computed Shape: {fmap_computed.shape}")
        break  # Just show the first batch
