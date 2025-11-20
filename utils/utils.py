import wandb
import json
import hashlib
import random
import torch
import numpy as np
import pickle
import argparse
import ast


def parse_args():
    parser = argparse.ArgumentParser(description="Training script with SLURM srun support")

    parser.add_argument('--job_num', type=int, default=0)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate rather than training")
    parser.add_argument("--log_wb", action="store_true", help="W&B logging")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--device', type=str, default="cuda", help="Compute device (cuda/cpu)")

    parser.add_argument("--upsampling", action="store_true", help="upsampling during generation")
    parser.add_argument("--no_guidance", action="store_true", help="guidance free generation")
    parser.add_argument('--regularize_guidance', type=int, default=0, help="[0: only p2p guidance, 1: +orth, 2: +comm, 3: +ortho+comm]")
    parser.add_argument("--fast_training", action="store_true", help="Only training without generation")
    parser.add_argument("--save_resume_ckpt", action="store_true", help="Save checkpoint for resume")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from checkpoint")

    parser.add_argument('--data', type=str, default="../data_processed/david_pairs", help="Train dataset path")
    parser.add_argument('--external_test', action='store_true', help="test on other dataset")
    parser.add_argument('--external_test_data', type=str, default="../data_processed/david_pairs", help="Path to dataset")

    parser.add_argument('--n_patch_res', type=int, default=2, help="Number of resolutions")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout probability for intermediate activations")
    parser.add_argument('--num_blocks', type=int, default=2, help="Number of residual blocks per resolution")
    parser.add_argument('--model_channels', type=int, default=16, help="Base multiplier for the number of channels in the network")
    parser.add_argument('--channel_mult', type=ast.literal_eval, default=[2, 4], help="Per-resolution multipliers for the number of channels")
    parser.add_argument('--channel_mult_emb', type=float, default=0.1, help="Multiplier for the embedding vector dimensionality")
    parser.add_argument('--implicit_mlp', action='store_true', help="enable implicit coordinate encoding")

    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--k1', type=int, default=128, help="Number of basis vectors for source mesh")
    parser.add_argument('--k2', type=int, default=128, help="Number of basis vectors for target mesh")
    parser.add_argument('--train_split', type=float, default=0.9, help="Train split ratio")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--real_p', type=float, default=0.5, help="Probability for real patches")
    parser.add_argument('--progressive', action='store_true', help="Enable progressive patch size")
    parser.add_argument('--duration', type=int, default=2, help="Training duration in thousands of images")
    parser.add_argument('--ema', type=float, default=0.5, help="EMA half-life")
    parser.add_argument('--bench', action='store_true', help="Enable cudnn benchmarking")
    return parser.parse_args()


def set_dict_fields(opts, model_save_path):
    c = {
        'dataset_kwargs': {'data_folder': opts['data'], 'k1': opts['k1'], 'k2': opts['k2'], 'device': opts['device'],
                           'train_split': opts['train_split'],
                           'batch_size': opts['batch_size'],
                           'external_test': opts['external_test'],
                           'external_test_data': opts['external_test_data'],
                           # 'test_data': opts['test_data']
                           },
        # 'data_loader_kwargs': {'pin_memory': True, 'num_workers': 4, 'prefetch_factor': 2},
        'network_kwargs': {'model_channels': opts['model_channels'], 'channel_mult': opts['channel_mult'],
                           'dropout': opts['dropout'],
                           'num_blocks': opts['num_blocks'], 'implicit_mlp': opts['implicit_mlp']},
        'loss_kwargs': {},
        'optimizer_kwargs': {'class_name': 'torch.optim.Adam', 'lr': opts['lr'], 'betas': [0.9, 0.999], 'eps': 1e-8},
        'real_p': opts['real_p'],
        'progressive': opts['progressive'],
        'total_kimg': max(int(opts['duration'] * 1000), 1),
        'ema_halflife_kimg': int(opts['ema'] * 1000),
        'batch_size': opts['batch_size'],
        'batch_gpu': None,
        'cudnn_benchmark': opts['bench'],
        'model_save_path': model_save_path,
        'n_patch_res': opts['n_patch_res'],
        'seed': opts["seed"],
        'resume_kwargs': {'save_resume_ckpt': opts["save_resume_ckpt"]},
        'guidance_kwargs': {'upsampling': opts["upsampling"], 'no_guidance': opts["no_guidance"], 'regularize_guidance': opts["regularize_guidance"]},
        'log_wb': opts["log_wb"],
        'fast_training': opts["fast_training"]
    }
    return c


def set_wandb(config_dict, project_name):
    """Establish W&B connection and ensure seamless resumption after preemption."""
    run = wandb.init(
        project=project_name,
        config=config_dict
    )
    return run


def hash_training_config(config):
    # Convert object attributes to a dictionary
    config_dict = vars(config).copy()  # Make a copy to avoid modifying the original object
    # Remove fields if exist
    config_dict.pop('job_num', None)
    config_dict.pop('evaluate', None)
    # config_dict.pop('log_wb', None)
    # config_dict.pop('upsampling', None)
    # config_dict.pop('no_guidance', None)
    # config_dict.pop('regularize_guidance', None)
    # config_dict.pop('save_resume_ckpt', None)
    # config_dict.pop('resume_from_checkpoint', None)

    # Convert dictionary to a sorted JSON string
    config_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Compute SHA-256 hash
    return hashlib.sha256(config_str.encode()).hexdigest()


def set_seed(seed=42):
    random.seed(seed)                   # Python random
    np.random.seed(seed)                 # NumPy
    torch.manual_seed(seed)              # PyTorch (CPU)
    torch.cuda.manual_seed(seed)         # PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)     # Multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable cuDNN benchmarking for reproducibility


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def load_network(network_pkl, device):
    """Load trained network from .pkl file"""
    print(f'Loading network from "{network_pkl}"...')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)
    net.eval()
    return net


def normalize_to_unit_sphere(X):
    """
    Normalizes 3D coordinates by centering them at the origin and scaling to fit within a unit sphere.
    Handles both NumPy arrays and PyTorch tensors without conversion.

    Parameters
    ----------
    X : np.ndarray of shape (N, 3) or torch.Tensor of shape (N, 3)
        3D point cloud coordinates.
    Returns
    -------
    X_normalized : np.ndarray or torch.Tensor of shape (N, 3)
        Normalized coordinates.
    """
    if isinstance(X, np.ndarray):
        # NumPy version
        centroid = np.mean(X, axis=0, keepdims=True)  # Shape (1, 3)
        X_centered = X - centroid
        max_dist = np.max(np.linalg.norm(X_centered, axis=1))
        X_normalized = X_centered / max_dist
    elif isinstance(X, torch.Tensor):
        # PyTorch version
        centroid = X.mean(dim=0, keepdim=True)  # Shape (1, 3)
        X_centered = X - centroid
        max_dist = torch.norm(X_centered, dim=1).max()
        X_normalized = X_centered / max_dist
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

    return X_normalized


def create_diagonal_sparse_tensor(values_vec, device):
    # Create indices for diagonal positions
    indices = torch.arange(values_vec.shape[0], device=device).repeat(2, 1)  # (2, N)
    # Create sparse diagonal matrix in COO format
    tensor_sparse = torch.sparse_coo_tensor(indices, values_vec, (values_vec.shape[0], values_vec.shape[0]),
                                            device=device, dtype=torch.float32)
    return tensor_sparse


def custom_collate_fn(batch):
    """
    Custom collation function to handle a mix of dense and sparse tensors in a batch.
    """
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        if batch[0].is_sparse:
            return batch  # Keep sparse tensors as a list (since PyTorch doesn't support stacking them)
        else:
            return torch.stack(batch, dim=0)  # Stack dense tensors
    elif isinstance(batch[0], dict):
        return {key: custom_collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (tuple, list)):
        return [custom_collate_fn(samples) for samples in zip(*batch)]
    return batch  # Default return for other types (scalars, strings, etc.)


def toNP(x):
    return x.detach().cpu().numpy()


from dataset.functional_map_dataset import FunctionalMapDataset
from dataset.infinite_sampler import InfiniteSampler
from torch.utils.data import DataLoader


def get_train_data(config):
    train_dataset = FunctionalMapDataset(config['data_folder'], config['k1'], config['k2'],
                                         "cpu",
                                         train=True,
                                         train_split=config['train_split'],
                                         with_bases=False,
                                         extended_data=False)
    sampler = InfiniteSampler(train_dataset, rank=0, num_replicas=1, seed=42)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,  # Use custom infinite sampler
        batch_size=config['batch_size']
    )
    return train_dataset, train_dataloader


def get_test_data(config):
    test_dataset = FunctionalMapDataset(config['data_folder'], config['k1'], config['k2'], config['device'],
                                        train=False,
                                        train_split=config['train_split'],
                                        # with_bases=True,
                                        extended_data=True,
                                        external_test=config['external_test'],
                                        external_test_data=config['external_test_data'])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_dataset, test_dataloader
