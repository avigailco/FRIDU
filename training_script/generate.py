# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

# Adapted from Patch Diffusion (https://github.com/Zhendong-Wang/Patch-Diffusion)
# which is based on EDM (https://github.com/NVlabs/edm)
# Licensed under MIT and NVIDIA Source Code Licenses respectively.

import torch
import numpy as np
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

from models.patches_precond import Patch_ImgCond_EDMPrecond
from utils.get_fmap_cmap import create_custom_colormap
from utils.utils import hash_training_config, custom_collate_fn
from utils.utils import get_test_data
from utils.mesh_visualization import threeD_plot_mapped_func
from utils.plot_fmap_utils import log_wandb_fmap
from utils.fmap_metrics import plot_euclidean_err_comparison, get_pair_cdf
from utils.fmap_metrics import compute_vertex_to_vertex_map_batched
from utils.fmap_metrics import plot_err_graph_over_dataset
from utils.fmap_metrics import compute_point2point_guidance, lap_bij_resolvant, orthogonality_s_loss
from utils.utils import parse_args, set_dict_fields

cmap = create_custom_colormap('default')


def sample_with_cfg(net, x, t, pos, f_wks, class_labels, cfg=1.3):
    """Performs classifier-free guidance (CFG) sampling if needed."""
    if cfg is None or cfg == 1.0:
        eps = net(x, t, pos, f_wks, class_labels).to(torch.float64)
    else:
        x_combined = torch.cat((x, x), dim=0)
        pos_combined = torch.cat((pos, pos), dim=0)
        f_wks_combined = torch.cat((torch.zeros_like(f_wks), f_wks), dim=0)
        class_combined = torch.cat((torch.zeros_like(class_labels).long(), class_labels), dim=0)

        uncond_eps, cond_eps = net(x_combined, t, pos_combined, f_wks_combined, class_combined).to(torch.float64).chunk(2, dim=0)
        eps = uncond_eps + cfg * (cond_eps - uncond_eps)
    return eps


def edm_sampler(
        net, latents, latents_pos, latents_img_cond, mask_pos, class_labels=None, cfg=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
    """
    Performs unconditional or classifier-free guided denoising using the
    Elucidated Diffusion Model (EDM) sampling scheme.

    Args:
        net (torch.nn.Module): Trained diffusion network.
        latents (Tensor): Input noise tensor for diffusion sampling.
        latents_pos (Tensor): Positional encodings for conditioning.
        latents_img_cond (Tensor): Conditioning tensor (initial fmap).
        mask_pos (bool): Whether to use positional mask conditioning.
        class_labels (Tensor, optional): Class labels for conditional sampling.
        cfg (float, optional): CFG scale for guidance; 1.0 disables it.
        randn_like (callable, optional): Noise generator function. Default: torch.randn_like.
        num_steps (int, optional): Number of diffusion time steps. Default: 18.
        sigma_min, sigma_max (float, optional): Minimum/maximum noise levels.
        rho (float, optional): Power for noise schedule interpolation.
        S_churn, S_min, S_max, S_noise (float, optional): Noise perturbation params.
    """
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    # t_steps: decreasing noise levels for the diffusion process
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    x_next = latents.to(torch.float64) * t_steps[0]  # initialize the sample x_t
    # Iterates through the noise schedule in reverse orde
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        # Apply Stochastic Churn (Optional Noise Perturbation)
        # Perturbs the sample to add controlled randomness (stochastic churn).
        # Helps explore different modes of the generative model.
        # This ensures that the diffusion process doesn’t always take the exact same trajectory.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Compute Euler Update Step
        if mask_pos:
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        else:
            # Classifier-Free Guidance (sample_with_cfg) for text/image conditioning.
            denoised = sample_with_cfg(net, x_hat, t_hat, latents_pos, latents_img_cond, class_labels, cfg)
        d_cur = (x_hat - denoised) / t_hat  # predicted noise?
        x_next = x_hat + (t_next - t_hat) * d_cur  # updates the sample

        # Apply 2nd order correction (improves stability).
        # refines d_t using a second estimate.
        if i < num_steps - 1:
            if mask_pos:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
            else:
                denoised = sample_with_cfg(net, x_hat, t_hat, latents_pos, latents_img_cond, class_labels, cfg)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def edm_sampler_guidance(
        net, latents, latents_pos, latents_img_cond, mask_pos,
        Lambda1, Lambda2, psi1, psi2, mass2,
        upsampling=False,
        class_labels=None, cfg=None, randn_like=torch.randn_like,
        num_steps=18, k=2, m=3, guidance_strength=0.1,  # k = self-recurrence, m = backward steps
        sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, regularize_guidance=0):
    """
    Denoises latent functional maps using an Elucidated Diffusion Model (EDM) with
    universal guidance (forward, backward, and self-recurrent refinement).
    Algorithm 1: Universal Guidance.
     - Forward Guidance (Equation 6) modifies noise prediction.
     - Backward Guidance (Equation 9) refines denoised images (if m > 0).
     - Self-Recurrence (k steps) refines the prediction per timestep.

    Args:
        net (torch.nn.Module): Trained diffusion network.
        latents (Tensor): Initial noise tensor for sampling.
        latents_pos (Tensor): Positional encodings for each pixel.
        latents_img_cond (Tensor): Conditioning tensor (initial noisy functional map).
        mask_pos (bool): If True, positional mask is applied.
        Lambda1, Lambda2 (Tensor): Eigenvalues of source and target mesh Laplacians.
        psi1, psi2 (Tensor): Eigenbases of source and target mesh Laplacians.
        mass2 (Tensor): Target mesh mass matrix (diagonal or sparse tensor).
        upsampling (bool, optional): Enables progressive increasing of functional map dimension.
        class_labels (Tensor, optional): Optional class labels for conditional sampling.
        cfg (float, optional): Classifier-free guidance scale. Default: None.
        randn_like (callable, optional): Noise generator function. Default: torch.randn_like.
        num_steps (int, optional): Number of diffusion denoising steps. Default: 18.
        k (int, optional): Self-recurrence refinement iterations. Default: 2.
        m (int, optional): Backward guidance optimization steps. Default: 3.
        guidance_strength (float, optional): Strength of forward guidance. Default: 0.1.
        sigma_min, sigma_max (float, optional): Min/max noise levels. Default: 0.002, 80.
        rho (float, optional): Noise scheduling exponent. Default: 7.
        S_churn, S_min, S_max, S_noise (float, optional): EDM stochastic churn parameters.
        regularize_guidance (int, optional): Adds orthogonality/laplacian guidance constraints.
    """
    # --------------------------------------------------
    # Compute noise schedule
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    # t_steps: decreasing noise levels for the diffusion process
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
    # --------------------------------------------------
    #  Initialize x_T
    x_next = latents.to(torch.float64) * t_steps[0]
    # --------------------------------------------------
    if upsampling:
        # upsampling parameters
        k_init, k_final = 20, 128  # init and final fmap dimension
        curr_k = k_init
        upsamp_iter, upsamp_step = 8, 14  # after each upsamp_iter increase curr_k by upsamp_step
    else:
        curr_k = psi1.shape[2]
    # --------------------------------------------------
    # Iterate through time steps in reverse order
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        # Compute gamma for noise schedule adjustment
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        # --------------------------------------------------
        # **Self-recurrence loop (k steps)**
        for n in range(k):
            with ((torch.enable_grad())):  # Enable gradients only for guidance computation
                # **Step 1: Forward Guidance (Equation 6)**
                # Compute predicted denoised output
                if mask_pos:
                    denoised = net(x_hat, t_hat, latents_pos, latents_img_cond).to(torch.float64)
                else:
                    # Classifier-Free Guidance (sample_with_cfg) for text/image conditioning.
                    denoised = sample_with_cfg(net, x_hat, t_hat, latents_pos, latents_img_cond, class_labels, cfg)

                C_pred = denoised  # Assuming denoised output is a functional map
                C_pred.requires_grad_(True)
                with torch.no_grad():
                    vertex_map, _ = compute_vertex_to_vertex_map_batched(C_pred[0, 0][:curr_k, :curr_k].to(psi1.dtype),
                                                                         psi2[0][:, :curr_k],
                                                                         psi1[0][:, :curr_k])
                # Compute and normalize the gradient of the guidance loss
                p2p_guidance = torch.autograd.grad(compute_point2point_guidance(C_pred, mass2, psi1, psi2, vertex_map=vertex_map), C_pred, retain_graph=True)[0]
                p2p_guidance = p2p_guidance / (p2p_guidance.norm())
                guidance_grad = p2p_guidance

                # add task-dependent guidance to the p2p guidance
                if regularize_guidance != 0:
                    w_lap_bij, w_ortho = 1, 1  # additional guidance weight
                    if regularize_guidance == 1 or regularize_guidance == 3:  # include orthogonality
                        guidance_ortho = torch.autograd.grad(orthogonality_s_loss(C_pred), C_pred, retain_graph=True)[0]
                        guidance_ortho = guidance_ortho / (guidance_ortho.norm())
                    if regularize_guidance == 2 or regularize_guidance == 3:  # include laplacian commutativity
                        guidance_lap_bij = torch.autograd.grad(lap_bij_resolvant(C_pred, Lambda1, Lambda2), C_pred, retain_graph=True)[0]
                        guidance_lap_bij = guidance_lap_bij / (guidance_lap_bij.norm())

                    if regularize_guidance == 1:
                        guidance_grad = p2p_guidance + w_ortho*guidance_ortho
                    elif regularize_guidance == 2:
                        guidance_grad = p2p_guidance + w_lap_bij * guidance_lap_bij
                    elif regularize_guidance == 3:
                        guidance_grad = p2p_guidance + w_ortho * guidance_ortho + w_lap_bij * guidance_lap_bij

                # Apply Forward Guidance (Equation 6)
                predicted_noise = (x_hat - denoised) / t_hat
                s_t = ((t_hat ** 2) / (1 + t_hat ** 2)).sqrt()  # t_hat
                d_cur = predicted_noise + guidance_strength * s_t * guidance_grad
                # --------------------------------------------------
                # **Step 2: Backward Guidance (Equation 9)**
                if m > 0:
                    delta_x0 = torch.zeros_like(C_pred, requires_grad=True)  # Ensure requires_grad=True
                    optimizer = torch.optim.SGD([delta_x0], lr=1)  # Define an optimizer
                    for _ in range(m):  # Perform m steps of gradient descent
                        optimizer.zero_grad()  # Zero gradients before backpropagation
                        with torch.no_grad():
                            vertex_map, _ = compute_vertex_to_vertex_map_batched(C_pred[0, 0][:curr_k, :curr_k].to(psi1.dtype),
                                                                                 psi2[0][:, :curr_k],
                                                                                 psi1[0][:, :curr_k])

                        p2p_guidance_backward = compute_point2point_guidance(C_pred, mass2, psi1, psi2, vertex_map=vertex_map)
                        loss = p2p_guidance_backward
                        # add task-dependent guidance to the p2p guidance
                        if regularize_guidance != 0:
                            if regularize_guidance == 1 or regularize_guidance == 3:
                                loss_ortho = orthogonality_s_loss(C_pred)
                                added_guidance_loss1 = w_ortho * loss_ortho
                                scale1 = loss.item() / (added_guidance_loss1.item())
                            if regularize_guidance == 2 or regularize_guidance == 3:
                                loss_lap_bij = lap_bij_resolvant(C_pred, Lambda1, Lambda2)
                                added_guidance_loss2 = w_lap_bij * loss_lap_bij
                                scale2 = loss.item() / (added_guidance_loss2.item())
                            if regularize_guidance == 1:
                                loss_scaled = loss + scale1 * added_guidance_loss1
                            elif regularize_guidance == 2:
                                loss_scaled = loss + scale2 * added_guidance_loss2
                            elif regularize_guidance == 3:
                                loss_scaled = loss + scale1 * added_guidance_loss1 + scale2 * added_guidance_loss2
                            loss = loss_scaled
                        loss.backward(retain_graph=True)  # Compute gradients
                        optimizer.step()  # Gradient descent step
                    # Update noise prediction using backward guidance
                    d_cur = d_cur - delta_x0
            # **Step 3: Diffusion Update (Euler Step)**
            x_next = x_hat + (t_next - t_hat) * d_cur  # updates the sample
            # --------------------------------------------------
            # **Step 4: Second-Order Correction (for stability)**
            if i < num_steps - 1:
                if mask_pos:
                    denoised = net(x_next, t_next, class_labels).to(torch.float64)
                else:
                    denoised = sample_with_cfg(net, x_next, t_next, latents_pos, latents_img_cond, class_labels, cfg)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            # --------------------------------------------------
            # Self-recurrence
            x_hat = x_next + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_next)
            # --------------------------------------------------
        # # --------------------------------------------------
        if upsampling:
            if i != 0 and (i % upsamp_iter) == 0:
                curr_k += upsamp_step
                curr_k = min(curr_k, k_final)
    return x_next


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def generate(net, dataset, resolution=128, device="cuda", guidance_kwargs={},
             evaluate=False, log_wb=False, batch_size=1, seperate_char="\\", refine_iter=1):
    """
    Generates refined (denoised) functional maps from noised inputs using FRIDU's
    diffusion model, optionally with geometric guidance and evaluation.

    Args:
        net (torch.nn.Module): Trained FRIDU diffusion model.
        dataset (Dataset): Dataset providing functional maps and geometry.
        resolution (int, optional): Map resolution (basis dimension). Default: 128.
        device (str, optional): Device to run inference on. Default: "cuda".
        guidance_kwargs (dict, optional): Parameters controlling guidance (e.g., upsampling, regularization).
        evaluate (bool, optional): If True, computes and plots reconstruction metrics. Default: False.
        log_wb (bool, optional): Enables W&B logging. Default: False.
        batch_size (int, optional): Number of samples per batch. Default: 1.
        seperate_char (str, optional): Path separator for saving results. Default: "\\".
        refine_iter (int, optional): Number of iterative refinement passes. Default: 1.
    """
    net.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False)
    # ----------------------------------------------------------
    refined_err_lst, initial_err_lst = [], []
    inference_time = []
    visualize_per_batch = False  # show visualizations per batch (guidance loss, 3d mapping, distance err)
    # ----------------------------------------------------------
    with ((torch.no_grad())):
        for batch_idx, (source_basis, target_basis, source_eigvals, target_eigvals, fmap_gt, fmap_init, p2p,
                        vertices1, vertices2, faces1, faces2, mass1, mass2, L1, L2) in enumerate(tqdm(dataloader, total=1)):
            fmap_gt = fmap_gt[:, :resolution, :resolution].unsqueeze(1)  # Shape: [batch, 1, H, W]
            fmap_init = fmap_init[:, :resolution, :resolution].unsqueeze(1)
            mass1, mass2 = mass1[0].unsqueeze(0), mass2[0].unsqueeze(0)
            # Initialize batch_seeds for random latent generation
            batch_seeds = torch.randint(0, 2**32, (batch_size,), dtype=torch.int64).tolist()
            denoise_cond = [fmap_init]
            for iter in range(refine_iter):
                rnd = StackedRandomGenerator(device, batch_seeds)
                x_start, y_start = 0, 0
                image_size, image_channel = resolution, 1
                x_pos = torch.arange(x_start, x_start+image_size).view(1, -1).repeat(image_size, 1)
                y_pos = torch.arange(y_start, y_start+image_size).view(-1, 1).repeat(1, image_size)
                x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
                y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
                latents_pos = torch.stack([x_pos, y_pos], dim=0).to(device)
                latents_pos = latents_pos.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                latents = rnd.randn([batch_size, image_channel, image_size, image_size], device=device)

                # Generate denoised fmaps using EDM sampler
                if guidance_kwargs["no_guidance"]:
                    denoised_maps = edm_sampler(net, latents, latents_pos, denoise_cond[-1],
                                                mask_pos=False, num_steps=50).to(dtype=torch.float32)
                else:
                    m, k = 2, 5
                    cross_evaluation = False  # test on data from another dataset, e.g., FAUST
                    guidance_strength = 2000 if cross_evaluation else 500
                    start = time.time()
                    denoised_maps = edm_sampler_guidance(
                        net, latents, latents_pos, denoise_cond[-1].to(torch.float32),
                        mask_pos=False, num_steps=50,
                        Lambda1=source_eigvals, Lambda2=target_eigvals,
                        psi1=source_basis, psi2=target_basis, mass2=mass2,
                        upsampling=guidance_kwargs["upsampling"],
                        m=m, k=k, guidance_strength=guidance_strength,
                        regularize_guidance=guidance_kwargs["regularize_guidance"]).to(dtype=torch.float32)
                    end = time.time()
                inference_time.append(end-start)
                denoise_cond.append(denoised_maps)
            # --------------------------------------------------------------------------------
            if evaluate:
                # compute p2p (gt, init, refined)
                _, P21_gt = compute_vertex_to_vertex_map_batched(fmap_gt[0, 0], target_basis[0], source_basis[0])
                _, P21_init = compute_vertex_to_vertex_map_batched(fmap_init[0, 0], target_basis[0], source_basis[0])
                denoised_maps = denoise_cond[-1]
                _, P21_refined = compute_vertex_to_vertex_map_batched(denoised_maps[0, 0], target_basis[0], source_basis[0])
                if visualize_per_batch:
                    # 3d visualization of function mapping
                    f1 = vertices1.sum(dim=2).T  # function to map
                    mapped_f1_gt = P21_gt @ f1
                    mapped_f1_refined = P21_refined @ f1
                    mapped_f1_init = P21_init @ f1
                    threeD_plot_mapped_func(vertices1, faces1, vertices2, faces2, f1, mapped_f1_gt, mapped_f1_refined, mapped_f1_init, space=150)
                    # -----------------------------------------------------------------------------------------
                    # plot euclidean distance error graph
                    plot_euclidean_err_comparison(P21_gt, P21_refined, P21_init, vertices1[0])
                    # -----------------------------------------------------------------------------------------
                refined_err, initial_err = get_pair_cdf(P21_gt, P21_refined, P21_init, vertices1[0])
                refined_err_lst.append(refined_err)
                initial_err_lst.append(initial_err)
                # -----------------------------------------------------------------------------------------
            else:   # compute and log error
                loss_computed, loss_predicted, loss_gt = log_wandb_fmap(fmap_gt, fmap_init, denoised_maps,
                                                                        log_wb=log_wb, name=batch_idx,
                                                                        visualize_per_batch=visualize_per_batch)
            torch.cuda.empty_cache()
            # --------------------------------------------------------------------------------
            if not evaluate and batch_idx == 3:  # only generate 2 batches when called during training
                break

    if evaluate:
        plot_err_graph_over_dataset(refined_err_lst, initial_err_lst, x_label_str="Normalized Euclidean Error")


if __name__ == "__main__":
    args = parse_args()
    opts = vars(args)

    config_hash = hash_training_config(args)
    # config_hash = "michael_pretrained_model"  # michael wks based model, uncomment this to use our pretrained model
    model_save_path = f"../model_checkpoints/{config_hash}"
    print(f"Train Data: {opts['data']}")
    c = set_dict_fields(opts, model_save_path)

    test_dataset, _ = get_test_data(c["dataset_kwargs"])
    device = opts['device']
    img_resolution = c["dataset_kwargs"]['k1']
    img_channels = 1  # noised gt \ noise
    net_input_channels = img_channels + 1 + 2  # coordinates + f_wks
    print('Constructing network...')
    net = Patch_ImgCond_EDMPrecond(img_resolution=img_resolution,
                                   img_channels=net_input_channels,
                                   out_channels=img_channels,
                                   **c["network_kwargs"]).to(device)
    net.eval().requires_grad_(False)
    state_dict = torch.load(os.path.join(model_save_path, "ckpt_final_model.pth"), map_location=device)
    net.load_state_dict(state_dict)
    generate(net, test_dataset, resolution=img_resolution, device=device, guidance_kwargs=c["guidance_kwargs"],
             evaluate=True, log_wb=opts["log_wb"], seperate_char="\\")
