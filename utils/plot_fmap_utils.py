import matplotlib.pyplot as plt
import numpy as np
import torch
import os, sys
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
from utils.get_fmap_cmap import create_custom_colormap

cmap = create_custom_colormap('default')


def plot_fmaps_paper(fmap_gt, fmap_init, fmap_refined):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)  # Increase size & resolution
    # Determine the common color scale across all images
    vmin = min(fmap_init.min(), fmap_gt.min(), fmap_refined.min()).cpu().numpy()
    vmax = max(fmap_init.max(), fmap_gt.max(), fmap_refined.max()).cpu().numpy()
    # Plot images with shared vmin/vmax
    im0 = axes[0].imshow(fmap_init.cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    im1 = axes[1].imshow(fmap_refined.cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    im2 = axes[2].imshow(fmap_gt.cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    # Add one shared colorbar
    plt.tight_layout(pad=0.5)  # Reduce padding between subplots
    plt.subplots_adjust(left=0.05, right=1.09)  # Reduce space on left & right
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.8, aspect=30)
    # plt.show()


def plot_fmaps(fmap_gt, fmap_init, fmap_refined, name, i,
               loss_gt, loss_init, loss_refined, shared_colorbar=True, log_wb=False, plt_show=False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)  # Increase size & resolution
    font_s = 10
    e_mse = r"MSE"
    e_m = r"$\mathcal{E}^{\mathrm{m}}$"
    if shared_colorbar:
        # Determine the common color scale across all images
        vmin = min(fmap_init[i, 0].min(), fmap_gt[i, 0].min(), fmap_refined[i, 0].min()).cpu().numpy()
        vmax = max(fmap_init[i, 0].max(), fmap_gt[i, 0].max(), fmap_refined[i, 0].max()).cpu().numpy()
        # Plot images with shared vmin/vmax
        im0 = axes[0].imshow(fmap_init[i, 0].cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        im1 = axes[1].imshow(fmap_refined[i, 0].cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        im2 = axes[2].imshow(fmap_gt[i, 0].cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        if isinstance(loss_init, tuple):
            axes[0].set_title(f"Initial\n{e_mse}={loss_init[0]:.4f}, {e_m}={loss_init[1]:.4f}", fontsize=font_s)
            axes[1].set_title(f"Refined\n{e_mse}={loss_refined[0]:.4f}, {e_m}={loss_refined[1]:.4f}", fontsize=font_s)
            axes[2].set_title(f"Ground Truth\n{e_mse}={loss_gt[0]:.4f}, {e_m}={loss_gt[1]:.4f}", fontsize=font_s)
        else:
            axes[0].set_title(f"Initial\n{e_mse} = {loss_init:.6f}", fontsize=font_s)
            axes[1].set_title(f"Refined\n{e_mse} = {loss_refined:.6f}", fontsize=font_s)
            axes[2].set_title(f"Ground Truth\n{e_mse} = {loss_gt:.6f}", fontsize=font_s)
        # Add one shared colorbar
        plt.tight_layout(pad=0.5)  # Reduce padding between subplots
        plt.subplots_adjust(left=0.05, right=1.09)  # Reduce space on left & right
        cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.8, aspect=30)
        # cbar.set_label("Value")
    else:
        im0 = axes[0].imshow(fmap_init[i, 0].cpu().numpy(), cmap=cmap)
        im1 = axes[1].imshow(fmap_refined[i, 0].cpu().numpy(), cmap=cmap)
        im2 = axes[2].imshow(fmap_gt[i, 0].cpu().numpy(), cmap=cmap)
        if isinstance(loss_init, tuple):
            axes[0].set_title(f"Initial\n{e_mse}={loss_init[0]:.4f}, {e_m}={loss_init[1]:.4f}", fontsize=font_s)
            axes[1].set_title(f"Refined\n{e_mse}={loss_refined[0]:.4f}, {e_m}={loss_refined[1]:.4f}", fontsize=font_s)
            axes[2].set_title(f"Ground Truth\n{e_mse}={loss_gt[0]:.4f}, {e_m}={loss_gt[1]:.4f}", fontsize=font_s)
        else:
            axes[0].set_title(f"Initial\n{e_mse} = {loss_init:.6f}", fontsize=font_s)
            axes[1].set_title(f"Refined\n{e_mse} ={loss_refined:.6f}", fontsize=font_s)
            axes[2].set_title(f"Ground Truth\n{e_mse} = {loss_gt:.6f}", fontsize=font_s)
        plt.tight_layout(pad=0.5)  # Reduce padding between subplots
        plt.subplots_adjust(left=0.05, right=1.09)  # Reduce space on left & right
        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
        fig.colorbar(im2, ax=axes[2])
    if log_wb:
        wandb.log({f"Test sample_{name}, fmap": wandb.Image(fig, caption=f"Test sample_{i}, fmap")})
        plt.close(fig)
    if plt_show:
        plt.show()


def plot_fmaps_gt_diff(fmap_gt, fmap_init, fmap_refined, name, i, loss_init, loss_refined, shared_colorbar=False, log_wb=False, plt_show=False):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)  # Increase size & resolution
    font_s = 10
    # computed_gt_diff = ((fmap_computed[i, 0] - fmap_gt[i, 0])**2).cpu()
    # predicted_gt_diff = ((fmap_predicted[i, 0] - fmap_gt[i, 0]) ** 2).cpu()
    init_gt_diff = torch.abs(fmap_init[i, 0] - fmap_gt[i, 0]).cpu()
    refined_gt_diff = torch.abs(fmap_refined[i, 0] - fmap_gt[i, 0]).cpu()
    e_mse = r"MSE"
    e_m = r"$\mathcal{E}^{\mathrm{m}}$"
    if shared_colorbar:
        # Determine the common color scale across all images
        vmin = min(init_gt_diff.min(), refined_gt_diff.min()).numpy()
        vmax = max(init_gt_diff.max(), refined_gt_diff.max()).numpy()
        # Plot images with shared vmin/vmax
        im0 = axes[0].imshow(init_gt_diff.numpy(), cmap='binary', vmin=vmin, vmax=vmax)
        im1 = axes[1].imshow(refined_gt_diff.numpy(), cmap='binary', vmin=vmin, vmax=vmax)
        if isinstance(loss_init, tuple):
            axes[0].set_title(f"Init\n{e_mse}={loss_init[0]:.4f}, {e_m}={loss_init[1]:.4f}", fontsize=font_s)
            axes[1].set_title(f"Refined\n{e_mse}={loss_refined[0]:.4f}, {e_m}={loss_refined[1]:.4f}", fontsize=font_s)
        else:
            axes[0].set_title(f"Init\n{e_mse} = {loss_init:.6f}", fontsize=font_s)
            axes[1].set_title(f"Refined\n{e_mse} = {loss_refined:.6f}", fontsize=font_s)
        # Add one shared colorbar
        plt.tight_layout(pad=0.5)  # Reduce padding between subplots
        plt.subplots_adjust(left=0.08, right=0.96)  # Reduce space on left & right
        cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.8, aspect=30)
        # cbar.set_label("Value")
    else:
        im0 = axes[0].imshow(init_gt_diff.numpy(), cmap='binary')
        im1 = axes[1].imshow(refined_gt_diff.numpy(), cmap='binary')
        if isinstance(loss_init, tuple):
            axes[0].set_title(f"Init\n{e_mse}={loss_init[0]:.4f}, {e_m}={loss_init[1]:.4f}", fontsize=font_s)
            axes[1].set_title(f"Refined\n{e_mse}={loss_refined[0]:.4f}, {e_m}={loss_refined[1]:.4f}", fontsize=font_s)
        else:
            axes[0].set_title(f"Init\n{e_mse} = {loss_init:.6f}")
            axes[1].set_title(f"Refined\n{e_mse} = {loss_refined:.6f}")
        plt.tight_layout(pad=0.5)  # Reduce padding between subplots
        plt.subplots_adjust(left=0.08, right=0.96)  # Reduce space on left & right
        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
    if log_wb:
        wandb.log({f"Test sample_{name}, GT diff": wandb.Image(fig, caption=f"Test sample_{i}, GT diff")})
        plt.close(fig)
    if plt_show:
        plt.show()


def log_wandb_fmap(fmap_gt, fmap_init, fmap_refined, i=0, log_wb=False, name=0, visualize_per_batch=False):
    # mse loss
    loss_gt_mse = torch.mean((fmap_gt[i, 0] - fmap_gt[i, 0]) ** 2).item()
    loss_init_mse = torch.mean((fmap_init[i, 0] - fmap_gt[i, 0]) ** 2).item()
    loss_refined_mse = torch.mean((fmap_refined[i, 0] - fmap_gt[i, 0]) ** 2).item()
    loss_gt = loss_gt_mse
    loss_init = loss_init_mse
    loss_refined = loss_refined_mse

    plot_fmaps(fmap_gt, fmap_init, fmap_refined, name, i, loss_gt, loss_init, loss_refined, True,
               log_wb=log_wb, plt_show=visualize_per_batch)
    plot_fmaps_gt_diff(fmap_gt, fmap_init, fmap_refined, name, i, loss_init, loss_refined, True,
                       log_wb=log_wb, plt_show=visualize_per_batch)
    return loss_init, loss_refined, loss_gt
