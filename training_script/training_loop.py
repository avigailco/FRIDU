# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
"""Main training loop."""

# Adapted from Patch Diffusion (https://github.com/Zhendong-Wang/Patch-Diffusion)
# which is based on EDM (https://github.com/NVlabs/edm)
# Licensed under MIT and NVIDIA Source Code Licenses respectively.

import torch
import numpy as np
import time
import copy
import sys
import wandb
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

from models.patches_precond import Patch_ImgCond_EDMPrecond
from loss.patch_loss import Patch_ImgCond_EDMLoss
from utils.get_fmap_cmap import create_custom_colormap
from utils.utils import set_wandb, hash_training_config, set_dict_fields, parse_args
from utils.utils import get_train_data, get_test_data
from training_script.generate import generate

cmap = create_custom_colormap('default')


def training_loop(
        fast_training=False,
        dataset_kwargs={},  # Options for training set.
        network_kwargs={},  # Options for model and preconditioning.
        loss_kwargs={},  # Options for loss function.
        optimizer_kwargs={},  # Options for optimizer.
        seed=42,  # Global random seed.
        batch_size=512,      # Total batch size for one training iteration.
        batch_gpu=None,  # Limit batch size per GPU, None = no limit.
        total_kimg=200000,  # Training duration, measured in thousands of training images.
        ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
        ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
        lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
        loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
        kimg_per_tick=50,  # Interval of progress prints.
        cudnn_benchmark=True,     # Enable torch.backends.cudnn.benchmark.
        real_p=0.5,  # the ratio of full size image used in the training.
        progressive=False,
        device=torch.device('cuda'),
        log_wb=False,
        model_save_path='',
        n_patch_res=2,
        resume_kwargs={},
        guidance_kwargs={},
):

    start_time = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)  # np.random.randint(1 << 31)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // 1
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * 1

    # Load dataset.
    # if denoise_test:
    print('Loading test dataset...')
    test_dataset, test_dataloader = get_test_data(dataset_kwargs)

    print('Loading train dataset...')
    train_dataset, train_dataloader = get_train_data(dataset_kwargs)
    train_dataloader_iter = iter(train_dataloader)

    img_resolution = dataset_kwargs['k1']
    img_channels = 1  #
    net_input_channels = img_channels + 1 + 2   # coordinates + f_wks

    if img_resolution == 512:
        batch_mul_dict = {512: 1, 256: 2, 128: 4, 64: 16, 32: 32, 16: 64}
    elif img_resolution == 128:
        batch_mul_dict = {128: 1, 64: 2, 32: 4, 16: 8, 8: 16, 4: 32}
    elif img_resolution == 130:
        batch_mul_dict = {130: 1, 64: 2, 32: 4, 16: 8, 8: 16, 4: 32}
    elif img_resolution == 64:
        batch_mul_dict = {64: 1, 32: 2, 16: 4, 8: 8, 4: 16}

    # Number of patch resolutions
    def closest_power_of_two(x):
        return 1 << round(np.log2(x))
    if n_patch_res == 1:
        real_p = 1
        p_list = np.array([real_p])  # probability of selecting different patch sizes
        patch_list = [img_resolution]  # possible patch sizes
        batch_mul_avg = np.sum(p_list * np.array([1]))  # how many patches are processed per image
    elif n_patch_res == 2:
        p_list = np.array([(1 - real_p), real_p])   # probability of selecting different patch sizes
        patch_list = [closest_power_of_two(img_resolution // 2), img_resolution]  # possible patch sizes
        batch_mul_avg = np.sum(p_list * np.array([2, 1]))   # how many patches are processed per image
    elif n_patch_res == 3:
        p_list = np.array([(1-real_p)*2/5, (1-real_p)*3/5, real_p])
        patch_list = [closest_power_of_two(img_resolution // 4), closest_power_of_two(img_resolution // 2), img_resolution]
        batch_mul_avg = np.sum(np.array(p_list) * np.array([4, 2, 1]))

    # Construct network.
    print('Constructing network...')
    net = Patch_ImgCond_EDMPrecond(img_resolution=img_resolution,
                                   img_channels=net_input_channels,
                                   out_channels=img_channels,
                                   **network_kwargs).to(device)
    net.train().requires_grad_(True)

    # Setup optimizer.
    print('Setting up optimizer...')
    optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_kwargs.get('lr', 1e-4), betas=(0.9, 0.999), eps=1e-8)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    loss_fn = Patch_ImgCond_EDMLoss(**loss_kwargs)
    # ------------------------------------------------------------------------------------
    # Load checkpoint if specified
    if resume_kwargs['resume_from_checkpoint']:
        try:
            print(f"Loading checkpoint: {resume_kwargs['resume_checkpoint_path']}")
            checkpoint = torch.load(resume_kwargs['resume_checkpoint_path'], map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            ema.load_state_dict(checkpoint['ema_state_dict'])
            cur_nimg = checkpoint.get('cur_nimg', 0)  # Default to 0 if key missing
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Error loading checkpoint {resume_kwargs['resume_checkpoint_path']}: {e}")
            print("Starting training from scratch instead.")
            cur_nimg = 0  # Ensure training starts fresh if loading fails
    else:
        print("Starting training from scratch.")
        cur_nimg = 0  # Start from beginning
    # ------------------------------------------------------------------------------------
    print(f'Training for {total_kimg} kimg...')
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    generate_epoch = 1000  # Generate and save resume_ckpt after each generate_epoch iterations
    checkpoint_epoch = 100000  # Save model checkpoint after each checkpoint_epoch iterations
    while True:
        optimizer.zero_grad()
        for round_idx in range(num_accumulation_rounds):
            if progressive:   # patch sizes increase gradually
                p_cumsum = p_list.cumsum()
                p_cumsum[-1] = 10.
                prog_mask = (cur_nimg // 1000 / total_kimg) <= p_cumsum
                patch_size = int(patch_list[prog_mask][0])
                batch_mul_avg = batch_mul_dict[patch_size] // batch_mul_dict[img_resolution]
            else:   # patch sizes are chosen randomly
                patch_size = int(np.random.choice(patch_list, p=p_list))

            batch_mul = batch_mul_dict[patch_size] // batch_mul_dict[img_resolution]  # number of patches used
            fmap_gt, fmap_computed = [], []
            for _ in range(batch_mul):
                fmap_gt_, fmap_computed_ = next(train_dataloader_iter)  # Fetch next batch
                fmap_gt.append(fmap_gt_), fmap_computed.append(fmap_computed_)
            fmap_gt, fmap_computed = torch.cat(fmap_gt, dim=0), torch.cat(fmap_computed, dim=0)
            del fmap_gt_, fmap_computed_
            fmap_gt, fmap_computed = fmap_gt.unsqueeze(1), fmap_computed.unsqueeze(1)
            fmap_gt, fmap_computed = fmap_gt.to(dataset_kwargs['device']), fmap_computed.to(dataset_kwargs['device'])

            loss = loss_fn(net, fmap_gt, fmap_computed, patch_size, img_resolution)
            # ensure that the gradient updates are balanced regardless of patch size
            inter_loss = loss.sum().mul(loss_scaling / batch_gpu_total / batch_mul)
            if log_wb:
                wandb.log({"train_loss": inter_loss.item()})
            inter_loss.backward()

        # Update weights (ensures smooth convergence as more patches are seen).
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA (update rate depends on the patch size).
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if torch.cuda.device_count() > 1 and ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size * batch_mul_avg / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += int(batch_size * batch_mul_avg)
        epoch = (cur_nimg // int(batch_size * batch_mul_avg))
        if (not fast_training) and epoch % generate_epoch == 0:
            generate(net, test_dataset, resolution=img_resolution, device=device, guidance_kwargs=guidance_kwargs, log_wb=log_wb)
            net.train()  # Set back to training mode
            if resume_kwargs['save_resume_ckpt']:  # Save resume checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'cur_nimg': cur_nimg
                }
                torch.save(checkpoint, resume_kwargs["resume_checkpoint_path"])
                print(f"Checkpoint for resume saved at {resume_kwargs['resume_checkpoint_path']}")
            torch.cuda.empty_cache()

        if epoch % checkpoint_epoch == 0:  # Save checkpoint
            torch.save(net.state_dict(), os.path.join(model_save_path, f"ckpt_{epoch}.pth"))
            print(f"Save checkpoint, Model saved as 'ckpt_{epoch}.pth'")

        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
        tick_end_time = time.time()
        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Save final model
    torch.save(net.state_dict(), os.path.join(model_save_path, "ckpt_final_model.pth"))
    print("Training complete. Model saved as 'ckpt_final_model.pth'")


if __name__ == "__main__":
    args = parse_args()
    opts = vars(args)
    config_hash = hash_training_config(args)
    model_save_path = f"../model_checkpoints/{config_hash}"
    print(f"Train Data: {opts['data']}")
    c = set_dict_fields(opts, model_save_path)

    os.makedirs(model_save_path, exist_ok=True)
    resume_checkpoint_path = os.path.join(model_save_path, f"resume_checkpoint.pth")
    resume_from_checkpoint = args.resume_from_checkpoint and os.path.exists(resume_checkpoint_path)
    c['resume_kwargs'].update({'resume_checkpoint_path': resume_checkpoint_path,
                               'resume_from_checkpoint': resume_from_checkpoint})

    # Set logging
    if opts["log_wb"]:
        print("Setting W&B...")
        wandb_run = set_wandb(opts, "FRIDU")
        wandb.config.update({"config_hash": config_hash})

    # Start training
    training_loop(**c)
    if opts["log_wb"]:
        wandb.finish()
