# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

# Adapted from Patch Diffusion (https://github.com/Zhendong-Wang/Patch-Diffusion)
# which is based on EDM (https://github.com/NVlabs/edm)
# Licensed under MIT and NVIDIA Source Code Licenses respectively.

import torch
import torch.nn as nn
import torch.nn.functional as F

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".
# @persistence.persistent_class
class Patch_EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def pachify(self, images, patch_size, padding=None):
        device = images.device
        batch_size, resolution = images.size(0), images.size(2)

        if padding is not None:
            padded = torch.zeros((images.size(0), images.size(1), images.size(2) + padding * 2,
                                  images.size(3) + padding * 2), dtype=images.dtype, device=device)
            padded[:, :, padding:-padding, padding:-padding] = images
        else:
            padded = images

        h, w = padded.size(2), padded.size(3)
        th, tw = patch_size, patch_size
        if w == tw and h == th:
            i = torch.zeros((batch_size,), device=device).long()
            j = torch.zeros((batch_size,), device=device).long()
        else:
            i = torch.randint(0, h - th + 1, (batch_size,), device=device)
            j = torch.randint(0, w - tw + 1, (batch_size,), device=device)

        rows = torch.arange(th, dtype=torch.long, device=device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long, device=device) + j[:, None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:, torch.arange(batch_size)[:, None, None], rows[:, torch.arange(th)[:, None]],
                 columns[:, None]]
        padded = padded.permute(1, 0, 2, 3)

        x_pos = torch.arange(tw, dtype=torch.long, device=device).unsqueeze(0).repeat(th, 1).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        y_pos = torch.arange(th, dtype=torch.long, device=device).unsqueeze(1).repeat(1, tw).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        x_pos = x_pos + j.view(-1, 1, 1, 1)
        y_pos = y_pos + i.view(-1, 1, 1, 1)
        x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
        y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
        images_pos = torch.cat((x_pos, y_pos), dim=1)

        return padded, images_pos

    def __call__(self, net, images, patch_size, resolution, labels=None, augment_pipe=None):
        images, images_pos = self.pachify(images, patch_size)

        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        yn = y + n

        D_yn = net(yn, sigma, x_pos=images_pos, class_labels=labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Version with image condition added
# @persistence.persistent_class
class Patch_ImgCond_EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        """
        Initialize the EDM loss function with parameters for noise sampling.

        Parameters:
        ----------
        P_mean : float, default=-1.2 (assumes sigma_min=0.002, sigma_max = 80)
            The mean of the log-normal distribution used to sample noise levels (σ).
            Controls the center of the noise distribution.
            0.5 * log(sigma_min * sigma_max)

        P_std : float, default=1.2 (assumes sigma_min=0.002, sigma_max = 80)
            The standard deviation of the log-normal distribution for noise levels.
            Affects how widely noise levels are spread.
            0.5 * log(sigma_max / sigma_min)

        sigma_data : float, default=0.5
            Represents the expected standard deviation of the real dataset.
            Used in the loss function to balance noise weighting.

        Notes:
        ------
        - The log-normal distribution is used to ensure a wide range of noise values.
        - `sigma_data` helps normalize noise scales and stabilize training.
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def pachify(self, images, images_cond, patch_size, padding=None):
        device = images.device
        batch_size, resolution = images.size(0), images.size(2)

        if padding is not None:
            padded = torch.zeros((images.size(0), images.size(1), images.size(2) + padding * 2,
                                  images.size(3) + padding * 2), dtype=images.dtype, device=device)
            padded[:, :, padding:-padding, padding:-padding] = images

            padded_cond = torch.zeros((images_cond.size(0), images_cond.size(1), images_cond.size(2) + padding * 2,
                                       images_cond.size(3) + padding * 2), dtype=images_cond.dtype, device=device)
            padded_cond[:, :, padding:-padding, padding:-padding] = images_cond
        else:
            padded = images
            padded_cond = images_cond

        h, w = padded.size(2), padded.size(3)
        th, tw = patch_size, patch_size
        if w == tw and h == th:
            i = torch.zeros((batch_size,), device=device).long()
            j = torch.zeros((batch_size,), device=device).long()
        else:
            i = torch.randint(0, h - th + 1, (batch_size,), device=device)
            j = torch.randint(0, w - tw + 1, (batch_size,), device=device)

        rows = torch.arange(th, dtype=torch.long, device=device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long, device=device) + j[:, None]

        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:, torch.arange(batch_size)[:, None, None], rows[:, torch.arange(th)[:, None]],
                 columns[:, None]]
        padded = padded.permute(1, 0, 2, 3)

        padded_cond = padded_cond.permute(1, 0, 2, 3)
        padded_cond = padded_cond[:, torch.arange(batch_size)[:, None, None], rows[:, torch.arange(th)[:, None]],
                      columns[:, None]]
        padded_cond = padded_cond.permute(1, 0, 2, 3)

        x_pos = torch.arange(tw, dtype=torch.long, device=device).unsqueeze(0).repeat(th, 1).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        y_pos = torch.arange(th, dtype=torch.long, device=device).unsqueeze(1).repeat(1, tw).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        x_pos = x_pos + j.view(-1, 1, 1, 1)
        y_pos = y_pos + i.view(-1, 1, 1, 1)
        x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
        y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
        images_pos = torch.cat((x_pos, y_pos), dim=1)

        return padded, padded_cond, images_pos

    def __call__(self, net, images, images_cond, patch_size, resolution, labels=None, augment_pipe=None):
        images, images_cond, images_pos = self.pachify(images, images_cond, patch_size)

        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        yn = y + n  # xt = x0 + sigma_t I

        D_yn = net(yn, sigma, x_pos=images_pos, x_img_cond=images_cond, class_labels=labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

