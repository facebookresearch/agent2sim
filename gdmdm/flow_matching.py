import pdb
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class NoiseSchedulerFM(nn.Module):
    def __init__(self, num_timesteps=1000, beta_schedule="linear"):
        super().__init__()
        self.num_timesteps = num_timesteps


    def step(self, model_output, timestep, sample):
        """
        Step in the normalized space
        model_output: noise
        """
        delta = (model_output - sample) / (timestep+1)
        # delta = model_output / self.num_timesteps
        pred_prev_sample = sample + delta
        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        # t=0 -> clean
        # t=1 -> noise
        t_frac = (1+timesteps[:, None]) / self.num_timesteps # small val -> 1
        return (1 - t_frac) * x_start + t_frac * x_noise

    def sample_noise(self, clean, std, mean, max_t=None):
        """
        Sample noise in the normalized space and add it to the clean signal
        clean: unnormalized
        noise: normalized => GT in normlized space?
        noisy: unnormalized
        """
        shape = clean.shape
        noise = torch.randn(shape, device="cuda")
        clean = (clean - mean) / std
        timesteps = torch.randint(0, self.num_timesteps, (shape[0],), device="cuda")
        timesteps = timesteps.long()
        noisy = self.add_noise(clean, noise, timesteps)
        t_frac = timesteps[:, None] / self.num_timesteps
        noisy = noisy * std + mean
        return clean, noisy, t_frac
        # return clean - noise, noisy, t_frac

    def __len__(self):
        return self.num_timesteps
