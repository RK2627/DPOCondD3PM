"""Implements discrete diffusion noise transitions and schedules."""
import torch
import torch.nn.functional as F
import math


class AbsorbingMaskScheduler:


    def __init__(self, num_steps=200, mask_token_id=4, device="cuda:0"):
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.mask_token_id = mask_token_id


        t = torch.arange(0, num_steps, device=self.device, dtype=torch.float32)
        s = 0.008
        alphas_bar = torch.cos(((t / num_steps + s) / (1 + s)) * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        self.alphas = alphas_bar[1:] / alphas_bar[:-1]
        self.alphas_bar = alphas_bar


    def mask_rate(self, t):

        if isinstance(t, torch.Tensor):
            return 1.0 - self.alphas_bar[t].to(self.device)
        else:
            return 1.0 - self.alphas_bar[int(t)].to(self.device)

    @torch.no_grad()
    def q_sample(self, x0, t):

        B, L = x0.shape
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device).long()

        mask_prob = self.mask_rate(t).view(-1, 1)
        keep_mask = (torch.rand(B, L, device=self.device) > mask_prob).long()
        x_t = x0 * keep_mask + self.mask_token_id * (1 - keep_mask)
        return x_t


    @torch.no_grad()
    def sample_update_mask(self, x_t, t):


        B, L = x_t.shape
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device).long()


        mask_prob = self.mask_rate(t).view(-1, 1)
        mask_prob = mask_prob.expand(-1, L)

        update_mask = (torch.rand(B, L, device=self.device) < mask_prob)
        return update_mask


    def to(self, device):

        self.device = torch.device(device)
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        return self


class UniformMaskScheduler:


    def __init__(self, num_steps=200, mask_prob=0.15, mask_token_id=4, device="cuda:0"):
        self.num_steps = num_steps
        self.mask_prob = mask_prob
        self.device = torch.device(device)
        self.mask_token_id = mask_token_id

    @torch.no_grad()
    def q_sample(self, x0, t):


        B, L = x0.shape
        mask = (torch.rand(B, L, device=self.device) < self.mask_prob)
        x_t = x0.clone()
        x_t[mask] = self.mask_token_id
        return x_t

    @torch.no_grad()
    def sample_update_mask(self, x_t, t):


        B, L = x_t.shape
        update_mask = (torch.rand(B, L, device=self.device) < self.mask_prob)
        return update_mask

    def to(self, device):
        self.device = torch.device(device)
        return self
