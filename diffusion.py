import math

import torch
from torch import nn
import torch.nn.functional as F


class CDenoiserBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, condition):
        residual = x
        x = self.norm1(x)
        x = self.mlp(x)
        x = residual + self.alpha * x

        residual = x
        x = self.norm2(x + condition)
        x = residual + self.alpha * x
        return x


class CDenoiser(nn.Module):
    def __init__(self, num_entities, condition_dim, hidden_dim, num_blocks=1, num_steps=40):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.input_proj = nn.Linear(num_entities, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [CDenoiserBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_entities),
        )

    def sinusoidal_embedding(self, t):
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.hidden_dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x_t, x_c, t):
        dtype = self.input_proj.weight.dtype
        x_t = x_t.to(dtype=dtype)
        x_c = x_c.to(dtype=dtype)

        t_emb = self.time_embed(self.sinusoidal_embedding(t).to(dtype=dtype))
        c_emb = self.condition_proj(x_c)
        x_ct = c_emb + t_emb

        h = self.input_proj(x_t)
        for block in self.blocks:
            h = block(h, x_ct)
        return self.output_proj(h)


class KGDiffusion(nn.Module):
    def __init__(self, num_entities, condition_dim, hidden_dim, num_steps=40, num_blocks=1):
        super().__init__()
        self.num_entities = num_entities
        self.num_steps = num_steps
        self.denoiser = CDenoiser(
            num_entities=num_entities,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_steps=num_steps,
        )

        betas = self._cosine_beta_schedule(num_steps)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    @staticmethod
    def _cosine_beta_schedule(num_steps):
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps, dtype=torch.float32)
        alpha_bar = torch.cos(
            ((x / num_steps) + 0.008) / (1 + 0.008) * math.pi * 0.5
        ) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(0.0001, 0.9999)

    def forward_diffuse(self, x_0, t):
        x_0 = x_0.float()
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)
        eps = torch.randn_like(x_0)
        x_t = alpha_bar_t.sqrt() * x_0 + (1 - alpha_bar_t).sqrt() * eps
        return x_t, eps

    def compute_loss_G(self, x_0, x_c):
        x_0 = x_0.detach().float()
        batch_size = x_0.shape[0]
        t = torch.randint(
            0,
            self.num_steps,
            (batch_size,),
            device=x_0.device,
            dtype=torch.long,
        )
        x_t, eps_true = self.forward_diffuse(x_0, t)
        eps_pred = self.denoiser(x_t, x_c, t)
        return F.mse_loss(eps_pred.float(), eps_true.float())

    @torch.no_grad()
    def reverse_sample(self, x_c, device=None):
        device = device or x_c.device
        dtype = self.denoiser.input_proj.weight.dtype
        batch_size = x_c.shape[0]
        x_t = torch.randn(batch_size, self.num_entities, device=device, dtype=dtype)

        for step in reversed(range(self.num_steps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            eps_pred = self.denoiser(x_t, x_c, t)
            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bar[step]
            mean = (1 / alpha_t.sqrt()) * (
                x_t - (beta_t / (1 - alpha_bar_t).sqrt()) * eps_pred
            )
            if step > 0:
                x_t = mean + beta_t.sqrt() * torch.randn_like(x_t)
            else:
                x_t = mean
        return x_t
