"""
Based on: https://github.com/crowsonkb/k-diffusion
"""

import torch as torch
import torch.nn as nn
import numpy as np
import copy
import math
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim, gate_dim, hidden_dim, top_k, dropout=0.1):
        super(MoE, self).__init__()
        print("Initializing MoE with {} experts, input_dim={}, output_dim={}, gate_dim={}, hidden_dim={}, top_k={}".format(
            num_experts, input_dim, output_dim, gate_dim, hidden_dim, top_k))
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.top_k = top_k

        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, output_dim), nn.Mish())
            for _ in range(num_experts)])
        self.gate = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, gate_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(gate_dim, num_experts))])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def forward(self, x, metrics=None):
        batch_size = x.size(0)

        # Gate scores
        gate_scores_logits_ = self.gate[0](x)
        gate_scores_logits = gate_scores_logits_
        gate_scores = F.softmax(gate_scores_logits, dim=1)

        # Top-k gate scores and indices
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=1)

        if metrics is not None:
            for i in range(self.num_experts):
                metrics['expert_{}_usage_rate'.format(i)] = (top_k_indices == i).sum().item() / batch_size

        # Pass inputs through all experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        expert_outputs = expert_outputs.permute(1, 0, 2)

        # Advanced indexing for selecting top-k expert outputs
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.top_k).reshape(-1).to(x.device)
        selected_expert_outputs = expert_outputs[batch_indices, top_k_indices.reshape(-1)]
        selected_expert_outputs = selected_expert_outputs.reshape(batch_size, self.top_k, self.output_dim)

        # Scale the selected expert outputs by the corresponding gate scores
        scaled_expert_outputs = selected_expert_outputs * top_k_scores.unsqueeze(2)
        scaled_expert_outputs /= (top_k_scores.sum(dim=1, keepdim=True).unsqueeze(2) + 1e-9)

        # Sum the scaled expert outputs for the final output
        combined_output = scaled_expert_outputs.sum(dim=1)

        aux_loss = self.moe_auxiliary_loss(gate_scores, top_k_indices)

        return combined_output, aux_loss

    def moe_auxiliary_loss(self, gate_scores, top_k_indices, lambda_balance=1.0, lambda_entropy=1.0):
        batch_size, num_experts = gate_scores.size()

        # Load Balancing Loss
        one_hot = F.one_hot(top_k_indices, num_classes=num_experts).float()
        expert_load = one_hot.sum(dim=[0, 1]) / (batch_size + 1e-9)
        load_balancing_loss = expert_load.var()

        # Entropy Loss
        entropy = -(gate_scores * torch.log(gate_scores + 1e-9)).sum(dim=1).mean()

        # Combine the losses
        auxiliary_loss = lambda_balance * load_balancing_loss + lambda_entropy * entropy

        return auxiliary_loss

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DenoisingModel(nn.Module):
    """
    DenoisingModel Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=16):

        super(DenoisingModel, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            # nn.ReLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim

        moe_dim = hidden_dim // 4
        self.mid_layer = MoE(num_experts=16, input_dim=input_dim, output_dim=moe_dim,
                                gate_dim=moe_dim, hidden_dim=moe_dim, top_k=4, dropout=0.1)

        self.final_layer = nn.Linear(moe_dim, action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)

        x = torch.cat([x, t, state], dim=1)
        x, aux_loss = self.mid_layer(x)

        return self.final_layer(x), aux_loss

def get_generator(generator, num_samples=0, seed=0):
    if generator == "dummy":
        return DummyGenerator()
    else:
        raise NotImplementedError

class DummyGenerator:
    def randn(self, *args, **kwargs):
        return torch.randn(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return torch.randint(*args, **kwargs)

    def randn_like(self, *args, **kwargs):
        return torch.randn_like(*args, **kwargs)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = torch.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = torch.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings

class ConsistencyModel(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
    ):
        super(ConsistencyModel, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = DenoisingModel(state_dim=state_dim, action_dim=action_dim, 
                         device=device, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        if noise is None:
            noise = torch.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None):
            return self.denoise(self.model, x, t, state)[1]

        @torch.no_grad()
        def target_denoise_fn(x, t, state=None):
            return self.denoise(target_model, x, t, state)[1]

        @torch.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = torch.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = torch.get_rng_state()
        distiller = denoise_fn(x_t, t, state)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        torch.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss
    
    def loss(self, x_start, state, noise=None, td_weights=None):
        num_scales = self.steps
        if noise is None:
            noise = torch.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None):
            return self.denoise(self.model, x, t, state)[1]

        indices = torch.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = torch.get_rng_state()
        distiller = denoise_fn(x_t, t, state)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = torch.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        model_output, aux_loss = model(c_in * x_t, rescaled_t, state)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return model_output, denoised, aux_loss

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0, aux_loss = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0, aux_loss = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0, aux_loss
    
    def sample_onestep(self, state, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        _, denoised, aux_loss = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)
        return denoised, aux_loss

    def sample_multistep(self, state, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        x = x_T
        aux_losses = []
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            _, x0, aux_loss = self.denoise(self.model, x, t * s_in, state)
            aux_losses.append(aux_loss)
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        _, x, final_aux_loss = self.denoise(self.model, x, t * s_in, state)
        aux_losses.append(final_aux_loss)
        
        # Average all auxiliary losses from the sampling steps
        total_aux_loss = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=self.device)
        
        return x, total_aux_loss
    
    def forward(self, state, eval=False, multistep=False):
        if multistep:
            x_0, aux_loss = self.sample_multistep(state, eval=eval)
        else:
            x_0, aux_loss = self.sample_onestep(state, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        return x_0, aux_loss