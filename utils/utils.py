# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

import copy
import math

class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

def calculate_dormant_ratio(model, inputs, tau=0.01):
    """
    Compute the τ-dormant ratio of a model following Algorithm 1 in the DrM paper.

    Args:
        model (nn.Module): PyTorch model (e.g., Actor, Critic).
        inputs (Tensor): Input tensor for forward propagation.
        tau (float): Dormant threshold.

    Returns:
        float: dormant ratio (dormant_neurons / total_neurons)
    """
    # model.eval()
    total_neurons = 0
    dormant_neurons = 0
    layer_outputs = []

    # Hook to store the outputs of Linear layers
    def save_output(module, input, output):
        layer_outputs.append(output)

    handles = []
    # Register hooks on all nn.Linear layers
    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(save_output))

    # Forward pass
    with torch.no_grad():
        _ = model(*inputs)

    # Analyze outputs
    for output in layer_outputs:
        # Step 6: mean absolute activation over batch
        mean_abs = output.abs().mean(dim=0)  # shape: [num_neurons]
        # Step 7: mean across neurons
        avg_output = mean_abs.mean()  # scalar
        # Step 8: count neurons with low activation
        dormant_count = (mean_abs < tau * avg_output).sum().item()
        neuron_count = mean_abs.numel()

        dormant_neurons += dormant_count
        total_neurons += neuron_count

    # Clean up hooks
    for h in handles:
        h.remove()

    # Step 12: return ratio
    return dormant_neurons / total_neurons if total_neurons > 0 else 0.0

def perturb_network(network, optimizer, beta, alpha_min=0.2, alpha_max=0.9, k=2.0):
    """
    Perform dormant-ratio-guided soft weight reset on a network.
    
    Args:
        network (nn.Module): Network to perturb.
        optimizer (Optimizer): Optimizer to reset.
        beta (float): Dormant ratio.
        alpha_min (float): Minimum interpolation factor.
        alpha_max (float): Maximum interpolation factor.
        k (float): Perturbation rate.
        
    Returns:
        nn.Module, Optimizer: Perturbed network and reset optimizer.
    """
    alpha = 1 - k * beta
    alpha = max(alpha_min, min(alpha_max, alpha))  # clip(alpha, α_min, α_max)

    new_net = copy.deepcopy(network.__class__.__new__(network.__class__))  # create empty network
    new_net.__init__(*network.__init_args__)  # reinitialize weights (must store args)
    new_net = new_net.to(next(network.parameters()).device)

    with torch.no_grad():
        for old_param, new_param in zip(network.parameters(), new_net.parameters()):
            old_param.data.copy_(alpha * old_param.data + (1 - alpha) * new_param.data)

    # Reset optimizer state
    optimizer = type(optimizer)(network.parameters(), lr=optimizer.defaults['lr'])

    return network, optimizer
