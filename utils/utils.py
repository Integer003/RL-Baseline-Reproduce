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

def compute_layer_dormant_ratio(layer, inputs, tau=0.01):
    """
    Compute the dormant ratio for a single fully connected layer.
    
    Args:
        layer (nn.Module): The fully connected layer to analyze.
        inputs (torch.Tensor): Input tensor of shape [batch_size, num_features].
        tau (float): Threshold for determining dormant neurons.
    """
    with torch.no_grad():
        outputs = layer(inputs)  # shape: [batch_size, num_neurons]
        abs_outputs = outputs.abs()  # absolute activations
        mean_per_neuron = abs_outputs.mean(dim=0)  # E_x |h_i(x)|, shape: [num_neurons]

        global_mean = mean_per_neuron.mean()  # average across neurons
        scores = mean_per_neuron / (global_mean + 1e-8)

        dormant_mask = scores <= tau
        num_dormant = dormant_mask.sum().item()
        total_neurons = outputs.shape[1]

    return num_dormant, total_neurons

def compute_actor_dormant_ratio(actor, dataset, tau=0.01):
    """
    Compute the dormant ratio for the Actor network.
    `dataset` should be a torch.Tensor of shape [num_samples, repr_dim]
    """
    actor.eval()

    trunk_input = dataset  # shape: [N, repr_dim]
    with torch.no_grad():
        trunk_output = actor.trunk(trunk_input)
    
    # Calculate dormant neurons in trunk
    n_dormant_trunk, n_total_trunk = compute_layer_dormant_ratio(actor.trunk[0], trunk_input, tau)

    # Policy layer 1
    hidden1 = actor.policy[0](trunk_output)
    n_dormant_p1, n_total_p1 = compute_layer_dormant_ratio(actor.policy[0], trunk_output, tau)

    # Policy layer 2
    hidden2 = actor.policy[2](F.relu(hidden1))
    n_dormant_p2, n_total_p2 = compute_layer_dormant_ratio(actor.policy[2], F.relu(hidden1), tau)

    total_dormant = n_dormant_trunk + n_dormant_p1 + n_dormant_p2
    total_neurons = n_total_trunk + n_total_p1 + n_total_p2

    beta_tau = total_dormant / total_neurons
    return beta_tau

def compute_critic_dormant_ratio(critic, obs_batch, action_batch, tau=0.01):
    """
    Compute the dormant ratio for the Critic network.
    Args:
        obs_batch: torch.Tensor of shape [N, repr_dim]
        action_batch: torch.Tensor of shape [N, action_dim]
    """
    critic.eval()

    # Trunk part
    with torch.no_grad():
        trunk_out = critic.trunk(obs_batch)

    n_dormant_trunk, n_total_trunk = compute_layer_dormant_ratio(critic.trunk[0], obs_batch, tau)

    # Q1 network
    q1_input = torch.cat([trunk_out, action_batch], dim=-1)
    q1_hidden1 = critic.Q1[0](q1_input)
    q1_hidden1_relu = F.relu(q1_hidden1)
    q1_hidden2 = critic.Q1[2](q1_hidden1_relu)

    n_dormant_q1_0, n_total_q1_0 = compute_layer_dormant_ratio(critic.Q1[0], q1_input, tau)
    n_dormant_q1_2, n_total_q1_2 = compute_layer_dormant_ratio(critic.Q1[2], q1_hidden1_relu, tau)

    # Q2 network
    q2_input = torch.cat([trunk_out, action_batch], dim=-1)
    q2_hidden1 = critic.Q2[0](q2_input)
    q2_hidden1_relu = F.relu(q2_hidden1)
    q2_hidden2 = critic.Q2[2](q2_hidden1_relu)

    n_dormant_q2_0, n_total_q2_0 = compute_layer_dormant_ratio(critic.Q2[0], q2_input, tau)
    n_dormant_q2_2, n_total_q2_2 = compute_layer_dormant_ratio(critic.Q2[2], q2_hidden1_relu, tau)

    # Sum all
    total_dormant = (n_dormant_trunk + n_dormant_q1_0 + n_dormant_q1_2 +
                     n_dormant_q2_0 + n_dormant_q2_2)
    total_neurons = (n_total_trunk + n_total_q1_0 + n_total_q1_2 +
                     n_total_q2_0 + n_total_q2_2)

    beta_tau = total_dormant / total_neurons
    return beta_tau
