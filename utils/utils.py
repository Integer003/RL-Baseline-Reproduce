import random
import re
import time

from copy import deepcopy
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

import torch.nn.functional as F
from omegaconf import OmegaConf

from queue import PriorityQueue

import sys
sys.path.append(".")



# !TACO
### input shape: (batch_size, length, action_dim)
### output shape: (batch_size, action_dim)
class ActionEncoding(nn.Module):
    def __init__(self, action_dim, latent_action_dim, multistep):
        super().__init__()
        self.action_dim = action_dim
        self.action_tokenizer = nn.Sequential(
            nn.Linear(action_dim, 64), nn.Tanh(),
            nn.Linear(64, latent_action_dim)
        )
        self.action_seq_tokenizer = nn.Sequential(
            nn.Linear(latent_action_dim*multistep, latent_action_dim*multistep),
            nn.LayerNorm(latent_action_dim*multistep), nn.Tanh()
        )
        self.apply(weight_init)
        
    def forward(self, action, seq=False):
        if seq:
            batch_size = action.shape[0]
            action = self.action_tokenizer(action) #(batch_size, length_action_dim)
            action = action.reshape(batch_size, -1)
            return self.action_seq_tokenizer(action)
        else:
            return self.action_tokenizer(action)

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

# !TACO
def expectile_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)

# !TACO
def encode_multiple(encoder, xs, detach_lst):
    length = [x.shape[0] for x in xs]
    xs, xs_lst = torch.cat(xs, dim=0), []
    xs = encoder(xs)
    start = 0
    for i in range(len(detach_lst)):
        x = xs[start:start+length[i], :]
        if detach_lst[i]:
            x = x.detach()
        xs_lst.append(x)
        start += length[i]
    return xs_lst

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


class LinearOutputHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)


# !drm
def cal_dormant_ratio(model, *inputs, percentage=0.025, seq=False, metrics=None):
    hooks = []
    hook_handlers = []
    total_neurons = 0
    dormant_neurons = 0

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = LinearOutputHook()
            hooks.append(hook)
            hook_handlers.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(*inputs)
    layer_id = 0
    for module, hook in zip(
        (module for module in model.modules() if isinstance(module, nn.Linear)), hooks):
        with torch.no_grad():
            for output_data in hook.outputs:
                if seq:
                    mean_output = output_data.abs().mean((0, 1))
                else:
                    mean_output = output_data.abs().mean(0)
                avg_neuron_output = mean_output.mean()
                dormant_indices = (mean_output < avg_neuron_output *
                                   percentage).nonzero(as_tuple=True)[0]
                total_neurons += module.weight.shape[0]
                dormant_neurons += len(dormant_indices)  
                if metrics is not None:   
                    metrics['layer_{}_dormant_ratio'.format(layer_id)] = len(dormant_indices) / module.weight.shape[0]
                layer_id += 1

    for hook in hooks:
        hook.outputs.clear()

    for hook_handler in hook_handlers:
        hook_handler.remove()

    return dormant_neurons / total_neurons, metrics

# !drm
def perturb_drm(net, optimizer, perturb_factor):
    linear_keys = [
        name for name, mod in net.named_modules()
        if isinstance(mod, torch.nn.Linear)
    ]
    new_net = deepcopy(net)
    new_net.apply(weight_init)

    for name, param in net.named_parameters():
        if any(key in name for key in linear_keys):
            noise = new_net.state_dict()[name] * (1 - perturb_factor)
            param.data = param.data * perturb_factor + noise
        else:
            param.data = net.state_dict()[name]
    optimizer.state = defaultdict(dict)
    return net, optimizer

# !moe
def perturb(net, optimizer, perturb_factor, tp_set=None, name="actor"):
    if tp_set is None:
        linear_keys = [
            name for name, mod in net.named_modules()
            if isinstance(mod, torch.nn.Linear)
        ]
        new_net = deepcopy(net)
        new_net.apply(weight_init)

        for n, param in net.named_parameters():
            if any(key in n for key in linear_keys):
                noise = new_net.state_dict()[n] * (1 - perturb_factor)
                param.data = param.data * perturb_factor + noise
            else:
                param.data = net.state_dict()[n]
        optimizer.state = defaultdict(dict)
        return net, optimizer
    else:
        print("perturbing model parameters with tp_set...")
        linear_keys = [
            name for name, mod in net.named_modules()
            if (isinstance(mod, torch.nn.Linear) or isinstance(mod, torch.nn.Conv2d))
        ]
        new_net = deepcopy(net)
        new_net.apply(weight_init)
        if name == "actor_enc":
            tp_set.sampled_model(new_net, tp_set.sample_params(tp_set.cal_params_stats(tp_set.actors)))
        elif name == "actor_moe_expert":
            tp_set.sampled_model(new_net, tp_set.sample_params(tp_set.cal_params_stats(tp_set.moes, moe_expert=True)), moe_expert=True)
        elif name == "actor_moe_gate":
            tp_set.sampled_model(new_net, tp_set.sample_params(tp_set.cal_params_stats(tp_set.gates, moe_gate=True)), moe_gate=True)
        elif name == "critic":
            tp_set.sampled_model(new_net, tp_set.sample_params(tp_set.cal_params_stats(tp_set.critics)))
        elif name == "critic_target":
            tp_set.sampled_model(new_net, tp_set.sample_params(tp_set.cal_params_stats(tp_set.critic_targets)))
        elif name == "value_buffer":
            tp_set.sampled_model(new_net, tp_set.sample_params(tp_set.cal_params_stats(tp_set.value_buffers)))

        for n, param in net.named_parameters():     
            if any(key in n for key in linear_keys):
                param.data = param.data * perturb_factor\
                    + new_net.state_dict()[n] * (1 - perturb_factor)
            else:
                param.data = net.state_dict()[n]
        optimizer.state = defaultdict(dict)
        return net, optimizer


# !moe
class models_tuple(object):
    def __init__(self, maxsize=128, moe=False, gate=False):
        self.maxsize = maxsize
        self.length = 0
        self.episode_reward = []
        self.actors = []
        self.critics = []
        self.critic_targets = []
        self.value_buffers = []
        self.moe = moe
        if self.moe:
            self.moes = []
        self.gate = gate
        if self.gate:
            self.gates = []

    def add(self, episode_reward, actor, critic, critic_target, value_buffer, moe=None, gate=None):
        if self.length < self.maxsize:
            self.episode_reward.append(episode_reward)
            self.actors.append(actor)
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.value_buffers.append(value_buffer)
            self.length += 1               
            if self.moe:
                self.moes.append(moe)
            if self.gate:
                self.gates.append(gate)
        else:
            min_idx = self.episode_reward.index(min(self.episode_reward))
            if episode_reward > self.episode_reward[min_idx]:
                self.episode_reward[min_idx] = episode_reward
                self.actors[min_idx] = actor
                self.critics[min_idx] = critic
                self.critic_targets[min_idx] = critic_target
                self.value_buffers[min_idx] = value_buffer
                if self.moe:
                    self.moes[min_idx] = moe
                if self.gate:
                    self.gates[min_idx] = gate
    
    def log(self, metrics):
        metrics['tp_set_mean_episode_reward'] = np.mean(self.episode_reward)
        return metrics

    def cal_params_stats(self, models, moe_expert=False, moe_gate=False):
        print("Calculating parameters statistics...")
        param_stats = {}
        
        for name, _ in models[0].named_parameters():
            # if moe_expert != ("moe.experts" in name) or moe_gate != ("moe.gate" in name):
            if not(moe_expert or moe_gate) and "moe" in name:
                continue
            
            print(f"  - Processing parameter: {name}")
            stacked_params = torch.stack([model.state_dict()[name] for model in models])
            mean = stacked_params.mean(dim=0)
            std = torch.clamp(stacked_params.std(dim=0), min=1e-7)
            param_stats[name] = (mean, std)
        
        return param_stats

    def sample_params(self, params_stats):
        sampled_params = {}
        for name, (mean, std) in params_stats.items():
            dist = pyd.Normal(mean, std)
            sampled_params[name] = dist.sample()
        return sampled_params

    def sampled_model(self, model, sampled_params, moe_expert=False, moe_gate=False):
        state_dict = model.state_dict()
        
        for name, param in sampled_params.items():
            if name in state_dict:
                state_dict[name].copy_(param)
        
        # Load the updated state dictionary back into the model
        model.load_state_dict(state_dict)
        return model
