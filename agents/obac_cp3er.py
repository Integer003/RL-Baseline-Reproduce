import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utils as utils

from agents.models.cp3er_networks import CPActor as Actor
from agents.models.cp3er_networks import MoGCritic as Critic
from agents.models.cp3er_networks import Encoder
from agents.models.cp3er_networks import MoGValue as VNetwork

from torch.distributions import Normal, Categorical, MixtureSameFamily

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)



class CP3ERAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau,
                 update_every_steps, use_tb, num_expl_steps, expectile, replay_ratio=1):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.replay_ratio = replay_ratio
        self.num_expl_steps = num_expl_steps
        self.expectile = expectile

        # mog
        self.num_groups = None      # GroupNorm or LayerNorm
        self.num_components = 3     # Number of Gaussian 
        self.init_scale = 1e-3

        # models
        self.encoder = Encoder(self.obs_shape).to(self.device)
        self.actor = Actor(self.encoder.repr_dim, self.action_shape[0], self.device, self.feature_dim, self.hidden_dim)
        # critic
        self.critic = Critic(self.encoder.repr_dim, self.action_shape[0], self.feature_dim,self.hidden_dim, self.num_groups,self.num_components,self.init_scale).to(self.device)
        self.critic_target = Critic(self.encoder.repr_dim, self.action_shape[0], self.feature_dim,self.hidden_dim, self.num_groups,self.num_components,self.init_scale).to(self.device)
        # offline network
        self.value_buffer = VNetwork(self.encoder.repr_dim, feature_dim, hidden_dim, self.num_groups, self.num_components, self.init_scale).to(device)
        self.critic_buffer = Critic(self.encoder.repr_dim, self.action_shape[0], self.feature_dim,self.hidden_dim, self.num_groups,self.num_components,self.init_scale).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_buffer_opt = torch.optim.Adam(self.critic_buffer.parameters(), lr=lr)
        self.value_buffer_opt = torch.optim.Adam(self.value_buffer.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()

        # For logging
        self.obac_bc_times = 0
        self.obac_total_times = 0
    
    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.critic_target.train(training)
        self.value_buffer.train(training)
        self.critic_buffer.train(training)
    
    # act without exp
    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        action = self.actor(obs)
        return action.cpu().numpy()[0]

    def to_distribution(self, mus, stdevs, logits):
        if self.num_components == 1:
            # For a single component, create a standard normal distribution
            dist = Normal(loc=mus[:, 0], scale=stdevs[:, 0])
        else:
            # For multiple components, create a mixture of Gaussian distributions
            dist = MixtureSameFamily(
                mixture_distribution=Categorical(logits=logits),
                component_distribution=Normal(loc=mus, scale=stdevs)
            )
        return dist
    
    def update_value_buffer(self, obs, action):
        metrics = dict()
        obs = obs.detach()

        with torch.no_grad():
            online_info = self.critic(obs, action)
            mus, stdevs, logits = online_info['mus'], online_info['stdevs'], online_info['logits']
            critic_dist = self.to_distribution(mus, stdevs, logits)
            # Q = critic_dist.mean
            if self.init_scale == 0:
                Q = critic_dist.mean
            else:
                Q = critic_dist.sample((20,))

        # V = self.value_buffer(obs)
        V_info = self.value_buffer(obs)
        mus_v, stdevs_v, logits_v = V_info['mus'], V_info['stdevs'], V_info['logits']
        value_dist = self.to_distribution(mus_v, stdevs_v, logits_v)
        V = value_dist.mean

        vf_err = V - Q
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (1 - self.expectile)

        predictor_loss = -value_dist.log_prob(Q)
        predictor_loss = (vf_weight * predictor_loss).mean()

        if self.use_tb:
            metrics['predictor_loss'] = predictor_loss.item()

        self.value_buffer_opt.zero_grad(set_to_none=True)
        predictor_loss.backward()
        self.value_buffer_opt.step()

        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        # both for critic and critic buffer
        metrics = dict()

        critic_loss , aux = self._compute_critic_loss(obs, action, reward, discount,next_obs, step)

        if self.use_tb:
            metrics['critic_q'] = aux['online_Q_mean']
            metrics['critic_target_q'] = aux['target_Q_mean']
            metrics['critic_loss_std'] = aux['critic_loss_std']
            metrics['critic_loss'] = aux['critic_loss']

        critic_loss_buffer, aux = self._compute_critic_loss_buffer(obs, action, reward, discount, next_obs, step)

        if self.use_tb:
            metrics['critic_loss_buffer'] = aux['critic_loss_buffer']
            metrics['critic_loss_std_buffer'] = aux['critic_loss_std_buffer']
            metrics['target_Q_buffer'] = aux['target_Q_buffer']
            metrics['Q_buffer'] = aux['Q_buffer']
   
        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward(retain_graph=True)
        self.critic_opt.step()

        self.critic_buffer_opt.zero_grad(set_to_none=True)
        critic_loss_buffer.backward()
        self.critic_buffer_opt.step()

        self.encoder_opt.step()

        return metrics
    

    def update_actor(self, obs, action, step):
        metrics = dict()

        new_action = self.actor(obs)


        critic_info = self.critic(obs, new_action)
        mus, stdevs, logits = critic_info['mus'], critic_info['stdevs'], critic_info['logits']
        critic_dist = self.to_distribution(mus, stdevs, logits)

        q_estimate  = critic_dist.mean
        q_mean = torch.mean(q_estimate)
        q_loss = -q_mean

        critic_info = self.critic_buffer(obs, action)
        mus_buffer, stdevs_buffer, logits_buffer = critic_info['mus'], critic_info['stdevs'], critic_info['logits']
        critic_dist_buffer = self.to_distribution(mus_buffer, stdevs_buffer, logits_buffer)

        q_buffer_estimate = critic_dist_buffer.mean

        bc_loss = (self.actor.cm.consistency_losses(action, obs) * (q_buffer_estimate > q_estimate).float().detach()).mean()
        actor_loss = q_loss + 0.05 * bc_loss
        self.obac_bc_times = (q_buffer_estimate > q_estimate).float().sum().item()
        self.obac_total_times = obs.shape[0]

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['q_loss'] = q_loss.item()
            metrics['bc_loss'] = bc_loss.item()
            metrics['obac_rate'] = self.obac_bc_times / self.obac_total_times

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        org_obs, action, reward, discount, org_next_obs = utils.to_torch(
            batch, self.device)
        
        reward = reward.unsqueeze(1) if reward.dim() == 1 else reward
        discount = discount.unsqueeze(1) if discount.dim() == 1 else discount

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        for _ in range(self.replay_ratio):
            # augment
            obs = self.aug(org_obs.float())
            next_obs = self.aug(org_next_obs.float())
            org_obs =  obs / 255.0 - 0.5
            obs = self.encoder(obs)

            with torch.no_grad():
                next_obs = self.encoder(next_obs)

            # update value buffer
            value_buffer_metrics = self.update_value_buffer(obs, action)
            # update critic & critic buffer
            critic_metrics = self.update_critic(obs, action, reward, discount, next_obs, step)
            # update actor
            actor_metrics = self.update_actor(obs.detach(), action, step)
            
            # update critic target
            utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
            
        metrics.update(critic_metrics)
        metrics.update(actor_metrics)
        metrics.update(value_buffer_metrics)

        return metrics
    
    def _compute_critic_loss(self,  obs, act, rew, discount, next_obs, step):

        with torch.no_grad():

            next_action = self.actor(next_obs)

            target_info = self.critic_target(next_obs, next_action)
            mus, stdevs, logits = target_info['mus'], target_info['stdevs'], target_info['logits']

            if self.init_scale == 0:
                target_Q_dist = self.to_distribution(mus, stdevs, logits)
                target_Q = target_Q_dist.mean
            else:
                target_Q_dist = self.to_distribution(mus, stdevs, logits)
                target_Q = target_Q_dist.sample((20,))

            # compute target_Q
            target_Q = rew + discount * target_Q

        online_info = self.critic(obs, act)
        mus, stdevs, logits = online_info['mus'], online_info['stdevs'], online_info['logits']
        online_Q_dist = self.to_distribution(mus, stdevs, logits)

        # compute loss 
        critic_loss = -online_Q_dist.log_prob(target_Q)
        critic_loss_mean = critic_loss.mean()
        critic_loss_std = critic_loss.std()

        aux = {
            'critic_loss': critic_loss_mean.item(),
            'critic_loss_std': critic_loss_std.item(),
            'target_Q_mean': target_Q.mean().item(),
            'online_Q_mean': online_Q_dist.mean.mean().item(),
        }

        return critic_loss_mean, aux
    
    def _compute_critic_loss_buffer(self,  obs, act, rew, discount, next_obs, step):

        with torch.no_grad():
            value_info = self.value_buffer(next_obs)
            mus_v, stdevs_v, logits_v = value_info['mus'], value_info['stdevs'], value_info['logits'] 
            value_dist = self.to_distribution(mus_v, stdevs_v, logits_v)
            # target_Q = value_dist.mean
            if self.init_scale == 0:
                target_Q = value_dist.mean
            else:
                target_Q = value_dist.sample((20,))
            target_Q = rew + discount * target_Q

        offline_info = self.critic_buffer(obs, act)
        mus, stdevs, logits = offline_info['mus'], offline_info['stdevs'], offline_info['logits']
        offline_Q_dist = self.to_distribution(mus, stdevs, logits)

        critic_loss = -offline_Q_dist.log_prob(target_Q)
        critic_loss_mean = critic_loss.mean()
        critic_loss_std = critic_loss.std()

        aux = {
            'critic_loss_buffer': critic_loss_mean.item(),
            'critic_loss_std_buffer': critic_loss_std.item(),
            'target_Q_buffer': target_Q.mean().item(),
            'Q_buffer': offline_Q_dist.mean.mean().item(),
        }

        return critic_loss_mean, aux
