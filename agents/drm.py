# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utils as utils


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

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


# class Actor(nn.Module):
#     def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
#         super().__init__()

#         self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
#                                    nn.LayerNorm(feature_dim), nn.Tanh())

#         self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
#                                     nn.ReLU(inplace=True),
#                                     nn.Linear(hidden_dim, hidden_dim),
#                                     nn.ReLU(inplace=True),
#                                     nn.Linear(hidden_dim, action_shape[0]))

#         self.apply(utils.weight_init)

#     def forward(self, obs, std):
#         h = self.trunk(obs)

#         mu = self.policy(h)
#         mu = torch.tanh(mu)
#         std = torch.ones_like(mu) * std

#         dist = utils.TruncatedNormal(mu, std)
#         return dist


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, dr_tau):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        # self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(hidden_dim, hidden_dim),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(hidden_dim, action_shape[0]))
        
        self.policy_linear1 = nn.Linear(feature_dim, hidden_dim)
        self.policy_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_linear3 = nn.Linear(hidden_dim, action_shape[0])
        self.policy_activation = nn.ReLU(inplace=True)

        self.apply(utils.weight_init)

        self.dr_tau = dr_tau
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.action_shape = action_shape

    def get_dormant_count(self, mu):
        # calculate the dormant ratio & number of dormant neurons

        mu = torch.absolute(mu)
        mu = mu.mean(dim=0)
        mean_mu = mu.mean()

        H = (mu < mean_mu * self.dr_tau).float().sum()
        return H

    def forward(self, obs):
        dormant_count = 0

        h = self.trunk(obs)
        dormant_count += self.get_dormant_count(h)

        mu = self.policy_activation(self.policy_linear1(h))
        dormant_count += self.get_dormant_count(mu)

        mu = self.policy_activation(self.policy_linear2(mu))
        dormant_count += self.get_dormant_count(mu)

        mu = self.policy_linear3(mu)
        mu = torch.tanh(mu)
        dormant_count += self.get_dormant_count(mu)

        beta = dormant_count / (self.feature_dim + self.hidden_dim * 2 + self.action_shape[0])

        return mu, beta

def get_action_distribution(mu, std):
    std = torch.ones_like(mu) * std
    dist = utils.TruncatedNormal(mu, std)
    return dist

class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

class Evaluator(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.value = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)

        value = self.value(h)

        return value


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, dr_tau, use_perturbation, perturbation_interval, perturbation_rate, max_perturbation, min_perturbation, beta_threshold, exploration_temperature, use_drg_exploration, use_drg_exploitation, exploitation_temperature, exploitation_expectile, max_exploitation_lambda):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.action_shape = action_shape
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.dr_tau = dr_tau
        self.use_perturbation = use_perturbation
        self.perturbation_interval = perturbation_interval
        self.perturbation_rate = perturbation_rate
        self.max_perturbation = max_perturbation
        self.min_perturbation = min_perturbation
        self.beta_threshold = beta_threshold
        self.exploration_temperature = exploration_temperature
        self.use_drg_exploration = use_drg_exploration
        self.awaken_step = None
        self.use_drg_exploitation = use_drg_exploitation
        self.exploitation_temperature = exploitation_temperature
        self.exploitation_expectile = exploitation_expectile
        self.max_exploitation_lambda = max_exploitation_lambda

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim, self.dr_tau).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.evaluator = Evaluator(self.encoder.repr_dim, feature_dim, hidden_dim).to(device)

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.evaluator_opt = torch.optim.Adam(self.evaluator.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.critic_target.train()
        self.evaluator.train(training)

    def calculate_alpha_perturbation(self, beta):
        return np.clip(1.0 - beta * self.perturbation_rate, self.min_perturbation, self.max_perturbation)

    def get_stddev(self, step, beta): # for exploration
        beta = beta.item() if isinstance(beta, torch.Tensor) else beta
        if self.use_drg_exploration:
            stddev = 1 / (1 + np.exp(-(beta - self.beta_threshold) / self.exploration_temperature))
            if beta < self.beta_threshold and step > self.num_expl_steps:
                if self.awaken_step is None:
                    self.awaken_step = step
                stddev_0 = utils.schedule(self.stddev_schedule, step - self.awaken_step)
                stddev = np.max([stddev, stddev_0])
        else:
            stddev = utils.schedule(self.stddev_schedule, step)
        return stddev
    
    def get_lambda_exploitation(self, beta):
        beta = beta.item() if isinstance(beta, torch.Tensor) else beta
        lambda_exploitation = self.max_exploitation_lambda / (1 + np.exp(-(beta - self.beta_threshold) / self.exploitation_temperature))
        return lambda_exploitation


    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        mu, beta = self.actor(obs)
        stddev = self.get_stddev(step, beta)
        dist = get_action_distribution(mu, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            mu, beta = self.actor(next_obs)
            stddev = self.get_stddev(step, beta)
            dist = get_action_distribution(mu, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V_0 = torch.min(target_Q1, target_Q2)
            if self.use_drg_exploitation:
                target_V_1 = self.evaluator(next_obs)
                lambda_exploitation = self.get_lambda_exploitation(beta)
                target_V = lambda_exploitation * target_V_0 + (1 - lambda_exploitation) * target_V_1
                target_Q = reward + (discount * target_V).detach()
            else:
                target_Q = reward + (discount * target_V_0).detach()

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        mu, beta = self.actor(obs)
        stddev = self.get_stddev(step, beta)
        dist = get_action_distribution(mu, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # do perturbation on actor
        if self.use_perturbation and step % self.perturbation_interval == 0:
            alpha = self.calculate_alpha_perturbation(beta.item())
            self.random_actor = Actor(self.encoder.repr_dim, self.action_shape, self.feature_dim, self.hidden_dim, self.dr_tau).to(self.device)
            for param, random_param in zip(self.actor.parameters(), self.random_actor.parameters()):
                if param.requires_grad:
                    param.data.copy_(alpha * param.data + (1 - alpha) * random_param.data)
            # with torch.no_grad():
            #     for param in self.actor.parameters():
            #         # random_param = torch.empty_like(param).uniform_(-1.0, 1.0)
            #         random_param = torch.empty_like(param)
            #         utils.weight_init(random_param)
            #         param.data.copy_(alpha * param.data + (1 - alpha) * random_param)

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics['actor_beta'] = beta.item()

        return metrics
    
    def update_evaluator(self, obs, action, step):
        metrics = dict()

        with torch.no_grad():
            Q1, Q2 = self.critic(obs, action)
            Q = torch.min(Q1, Q2)

        V = self.evaluator(obs)
        weights = torch.where(V > Q, torch.ones_like(V) * (1 - self.exploitation_expectile), torch.ones_like(V) * self.exploitation_expectile)
        evaluator_loss = (F.mse_loss(V, Q, reduction='none') * weights).mean()


        self.evaluator_opt.zero_grad(set_to_none=True)
        evaluator_loss.backward()
        self.evaluator_opt.step()

        if self.use_tb:
            metrics['evaluator_loss'] = evaluator_loss.item()
            metrics['evaluator_value'] = V.mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update evaluator
        if self.use_drg_exploitation:
            metrics.update(self.update_evaluator(obs.detach(), action, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
