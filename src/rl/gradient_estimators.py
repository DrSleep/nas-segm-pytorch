"""REINFORCE and PPO for controller training"""

import random

import torch
import torch.nn as nn

from helpers.storage import RolloutStorage
from helpers.utils import parse_geno_log


class REINFORCE(object):
    """REINFORCE gradient estimator
    """
    def __init__(self, controller, lr, baseline_decay, max_grad_norm=2.0):
        """
        Args:
          controller (Controller): RNN architecture generator
          lr (float): learning rate for controller optimizer
          baseline_decay (float): moving average baseline decay
          max_grad_norm (float): controller gradient clip
        """
        self.baseline = None
        self.decay = baseline_decay
        self.controller = controller
        self.optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm

    def update(self, sample):
        """Perform one gradient step for controller and update baseline.
        Args:
          sample (tuple): (reward, action, log_prob)
            reward (float): current reward
            action (list): representation of current architecture
            log_prob (float): log probability of current architecture

        Returns:
          loss (torch.FloatTensor): controller loss
          entropy (torch.FloatTensor): entropy of current architecture
        """
        reward, action, _, _ = sample
        _, _, entropy, log_prob = self.controller.evaluate(action)
        with torch.no_grad():
            if self.baseline is None:
                self.baseline = reward
            else:
                self.baseline = self.decay * self.baseline + (1 - self.decay) * reward

            adv = reward - self.baseline

        loss = -log_prob * adv
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.controller.parameters(),
                                 self.max_grad_norm)
        self.optimizer.step()
        return loss, entropy

    def state_dict(self):
        return {'baseline': self.baseline,
                'controller': self.controller.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, states):
        self.controller.load_state_dict(states['controller'])
        self.baseline = states['baseline']
        self.optimizer.load_state_dict(states['optimizer'])


class PPO(object):
    """Proximal Policy Optimization with rollout buffer
    part of the update code modified from:

    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py
    """
    def __init__(self, controller, clip_param, lr, baseline_decay,
                 action_size=18, ppo_epoch=1, num_mini_batch=100,
                 max_grad_norm=2.0, entropy_coef=0, num_steps=100, num_processes=1):
        """
        Args:
          controller (Controller): RNN architecture generator
          clip_param (float): PPO clip parameter epsilon
          lr (float): learning rate for controller optimizer
          baseline_decay (float): moving average baseline decay
          action_size (int): length of architecture representation
          ppo_epoch (int): number of epochs to train
          num_mini_batch (int): number of mini batches in the rollout buffer
          max_grad_norm (float): controller gradient clip
          entropy_coef (float): gradient coefficient for entropy regularization
          num_steps (int): number of steps to train
          num_processes (int): samples per step
        """
        self.ppo_epoch = ppo_epoch
        self.controller = controller
        self.optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
        self.num_mini_batch = num_mini_batch
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.rollouts = RolloutStorage(num_steps, num_processes, action_size)
        self.baseline = None
        self.decay = baseline_decay

    def state_dict(self):
        return {'baseline': self.baseline,
                'rollouts': self.rollouts,
                'controller': self.controller.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, states):
        self.controller.load_state_dict(states['controller'])
        self.optimizer.load_state_dict(states['optimizer'])
        self.baseline = states['baseline']
        if not 'rollouts' in states:
            # continue from old checkpoint format
            # fill in rollouts
            with open('genotypes.out') as ro_file:
                lines = ro_file.readlines()
                # randomly pick
                random.shuffle(lines)
                records = lines[:self.rollouts.num_steps]
            for record in records:
                reward, action = parse_geno_log(record)
                with torch.no_grad():
                    _, _, _, log_prob = self.controller.evaluate(action)
                self.update((reward, action, log_prob), is_train=False)
            print(self.rollouts.actions)
        else:
            self.rollouts = states['rollouts']

    def update(self, sample, is_train=True):
        reward, action, log_prob = sample
        if self.baseline is None:
            self.baseline = reward
        else:
            self.baseline = self.decay * self.baseline + (1 - self.decay) * reward
        self.rollouts.insert(action, log_prob, reward)
        if not is_train:
            return -1, -1
        advantages = self.rollouts.rewards - self.baseline
        loss_epoch = 0
        entropy_epoch = 0

        for _ in range(self.ppo_epoch):
            data_generator = self.rollouts.generator(advantages, self.num_mini_batch)
            for sample in data_generator:
                actions_batch, rewards_batch, old_actions_log_probs_batch, \
                    adv_targ = sample

                action_log_probs, entropy = self.controller.evaluate_actions(actions_batch)

                ratio = torch.exp(action_log_probs -
                                  torch.from_numpy(old_actions_log_probs_batch).float())
                adv_targ_th = torch.from_numpy(adv_targ).float()
                surr1 = ratio * adv_targ_th
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ_th
                action_loss = -torch.min(surr1, surr2).mean()
                self.optimizer.zero_grad()
                dist_entropy = entropy.mean()
                (action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.controller.parameters(), self.max_grad_norm)
                self.optimizer.step()

                loss_epoch += action_loss.item()
                entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        return loss_epoch / num_updates, entropy_epoch / num_updates
