"""Useful utils for controllers"""

import torch
import torch.nn.functional as F

def calc_prob(critic_logits, x):
    """compute entropy and log_prob."""
    softmax_logits, log_softmax_logits = critic_logits
    ent = softmax_logits * log_softmax_logits
    ent = -1 * ent.sum()
    log_prob = -F.nll_loss(log_softmax_logits, x.view(1))
    return ent, log_prob


def compute_critic_logits(logits):
    softmax_logits = F.softmax(logits, dim=-1)
    log_softmax_logits = F.log_softmax(logits, dim=-1)
    critic_logits = (softmax_logits, log_softmax_logits)
    return critic_logits


def sample_logits(critic_logits):
    softmax_logits = critic_logits[0]
    x = softmax_logits.multinomial(num_samples=1)
    ent, log_prob = calc_prob(critic_logits, x)
    return x, ent, log_prob


def torch_long(x):
    x = torch.tensor(x, dtype=torch.long)
    return x.view((1, 1))

