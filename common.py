import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size):
        self.state_buffer = torch.empty((0, state_dim))
        self.action_buffer = torch.empty((0, action_dim))
        self.reward_buffer = torch.empty((0, 1), dtype=torch.float32)
        self.next_state_buffer = torch.empty((0, state_dim))
        self.done_buffer = torch.empty((0, 1), dtype=torch.int)
        
        self.all_buffers = [
            self.state_buffer, self.action_buffer, self.reward_buffer,
            self.next_state_buffer, self.done_buffer]
        self.max_size = max_size

    def add_entry(self, state, action, reward, next_state, done):
        if self.state_buffer.shape[0] >= self.max_size:
                self.state_buffer = self.state_buffer[1:]
                self.action_buffer = self.action_buffer[1:]
                self.reward_buffer = self.reward_buffer[1:]
                self.next_state_buffer = self.next_state_buffer[1:]
                self.done_buffer = self.done_buffer[1:]

        self.state_buffer = torch.cat(
            (self.state_buffer, torch.from_numpy(state).reshape(1,-1)),dim=0)
        self.action_buffer = torch.cat(
            (self.action_buffer, torch.tensor([action]).reshape(1,-1)), dim=0)
        self.reward_buffer = torch.cat(
            (self.reward_buffer, torch.tensor([reward], dtype=torch.float32).reshape(1,-1)), dim=0)
        self.next_state_buffer = torch.cat(
            (self.next_state_buffer, torch.from_numpy(next_state).reshape(1,-1)), dim=0)
        self.done_buffer = torch.cat(
            (self.done_buffer, torch.tensor([done], dtype=torch.int).reshape(1,-1)), dim=0)
    
    def sample(self, batch_size=1) -> dict:
        indices = torch.randint(0, self.state_buffer.shape[0], size=(batch_size,1)).squeeze()
        batch = (
            self.state_buffer[indices],
            self.action_buffer[indices],
            self.reward_buffer[indices],
            self.next_state_buffer[indices],
            self.done_buffer[indices]
            )
        return batch


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)


    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)

        q = F.relu(self.l1(state_action))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q
        #return self.SpinningUp(state_action)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.min_action = min_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.tanh(self.l3(a))

        return a
    
    def act(self, state, noise=0):
        action_range = self.max_action - self.min_action

        a = self.forward(state) + noise
        action = torch.clamp(
            a*action_range + self.min_action, min=self.min_action, max=self.max_action)
        return action

class DiscreteActor(nn.Module):
    """
    Computes the probability of picking each action, and select an action by sampling from the resulting distribution
    Incompatible with the deterministic policy assumption of TD3, but interesting nonetheless
    """
    def __init__(self, state_dim, n_actions, hidden_dim=256):
        super(DiscreteActor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        probs = F.relu(self.l1(state))
        probs = F.relu(self.l2(probs))
        probs = F.tanh(self.l3(probs))
        
        return probs

    def act(self, state, noise=0):
        probs = F.softmax(self.forward(state) + noise, dim=-1)
        #action = Categorical(probs.cpu()).sample()
        action = torch.argmax(gumbel_softmax(probs, hard=True), dim=1)
        return action#.item()

"""
Utility functions for the Gumbel-Softmax trick (to re-parameterize a policy for discrete actions)

'During training, we let temperature > 0 to allow gradients past the sample, then gradually anneal the temperature (but not completely to 0, as the gradients would blow up)'
References:
-----------
https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/misc.py
"""

def sample_gumbel(shape, eps=1e-20):
    """
    Sample from Gumbel(0,1)
    """
    U = torch.rand(shape, requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Parameters
    ----------
    logits: Tensor of shape (batch_size, n_class)
        Unnormalized log-probs.

    temperature: float
        Must be non-negative.

    hard: bool
        If True, take argmax but differentiate w.r.t soft sample y.

    Returns
    -------
    y: Tensor of shape (batch_size, n_class)
        Sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will be a probability distribution that sums to 1 across classes.
    """
    y = logits + sample_gumbel(logits.shape)
    y =  F.softmax(y/temperature, dim=1)
    if hard:
        y_hard = F.one_hot(torch.argmax(logits, dim=1), num_classes=logits.shape[1])
        y = (y_hard-y).detach() + y
    return y 

    