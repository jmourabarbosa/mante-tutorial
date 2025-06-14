
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

import torch.nn as nn
from math import sqrt, floor
import random
import time





def accuracy_general(output, targets, mask):
    good_trials = (targets != 0).any(dim=1).squeeze()
    target_decisions = torch.sign(
        (targets[good_trials, :, :] * mask[good_trials, :, :]).mean(dim=1).squeeze()
    )
    decisions = torch.sign(
        (output[good_trials, :, :] * mask[good_trials, :, :]).mean(dim=1).squeeze()
    )
    return (target_decisions == decisions).type(torch.float32).mean()



class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        noise_std,
        alpha=0.2,
        rho=1,
        train_wi=False,
        train_wo=False,
        train_wrec=True,
        train_h0=False,
        wi_init=None,
        wo_init=None,
        wrec_init=None,
        si_init=None,
        so_init=None,
    ):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float, value of dt/tau
        :param rho: float, std of gaussian distribution for initialization
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param wrec_init: torch tensor of shape (hidden_size, hidden_size)
        :param si_init: input scaling, torch tensor of shape (input_dim)
        :param so_init: output scaling, torch tensor of shape (output_dim)
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rho = rho
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.non_linearity = torch.tanh

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if wrec_init is None:
                self.wrec.normal_(std=rho / sqrt(hidden_size))
            else:
                self.wrec.copy_(wrec_init)
            if wo_init is None:
                self.wo.normal_(std=1 / hidden_size)
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            self.h0.zero_()
        self.wi_full, self.wo_full = [None] * 2
        self._define_proxy_parameters()

    def _define_proxy_parameters(self):
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so

    def forward(self, input, return_dynamics=False):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(h)
        self._define_proxy_parameters()
        noise = torch.randn(
            batch_size, seq_len, self.hidden_size, device=self.wrec.device
        )
        output = torch.zeros(
            batch_size, seq_len, self.output_size, device=self.wrec.device
        )
        if return_dynamics:
            trajectories = torch.zeros(
                batch_size, seq_len, self.hidden_size, device=self.wrec.device
            )

        # simulation loop
        for i in range(seq_len):
            h = (
                h
                + self.noise_std * noise[:, i, :]
                + self.alpha
                * (-h + r.matmul(self.wrec.t()) + input[:, i, :].matmul(self.wi_full))
            )
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full)
            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    def clone(self):
        new_net = RNN(
            self.input_size,
            self.hidden_size,
            self.output_size,
            self.noise_std,
            self.alpha,
            self.rho,
            self.train_wi,
            self.train_wo,
            self.train_wrec,
            self.train_h0,
            self.wi,
            self.wo,
            self.wrec,
            self.si,
            self.so,
        )
        return new_net
