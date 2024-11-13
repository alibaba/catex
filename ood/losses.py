# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn.functional as F

class LogitNormLoss(torch.nn.Module):
    def __init__(self, t=.07):
        super(LogitNormLoss, self).__init__()
        self.t = t
    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        loss = F.cross_entropy(logit_norm, target)

        return loss #* 50.