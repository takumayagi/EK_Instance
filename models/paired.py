#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NTXentLoss(torch.nn.Module):
    """
    Allow multiple positive example
    """
    def __init__(self, device, batch_size, temperature, pos_per_class):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.pos_per_class = pos_per_class

        self.pos_mask = self._get_correlated_mask().type(torch.bool)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        self.cnt = 0

    def _get_correlated_mask(self):
        return torch.block_diag(*[(torch.ones((self.pos_per_class, self.pos_per_class)) - torch.eye(self.pos_per_class)) for _ in range(self.batch_size // self.pos_per_class)])

    def forward(self, similarity_matrix, gt_labels):
        # filter out the scores from the positive samples
        positives = similarity_matrix[self.pos_mask].view(self.batch_size, -1)
        neg_mask = ~self.pos_mask * ~torch.eye(self.batch_size).bool()
        negatives = similarity_matrix[neg_mask].view(self.batch_size, -1)

        logits = torch.cat((torch.mean(positives, dim=1, keepdims=True), negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / ((self.pos_per_class - 1) * self.batch_size)
