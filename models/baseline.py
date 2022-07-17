#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import _resnet, BasicBlock, Bottleneck


class ResNet34MeanProj(nn.Module):
  """
  "ArcFace/N-pair" baseline
  Mean fusion on within-track detections
  """
  def __init__(self, num_classes=128, pretrained=True, layer4_stride=2):
    super().__init__()
    self.backbone = _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained=pretrained, progress=True, layer4_stride=layer4_stride)
    self.fc = nn.Linear(512, num_classes)
    self.fc2 = nn.Linear(512, num_classes)

  def forward(self, x, batch_idxs):
    h = self.backbone(x, pool=False)
    h_pooled = torch.flatten(F.adaptive_avg_pool2d(h, 1), 1)

    # aggregate feature
    h_agg = torch.stack([torch.mean(h_pooled[batch_idxs == bidx], dim=0) for bidx in torch.unique(batch_idxs, sorted=True)])

    # L2 normalization
    h_track = self.fc(h_agg)
    h_track = F.normalize(h_track, p=2, dim=1)

    return {
      "feature_map": F.adaptive_avg_pool2d(h, 7),
      "features": h_track,
    }


class ResNet34MeanProjNormalizedSoftmax(nn.Module):
  """
  "Softmax" baseline
  Mean fusion on within-track detections
  Combined with normalized softmax loss
  """
  def __init__(self, num_classes=128, pretrained=True, layer4_stride=2, train_class_cnt=-1, temperature=1.0):
    super().__init__()
    self.backbone = _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained=pretrained, progress=True, layer4_stride=layer4_stride)
    self.fc = nn.Linear(512, num_classes)
    self.W = nn.Parameter(torch.Tensor(num_classes, train_class_cnt))
    torch.nn.init.normal_(self.W)

    self.temperature = temperature

  def forward(self, x, batch_idxs):
    h = self.backbone(x, pool=False)
    h_pooled = torch.flatten(F.adaptive_avg_pool2d(h, 1), 1)

    # aggregate feature
    h_agg = torch.stack([torch.mean(h_pooled[batch_idxs == bidx], dim=0) for bidx in torch.unique(batch_idxs, sorted=True)])

    # L2 normalization
    h_track = self.fc(h_agg)
    h_track_norm = F.normalize(h_track, p=2, dim=1)

    # L2 normalization on embedding dimension
    normalized_W = F.normalize(self.W, p=2, dim=0)
    # Dot product
    logits = torch.sum(h_track_norm.unsqueeze(2) * normalized_W.unsqueeze(0), dim=1) / self.temperature

    return {
      "features": h_track_norm,
      "logits": logits
    }


class ResNet34SingleNormalizedSoftmax(nn.Module):
  """
  Used for Grad-CAM visualiation
  """
  def __init__(self, num_classes=128, pretrained=True, layer4_stride=2, train_class_cnt=-1, temperature=1.0):
    super().__init__()
    self.backbone = _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained=pretrained, progress=True, layer4_stride=layer4_stride)
    self.fc = nn.Linear(512, num_classes)
    self.W = nn.Parameter(torch.Tensor(num_classes, train_class_cnt))

    self.temperature = temperature

  def forward(self, x):
    h = self.backbone(x, pool=True)

    # L2 normalization
    h = self.fc(h)
    h = F.normalize(h, p=2, dim=1)

    # L2 normalization on embedding dimension
    normalized_W = F.normalize(self.W, p=2, dim=0)
    # Dot product
    logits = torch.sum(h.unsqueeze(2) * normalized_W.unsqueeze(0), dim=1) / self.temperature

    return logits


class ResNet34LateFusion(nn.Module):
  """
  "ImageNet" baseline
  Late fusion on within-track detections
  """
  def __init__(self, num_classes=128, pretrained=True, layer4_stride=2):
    super().__init__()
    self.backbone = _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained=pretrained, progress=True)

  def forward(self, x, batch_idxs):
    h = self.backbone(x, pool=True)

    # aggregate feature
    h_agg = torch.stack([torch.mean(h[batch_idxs == bidx], dim=0) for bidx in torch.unique(batch_idxs, sorted=True)])

    h = F.normalize(h_agg, p=2, dim=1)

    return {
      "features": h
    }
