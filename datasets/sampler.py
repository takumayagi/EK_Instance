#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

from collections import Counter

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):
  """
  Sample from different pids
  """
  def __init__(self, labels, n_classes, n_samples):
    self.labels_set = np.setdiff1d(np.unique(labels), np.array([-1]))
    self.label_to_indices = {label: np.where(np.array(labels) == label)[0]
                             for label in self.labels_set}
    for k in self.label_to_indices.keys():
      if len(self.label_to_indices[k]) < n_samples:
        self.label_to_indices[k] = np.repeat(self.label_to_indices[k], (n_samples - 1) // len(self.label_to_indices[k]) + 1)
    for l in self.labels_set:
      np.random.shuffle(self.label_to_indices[l])
    self.used_label_indices_count = {label: 0 for label in self.labels_set}
    self.count = 0
    self.n_classes = n_classes
    self.n_samples = n_samples
    labels = np.array(labels)
    self.len_dataset = np.sum(labels != -1)
    self.batch_size = self.n_samples * self.n_classes

  def __iter__(self):
    self.count = 0
    while self.count + self.batch_size < self.len_dataset:
      classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
      indices = []
      for class_ in classes:
        st = self.used_label_indices_count[class_]
        indices.extend(self.label_to_indices[class_][
                       st:st+self.n_samples])
        self.used_label_indices_count[class_] += self.n_samples
        if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
          np.random.shuffle(self.label_to_indices[class_])
          self.used_label_indices_count[class_] = 0
      yield indices
      self.count += self.n_classes * self.n_samples

  def __len__(self):
    return self.len_dataset // self.batch_size


class ClassWiseSampler(BatchSampler):
  """
  Sample by class (pseudo-class)
  """
  def __init__(self, labels):
    self.labels_set = np.unique(labels)
    self.label_to_indices = {label: np.where(np.array(labels) == label)[0]
                             for label in self.labels_set}

  def __iter__(self):
    for k, v in sorted(self.label_to_indices.items()):
      yield v

  def __len__(self):
    return len(self.labels_set)


class ClassBalancedSampler(BatchSampler):
  """
  Sample by inverse frequency
  """
  def __init__(self, labels):
    self.labels = labels
    self.classes = list(sorted(np.unique(labels).tolist()))
    self.labels_by_class = [[idx for idx, x in enumerate(labels) if x == cidx] for cidx in self.classes]

  def __iter__(self):

    self.count = 0
    while self.count < len(self.labels):
      cidx = torch.randint(len(self.classes), (1,))
      labels = self.labels_by_class[cidx.item()]

      yield labels[torch.randint(len(labels), (1,)).item()]
      self.count += 1

  def __len__(self):
    return len(self.labels)


class ClassUniformSampler(BatchSampler):
  """
  Sample nb_sample image per class
  """
  def __init__(self, labels, nb_samples, batch_size):
    self.labels_set = np.unique(labels)
    self.label_to_indices = {label: np.where(np.array(labels) == label)[0]
                             for label in self.labels_set}
    self.nb_samples = nb_samples
    self.batch_size = batch_size
    self.nb_classes = len(self.labels_set)

  def __iter__(self):
    indices = []
    for idx, (k, class_indices) in enumerate(sorted(self.label_to_indices.items())):
      indices.append(np.random.choice(class_indices, self.nb_samples, replace=False))
      if (idx + 1) % self.batch_size  == 0:
        yield np.concatenate(indices).tolist()
        indices = []

    yield np.concatenate(indices).tolist()

  def __len__(self):
    return len(self.labels_set)


class HardClassMiningBatchSampler(BatchSampler):
  """
  Sample from different pids
  """
  def __init__(self, labels, class_labels, class_sims, n_classes, n_samples, ratio=1.0):
    self.labels_set = np.setdiff1d(np.unique(labels), np.array([-1]))
    self.label_to_indices = {label: np.where(np.array(labels) == label)[0]
                             for label in self.labels_set}
    for k in self.label_to_indices.keys():
      if len(self.label_to_indices[k]) < n_samples:
        self.label_to_indices[k] = np.repeat(self.label_to_indices[k], (n_samples - 1) // len(self.label_to_indices[k]) + 1)
    for l in self.labels_set:
      np.random.shuffle(self.label_to_indices[l])
    self.used_label_indices_count = {label: 0 for label in self.labels_set}
    self.count = 0
    self.class_labels = class_labels
    self.class_sims = class_sims
    self.n_classes = n_classes
    self.n_samples = n_samples
    labels = np.array(labels)
    self.len_dataset = np.sum(labels != -1)
    self.batch_size = self.n_samples * self.n_classes
    self.ratio = ratio

  def __iter__(self):
    self.count = 0
    while self.count + self.batch_size < self.len_dataset:
      key_class = np.random.choice(self.labels_set, 1)[0]
      class_idx = np.arange(len(self.class_labels))[self.class_labels == key_class][0]
      hard_class_idxs = np.argsort(self.class_sims[class_idx])[::-1]
      if self.ratio == 1.0:  # strictly hard
        hard_classes = self.class_labels[hard_class_idxs[:self.n_classes]]
      else:
        hard_classes = np.random.choice(self.class_labels[hard_class_idxs[:int(self.n_classes*self.ratio)]], self.n_classes, False)
      indices = []
      for class_ in hard_classes:
        st = self.used_label_indices_count[class_]
        indices.extend(self.label_to_indices[class_][
                       st:st+self.n_samples])
        self.used_label_indices_count[class_] += self.n_samples
        if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
          np.random.shuffle(self.label_to_indices[class_])
          self.used_label_indices_count[class_] = 0
      yield indices
      self.count += self.n_classes * self.n_samples

  def __len__(self):
    return self.len_dataset // self.batch_size


if __name__ == "__main__":
  sampler = ClassWiseSampler([0, 1, 0, 0, 1, 1, 2, 2, 2])
  for idxs in sampler:
    print(idxs)
