#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os
import os.path as osp
import sys
import json
import glob
from collections import Counter

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF


class SOPBaseDataset(Dataset):
  def __init__(self, args, root_image_dir, root_data_dir, train, pad=0.0, debug=False, im_width=224, im_height=224):

    self.root_image_dir = root_image_dir
    self.root_data_dir = root_data_dir
    self.train = train
    self.pad = pad
    self.debug = debug

    self.mean = [0.485, 0.456, 0.406]
    self.std = [0.229, 0.224, 0.225]

    self.image_size = (im_width, im_height)


class SOPBatchDataset(SOPBaseDataset):

  def __init__(self, args, root_image_dir, root_data_dir, split, train, pad=0.0, debug=False, im_width=224, im_height=224):
    super().__init__(args, root_image_dir, root_data_dir, train, pad, debug, im_width, im_height)

    self.data = []
    self.labels = []

    # XXX Prepare valiation split
    if split in ["trainval", "train", "valid"]:
      split_path = osp.join(root_data_dir, "Ebay_train.txt")
    elif split == "test":
      split_path = osp.join(root_data_dir, "Ebay_test.txt")
    else:
      raise NotImplementedError()

    with open(split_path) as f:
      for idx, line in enumerate(f):
        if idx == 0:
          continue
        _, class_id, _, impath = line.strip("\n").split(" ")
        if split == "valid" and int(class_id) % 5 != 0:
          continue
        if split == "train" and int(class_id) % 5 == 0:
          continue
        self.data.append(impath)
        self.labels.append(int(class_id))

    self.class_cnt = len(np.unique(self.labels))
    trans_dict = dict([[x, y] for y, x in enumerate(sorted(np.unique(self.labels)))])
    self.labels = [trans_dict[x] for x in self.labels]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idxs):
    """
    Read simgle image
    """
    imgs, labels, batch_idxs, vis_track_imgs, vis_imgs = [], [], [], [], []
    for bidx, i in enumerate(idxs):
      impath = osp.join(self.root_image_dir, self.data[i])

      img_full = Image.open(impath)
      if img_full.mode == "L":
        img_full = img_full.convert("RGB")
      iw, ih = img_full.size

      if self.train:
        rot_angle = 30 * torch.rand(1).item() - 15
        flip = torch.rand(1).item() > 0.5

        # Apply same transformation
        img = TF.rotate(img_full, rot_angle)

        ii, ij, ih, iw = transforms.RandomResizedCrop.get_params(img, scale=(0.85, 1.0), ratio=(0.75, 4/3))
        img = TF.resized_crop(img, ii, ij, ih, iw, self.image_size)
        if flip:
          img = TF.hflip(img)

        img = TF.pil_to_tensor(img).to(dtype=torch.float32).div(255)
        img = TF.normalize(img, self.mean, self.std)
      else:
        img = TF.resize(img_full, self.image_size)
        img = TF.pil_to_tensor(img).to(dtype=torch.float32).div(255)
        img = TF.normalize(img, self.mean, self.std)

      if not self.train:
        vis_img = np.array(TF.resize(img_full, self.image_size))
        vis_imgs.append(vis_img)
        vis_track_imgs.append(vis_img)

      imgs.append(img)
      batch_idxs.append(bidx)
      labels.append(self.labels[i])

    imgs = torch.stack(imgs)
    batch_idxs = torch.tensor(batch_idxs, dtype=int)
    labels = torch.tensor(labels, dtype=int)

    full_labels = labels[batch_idxs]
    lbls = full_labels.detach().cpu().numpy().tolist()
    unique_lbls = np.sort(np.unique(lbls)).tolist()
    class_idxs = torch.tensor([unique_lbls.index(x) for x in lbls], dtype=int)

    out_dict = {
      "imgs": imgs,
      "batch_idxs": batch_idxs,
      "labels": labels,
      "class_idxs": class_idxs
    }

    if not self.train:
      out_dict["vis_imgs"] = np.stack(vis_imgs)
      out_dict["vis_track_imgs"] = np.stack(vis_track_imgs)

    return out_dict
