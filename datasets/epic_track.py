#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os.path as osp
import json
import glob

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms


class EPICBaseDataset(Dataset):
  def __init__(self, args, root_image_dir, root_data_dir, pid_list, train, pad=0.0, debug=False, im_width=224, im_height=224):

    self.root_image_dir = root_image_dir
    self.root_data_dir = root_data_dir
    self.train = train
    self.pad = pad
    self.debug = debug

    self.mean = [0.485, 0.456, 0.406]
    self.std = [0.229, 0.224, 0.225]
    self.image_size = (im_width, im_height)


class EPICTrackDataset(EPICBaseDataset):

  def __init__(self, args, root_image_dir, root_data_dir, pid_list, train, pad=0.0, debug=False, im_width=224, im_height=224):
    super().__init__(args, root_image_dir, root_data_dir, pid_list, train, pad, debug, im_width, im_height)

    self.data = []
    self.labels = []
    self.pseudo_labels = []

    self.min_size = 0
    self.class_cnt = 0
    self.nb_frames = 4

    self.pid_list = pid_list
    self.track_ids = []

    # Open annotations
    for pid in pid_list:
      anno_dir = osp.join(root_data_dir, pid)
      for anno_path in list(sorted(glob.glob(osp.join(anno_dir, "*.json")))):

        with open(anno_path) as f:
          vott_dict = json.load(f)

        # Because vott_dict does not have clear unit of "track" so split it here
        found = False
        dets, prev_frame_num, prev_vid = [], None, None
        for key, info in vott_dict["frames"].items():
          if len(info) == 0:
            continue
          vid = "_".join(key.split("_")[:2])
          frame_num = int(key.split("_")[2][:-4])

          if prev_vid is not None and (prev_vid != vid or frame_num - prev_frame_num > 60):
            self.data.append(dets)
            self.labels.append(self.class_cnt)
            found = True
            dets = []

          det = [vid, frame_num, info[0]["x1"], info[0]["y1"], info[0]["x2"], info[0]["y2"]]
          dets.append(det)
          prev_vid = vid
          prev_frame_num = frame_num

        if len(dets) > 0:
          self.data.append(dets)
          self.labels.append(self.class_cnt)
          found = True

        if found:
          self.track_ids.append(osp.splitext(osp.basename(anno_path))[0])
          self.class_cnt += 1

  def __len__(self):
    return len(self.labels)

  def _crop_padded_image(self, img_full, x1, y1, x2, y2, pad=False):

    if not pad:
      nx1, ny1, nx2, ny2 = x1, y1, x2, y2
    else:
      if x2 - x1 > y2 - y1:
        w = x2 - x1
        ny1 = max(0, int((y1 + y2) / 2 - w / 2))
        ny2 = min(y1 + w, img_full.height)
        nx1, nx2 = x1, x2
      else:
        h = y2 - y1
        nx1 = max(0, int((x1 + x2) / 2 - h / 2))
        nx2 = min(x1 + h, img_full.width)
        ny1, ny2 = y1, y2
    return img_full.crop((nx1, ny1, nx2, ny2)).resize((112, 112))

  def __getitem__(self, i):
    raise NotImplementedError()


class EPICTrackBatchDataset(EPICTrackDataset):
  """
  Batch-wise dataset
  Augmentation: random resized crop, random rotation, random horizontal flip
  Return positional mask
  """
  def __getitem__(self, idxs):
    """
    Aggregate detections
    """
    imgs, masks, labels, batch_idxs, vis_track_imgs, vis_imgs, pseudo_labels = [], [], [], [], [], [], []
    for bidx, i in enumerate(idxs):
      dets = self.data[i]
      if self.train and len(dets) > self.nb_frames:
        selected_idxs = torch.randperm(len(dets))[:self.nb_frames]
      else:
        selected_idxs = range(len(dets))

      for idx in selected_idxs:
        vid, frame_num, x1, y1, x2, y2 = dets[idx]
        impath = osp.join(self.root_image_dir, vid[:3], vid, f"frame_{frame_num:010d}.jpg")

        img_full = Image.open(impath)
        iw, ih = img_full.size
        img_mask = np.zeros((ih, iw), dtype=np.uint8)

        if self.train:
          w, h = x2 - x1, y2 - y1
          px1 = max(0, int(x1 - w * self.pad))
          py1 = max(0, int(y1 - h * self.pad))
          px2 = min(img_full.width, int(x2 + w * self.pad))
          py2 = min(img_full.height, int(y2 + h * self.pad))
          img_crop = img_full.crop((px1, py1, px2, py2))
          img_mask[y1:y2, x1:x2] = 1
          mask_crop = Image.fromarray(img_mask[py1:py2, px1:px2])

          rot_angle = 30 * torch.rand(1).item() - 15
          flip = torch.rand(1).item() > 0.5

          # Apply same transformation
          img_crop = TF.rotate(img_crop, rot_angle)
          mask_crop = TF.rotate(mask_crop, rot_angle)

          ii, ij, ih, iw = transforms.RandomResizedCrop.get_params(img_crop, scale=(0.85, 1.0), ratio=(0.75, 4/3))
          img_crop = TF.resized_crop(img_crop, ii, ij, ih, iw, self.image_size)
          mask_crop = TF.resized_crop(mask_crop, ii, ij, ih, iw, (14, 14), interpolation=transforms.InterpolationMode.NEAREST)

          if flip:
            img_crop = TF.hflip(img_crop)
            mask_crop = TF.hflip(mask_crop)

          img = TF.pil_to_tensor(img_crop).to(dtype=torch.float32).div(255)
          img = TF.normalize(img, self.mean, self.std)
          mask = TF.pil_to_tensor(mask_crop).bool()
        else:
          img_crop = img_full.crop((x1, y1, x2, y2))
          img_mask[y1:y2, x1:x2] = 1
          mask_crop = Image.fromarray(img_mask[y1:y2, x1:x2])

          img_crop = TF.resize(img_crop, self.image_size)
          mask_crop = TF.resize(mask_crop, self.image_size)

          img = TF.pil_to_tensor(img_crop).to(dtype=torch.float32).div(255)
          img = TF.normalize(img, self.mean, self.std)
          mask = TF.pil_to_tensor(mask_crop).bool()

        if not self.train:
          vis_img = np.array(self._crop_padded_image(img_full, x1, y1, x2, y2))
          vis_imgs.append(vis_img)
          if idx == 0:
            vis_track_imgs.append(vis_img)

        imgs.append(img)
        masks.append(mask)
        batch_idxs.append(bidx)
      labels.append(self.labels[i])
      if len(self.pseudo_labels) > 0:
        pseudo_labels.append(self.pseudo_labels[i])

    imgs = torch.stack(imgs)
    masks = torch.stack(masks)
    batch_idxs = torch.tensor(batch_idxs, dtype=int)
    labels = torch.tensor(labels, dtype=int)

    full_labels = labels[batch_idxs]
    lbls = full_labels.detach().cpu().numpy().tolist()
    unique_lbls = np.sort(np.unique(lbls)).tolist()
    class_idxs = torch.tensor([unique_lbls.index(x) for x in lbls], dtype=int)

    out_dict = {
      "imgs": imgs,
      "masks": masks,
      "batch_idxs": batch_idxs,
      "labels": labels,
      "class_idxs": class_idxs
    }

    if not self.train:
      out_dict["vis_imgs"] = np.stack(vis_imgs)
      out_dict["vis_track_imgs"] = np.stack(vis_track_imgs)

    if len(self.pseudo_labels) > 0:
      pseudo_labels = torch.tensor(pseudo_labels, dtype=int)
      full_labels = pseudo_labels[batch_idxs]
      lbls = full_labels.detach().cpu().numpy().tolist()
      unique_lbls = np.sort(np.unique(lbls)).tolist()
      pseudo_class_idxs = torch.tensor([unique_lbls.index(x) for x in lbls], dtype=int)
      out_dict["pseudo_labels"] = pseudo_labels
      out_dict["pseudo_class_idxs"] = pseudo_class_idxs

    return out_dict


class EPICTrackDatasetSingle(EPICTrackDataset):
  """
  Single-image version
  """
  def __getitem__(self, i):
    """
    Aggregate detections
    """
    bidx = 0

    imgs, masks, labels, batch_idxs, vis_imgs = [], [], [], [], []
    dets = self.data[i]
    idx = len(dets) // 2  # Take middle

    vid, frame_num, x1, y1, x2, y2 = dets[idx]
    impath = osp.join(self.root_image_dir, vid[:3], vid, f"frame_{frame_num:010d}.jpg")

    img_full = Image.open(impath)
    iw, ih = img_full.size
    img_mask = np.zeros((ih, iw), dtype=np.uint8)

    img_crop = img_full.crop((x1, y1, x2, y2))
    img_mask[y1:y2, x1:x2] = 1
    mask_crop = Image.fromarray(img_mask[y1:y2, x1:x2])

    img_crop = TF.resize(img_crop, self.image_size)
    mask_crop = TF.resize(mask_crop, self.image_size)

    img = TF.pil_to_tensor(img_crop).to(dtype=torch.float32).div(255)
    img = TF.normalize(img, self.mean, self.std)
    mask = TF.pil_to_tensor(mask_crop).bool()

    vis_img = np.array(img_full.crop((x1, y1, x2, y2)).resize((224, 224)))
    vis_imgs.append(vis_img)

    imgs.append(img)
    masks.append(mask)
    batch_idxs.append(bidx)
    labels.append(self.labels[i])

    imgs = torch.stack(imgs)
    masks = torch.stack(masks)
    batch_idxs = torch.tensor(batch_idxs, dtype=int)
    labels = torch.tensor(labels, dtype=int)

    full_labels = labels[batch_idxs]
    lbls = full_labels.detach().cpu().numpy().tolist()
    unique_lbls = np.sort(np.unique(lbls)).tolist()
    class_idxs = torch.tensor([unique_lbls.index(x) for x in lbls], dtype=int)

    out_dict = {
      "imgs": imgs,
      "masks": masks,
      "batch_idxs": batch_idxs,
      "labels": labels,
      "class_idxs": class_idxs
    }

    out_dict["vis_imgs"] = np.stack(vis_imgs)

    return out_dict
