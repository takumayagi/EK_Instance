#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os.path as osp
import sys
import json
import glob
from collections import Counter

import numpy as np
import pickle as pkl
import cv2

from torch.utils.data import Dataset


known_class_ids = [2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 16, 17, 18, 20, 23, 24, 25, 26, 28, 29, 31, 33, 34, 35, 36, 39, 40, 41, 44, 45, 46, 47, 48]
novel_class_ids = [0, 1, 9, 10, 12, 19, 21, 22, 27, 30, 32, 37, 38, 42, 43, 49]

query_session = 1
gallery_sessions = [3, 7, 10]


class CORE50BaseDataset(Dataset):
  def __init__(self, data_dir, train, im_width=224, im_height=224, transform=True, **kwargs):
    self.data_dir = data_dir
    self.train = train
    self.im_width, self.im_height = im_width, im_height

    with open(osp.join(data_dir, "core50_class_names.txt")) as f:
      self.class_names = [x.strip("\n") for x in f.readlines()]

    with open(osp.join(data_dir, "paths.pkl"), 'rb') as f:
      self.paths = pkl.load(f)

    self.transform = transform
    self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    #self.mean = np.array([104, 117, 128], dtype=np.float32)
    #self.std = np.array([1, 1, 1], dtype=np.float32)
    self.debug = False

  def __len__(self):
    return len(self.paths)

  def _transform(self, img):
    img = cv2.resize(img, (self.im_width, self.im_height))

    if not self.debug:
      img = img.astype(np.float32)

      # Horizontal flip
      if self.train and np.random.random() > 0.5:
        img = img[:, ::-1, :]

      img = (img / 255. - self.mean) / self.std
      #img = (img - self.mean) / self.std
      img = img.transpose((2, 0, 1))

    return img

  def __getitem__(self, i):
    pass


class CORE50SessionDataset(CORE50BaseDataset):
  """
  Training: 1, 2, 4, 5, 6, 8, 9, 11
  Test: 3, 7, 10
  """
  def __init__(self, data_dir, eval_set, train, **kwargs):
    super().__init__(data_dir, train, **kwargs)

    self.valid_sessions = ["s1", "s2", "s4", "s5", "s6", "s8", "s9", "s11"] if eval_set == "train" else ["s3", "s7", "s10"]
    self.paths = [x for x in self.paths if x.split("/")[0] in self.valid_sessions]
    self.instance_idxs = [int(x.split("/")[1][1:]) - 1 for x in self.paths]

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, i):
    """
    Return (image, label)
    """
    img = cv2.imread(osp.join(self.data_dir, "core50_128x128", self.paths[i]))

    if self.transform:
      img = self._transform(img)

    return {
      "img": img,
      "gt_class": self.instance_idxs[i]
    }


class CORE50ObjectDataset(CORE50BaseDataset):
  """
  Object-wise split
  """
  def __init__(self, data_dir, eval_set, train, **kwargs):
    super().__init__(data_dir, train, **kwargs)

    if eval_set == "train":
      self.valid_objects = ["o{}".format(x+1) for x in known_class_ids]
    elif eval_set == "test":
      self.valid_objects = ["o{}".format(x+1) for x in novel_class_ids]
    else:
      self.valid_objects = ["o{}".format(x+1) for x in range(50)]

    self.paths = [x for x in self.paths if x.split("/")[1] in self.valid_objects]
    self.labels = [int(x.split("/")[1][1:]) - 1 for x in self.paths]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, i):
    """
    Return (image, label)
    """
    img = cv2.imread(osp.join(self.data_dir, "core50_128x128", self.paths[i]))

    if self.transform:
      img = self._transform(img)

    return {
      "img": img,
      "gt_class": self.instance_idxs[i]
    }


class CORE50ObjectBatchDataset(CORE50BaseDataset):
  """
  Object-wise split
  Batch, track
  """
  def __init__(self, data_dir, eval_set, train, **kwargs):
    super().__init__(data_dir, train, **kwargs)

    self.paths = []
    self.labels = []
    for sid in range(1, 11, 1):
      for oid in range(1, 51, 1):
        if eval_set == "train" and oid-1 not in known_class_ids:
          continue
        if eval_set == "test" and oid-1 not in novel_class_ids:
          continue
        paths = glob.glob(osp.join(data_dir, "core50_128x128", f"s{sid}", f"o{oid}", "*.png"))
        # Take 1fps image
        paths = paths[::20]
        self.paths.append([osp.join(f"s{sid}", f"o{oid}", osp.basename(x)) for x in paths])
        self.labels.append(oid - 1)

        # Take 1st image
        #paths = paths[0:1]
        #self.paths.append([osp.join(f"s{sid}", f"o{oid}", osp.basename(x)) for x in paths])
        #self.labels.append(oid - 1)

        # individual clustering
        #for pth in paths[::20]:
        #  self.paths.append([osp.join(f"s{sid}", f"o{oid}", osp.basename(pth))])
        #  self.labels.append(oid - 1)

    self.class_cnt = len(np.unique(self.labels))

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idxs):

    imgs, labels, batch_idxs = [], [], []
    vis_imgs, vis_track_imgs = [], []

    for bidx, i in enumerate(idxs):
      paths = self.paths[i]
      for idx, impath in enumerate(paths):
        img = cv2.imread(osp.join(self.data_dir, "core50_128x128", impath))[..., ::-1]
        #img = img[32:-32, 32:-32]
        img = img[24:-24, 24:-24]  # (128, 128) -> (80, 80)

        vis_img = cv2.resize(img, (112, 112))
        vis_imgs.append(vis_img)
        if idx == 0:
          vis_track_imgs.append(vis_img)

        if self.transform:
          img = self._transform(img)

        imgs.append(img)
        batch_idxs.append(bidx)
      labels.append(self.labels[i])

    imgs = np.stack(imgs, axis=0)
    batch_idxs = np.array(batch_idxs)
    labels = np.array(labels)
    vis_imgs = np.stack(vis_imgs, axis=0)
    vis_track_imgs = np.stack(vis_track_imgs, axis=0)

    return {
      "imgs": imgs,
      "labels": labels,
      "batch_idxs": batch_idxs,
      "class_idxs": batch_idxs,
      "vis_imgs": vis_imgs,
      "vis_track_imgs": vis_track_imgs,
    }


class CORE50SiameseDataset(CORE50BaseDataset):
  """
  Generate Siamese pair
  """
  def __init__(self, data_dir, train, sessions=[1, 2, 4, 5, 6, 8, 9, 11], **kwargs):
    super().__init__(data_dir, train, **kwargs)

    trans_dict = dict(zip(sessions, range(len(sessions))))

    session_names = ["s{}".format(x) for x in sessions]
    self.target_class_ids = ["o{}".format(x+1) for x in known_class_ids]
    self.paths = [x for x in self.paths if x.split("/")[1] in self.target_class_ids and x.split("/")[0] in session_names]
    self.class_ids = [trans_dict[int(x.split("/")[1][1:])] for x in self.paths]

    print(len(self.paths))
    print(Counter(self.class_ids))

  def __len__(self):
    return len(self.paths)  # 一応

  def __getitem__(self, i):

    impath1 = self.paths[i]
    instance_paths = [x for x in self.paths if x.split("/")[1] == "o{}".format(self.class_ids[i]+1)]
    impath2 = np.random.choice(instance_paths)

    img1 = cv2.imread(osp.join(self.data_dir, "core50_128x128", impath1))
    img2 = cv2.imread(osp.join(self.data_dir, "core50_128x128", impath2))

    #cv2.imwrite("tmp/train_debug/img_pos_{}.jpg".format(idx), np.concatenate((img1, img2), axis=1))

    if self.transform:
      img1 = self._transform(img1)
      img2 = self._transform(img2)

    return {
      "img1": img1,
      "img2": img2,
      "label": self.class_ids[i]
    }


class CORE50VerificationDataset(CORE50BaseDataset):
  def __init__(self, data_dir, train, image_set="query", query_session=1, gallery_sessions=[3, 7, 10], im_width=224, im_height=224, **kwargs):
    super().__init__(data_dir, train, im_width, im_height, **kwargs)

    self.target_objects = ["o{}".format(x+1) for x in range(50)]
    #self.target_objects = ["o{}".format(x+1) for x in novel_class_ids]
    #self.target_objects = ["o{}".format(x+1) for x in range(25, 50, 1)]  # Last 25 objects
    self.query_session = "s{}".format(query_session)
    self.gallery_sessions = ["s{}".format(s) for s in gallery_sessions]

    if image_set == "query":
      self.image_paths = [x for x in self.paths if x.split("/")[0] == self.query_session and x.split("/")[1] in self.target_objects and
                          osp.splitext(x.split("/")[2].split("_")[-1])[0] == "140"]
      self.image_idxs = [int(x.split("/")[1][1:]) - 1 for x in self.image_paths]
    elif image_set == "gallery":
      self.image_paths = [x for x in self.paths if x.split("/")[0] in self.gallery_sessions and x.split("/")[1] in self.target_objects and
                    osp.splitext(x.split("/")[2].split("_")[-1])[0] == "140"]
      self.image_idxs = [int(x.split("/")[1][1:]) - 1 for x in self.image_paths]
    else:
      raise NotImplementedError()

    print(len(self.image_paths))

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, i):
    img = cv2.imread(osp.join(self.data_dir, "core50_128x128", self.image_paths[i]))
    if self.transform:
      img = self._transform(img)

    return {
      "img": img,
      "label": self.image_idxs[i]
    }



if __name__ == "__main__":
  """
  dataset = CORE50RecognitionDataset("/work/yagi/datasets/core50", "train", True)
  print(dataset[0])
  print(len(dataset))
  dataset = CORE50RecognitionDataset("/work/yagi/datasets/core50", "test", True)
  print(len(dataset))
  dataset = CORE50IncrementalBatchDataset("/work/yagi/datasets/core50", "train", True)
  print(dataset.gt_classes)
  """
  dataset = CORE50VerificationDataset("/work/yagi/datasets/core50", "train", True)
  dct = dataset[0]

  dataset = CORE50SiameseDataset("/work/yagi/datasets/core50", "train", True)
  dct = dataset[0]


