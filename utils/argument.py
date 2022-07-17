#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os
import os.path as osp
from logging import getLogger

import argparse

from models.baseline import ResNet34MeanProj, ResNet34LateFusion, ResNet34MeanProjNormalizedSoftmax, ResNet34SingleNormalizedSoftmax


logger = getLogger('main')

def get_args():

  parser = argparse.ArgumentParser()

  # Directory
  parser.add_argument('--root_image_dir', type=str, default="data")
  parser.add_argument('--root_data_dir', type=str, default="data")
  parser.add_argument('--out_dir', type=str, default="outputs")
  parser.add_argument('--dir_name', type=str, default="tmp")
  parser.add_argument('--nb_workers', type=int, default=12)

  # Model
  parser.add_argument('--model', type=str, default="")
  parser.add_argument('--method', type=str, default="arcface")
  parser.add_argument('--stride', type=int, default=2)

  # Dataset
  parser.add_argument('--dataset', type=str, default="epic")
  parser.add_argument('--nb_images', type=int, default=10)
  parser.add_argument('--nb_tracks', type=int, default=3)
  parser.add_argument('--eval_set', type=str, default="valid")
  parser.add_argument('--train_pids', type=str, default="configs/train_pids.txt")
  parser.add_argument('--eval_pids', type=str, default="configs/valid_pids.txt")
  parser.add_argument('--test_pids', type=str, default="configs/test_pids.txt")
  parser.add_argument('--split', type=str, default="train")
  parser.add_argument('--eval_split', type=str, default="valid")
  parser.add_argument('--strict_crop', action='store_true')

  # Training
  parser.add_argument('--nb_iters', type=int, default=25000)
  parser.add_argument('--iter_evaluation', type=int, default=2500)
  parser.add_argument('--iter_snapshot', type=int, default=2500)
  parser.add_argument('--iter_display', type=int, default=250)
  parser.add_argument('--iter_visualize', type=int, default=1000)
  parser.add_argument('-b', '--batch_size', type=int, default=256)
  parser.add_argument('--samples_per_class', type=int, default=1)
  parser.add_argument('--optimizer', type=str, default="adam")
  parser.add_argument('--min_lr', type=float, default=3e-8)
  parser.add_argument('--lr', type=float, default=3e-4)
  parser.add_argument('--lr_step_list', type=float, nargs="*", default=[6000, 9000])
  parser.add_argument('--momentum', type=float, default=0.99)

  parser.add_argument('--scratch', action='store_true')
  parser.add_argument('--track_per_class', type=int, default=2)
  parser.add_argument('--processed', action='store_true')
  parser.add_argument('--eval_joint', action='store_true')
  parser.add_argument('--freeze_backbone', action='store_true')
  parser.add_argument('--hard_mining', action='store_true')
  parser.add_argument('--freeze_layer4', action='store_true')
  parser.add_argument('--train_class_cnt', type=int, default=-1)

  # Normalized softmax and N-paired
  parser.add_argument('--tau', type=float, default=0.07)

  # ArcFace
  parser.add_argument('--s', type=float, default=30.0)
  parser.add_argument('--m', type=float, default=0.50)

  # Clustering
  parser.add_argument('--cluster_alg', type=str, default="kmeans")
  parser.add_argument('--cpu', action='store_true')
  parser.add_argument('--nb_dims', type=int, default=256)
  parser.add_argument('--eval_hac_dists', type=float, nargs="*", default=[0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.75, 0.8, 0.85])

  # Visualization
  parser.add_argument('--vis', action='store_true')
  parser.add_argument('--vis_dir_name', type=str, default="vis_predictions")
  parser.add_argument('--grad_cam', action='store_true')

  # Test
  parser.add_argument('--resume', type=str, default="")

  # Misc
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--model_debug', action='store_true')
  parser.add_argument('--save_model', action='store_true')
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--seed', type=int, default=1701)  # XXX

  args = parser.parse_args()

  return args


def get_model(args):
  if args.method == "softmax":
    base_model = eval(args.model)(num_classes=args.nb_dims, pretrained=not args.scratch, layer4_stride=args.stride, train_class_cnt=args.train_class_cnt, temperature=args.tau)
  else:
    base_model = eval(args.model)(num_classes=args.nb_dims, pretrained=not args.scratch, layer4_stride=args.stride)

  return base_model
