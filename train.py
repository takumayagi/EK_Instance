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
import time
import datetime
from collections import Counter
import shutil
import warnings
import subprocess
warnings.simplefilter('ignore')

import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler

import faiss

from utils.argument import get_args, get_model

from models.metrics import ArcMarginProduct
from models.paired import NTXentLoss

from datasets.core50 import CORE50ObjectDataset, CORE50ObjectBatchDataset
from datasets.epic_track import EPICTrackBatchDataset, EPICTrackDatasetSingle
from datasets.sop import SOPBatchDataset
from datasets.sampler import BalancedBatchSampler, ClassUniformSampler, HardClassMiningBatchSampler
from utils.eval_utils import calc_results

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from mllogger import MLLogger


def hac(args, dist_list, nb_clusters, all_features, gt_labels, pid=None, sim_matrix=None):

  pr_labels_list = []
  for dist in dist_list:
    print(dist, end=" ")
    sys.stdout.flush()
    if sim_matrix is None:
      ag = AgglomerativeClustering(n_clusters=None, affinity="cosine", linkage="average", distance_threshold=dist, compute_full_tree=True)
      pr_labels = ag.fit_predict(all_features)
    else:
      ag = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="average", distance_threshold=dist, compute_full_tree=True)
      pr_labels = ag.fit_predict(1 - sim_matrix)

    wrong_links = []
    label_dict = dict([(idx, [gt_label]) for idx, gt_label in enumerate(gt_labels)])
    child_dict = dict([(idx, [idx]) for idx in range(len(pr_labels))])
    for idx, (x1, x2) in enumerate(ag.children_):
      if ag.distances_[idx] > dist:
        break
      child_dict[idx + len(pr_labels)] = child_dict[x1] + child_dict[x2]
      label_dict[idx + len(pr_labels)] = label_dict[x1] + label_dict[x2]
      gt_label1 = np.argmax(np.bincount(label_dict[x1]))
      gt_label2 = np.argmax(np.bincount(label_dict[x2]))
      if gt_label1 != gt_label2:
        wrong_links.append((x1, x2, idx, child_dict[x1], child_dict[x2], gt_label1, gt_label2, ag.distances_[idx]))

    pr_labels_list.append(pr_labels)

  print([len(np.unique(pr_labels)) for pr_labels in pr_labels_list])
  if pid is not None:
    return pr_labels_list, wrong_links
  else:
    return pr_labels_list


def kmeans(args, seeds, nb_clusters, all_features, gt_labels):
  """
  Spherical K-means using faiss
  """
  if args.cpu:
    index = faiss.IndexFlatIP(args.nb_dims)
  else:
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res, args.nb_dims, flat_config)

  pr_labels_list, dists_list = [], []
  for sidx, seed in enumerate(seeds):
    km = faiss.Clustering(args.nb_dims, nb_clusters)
    km.niter = 100 if args.eval else 20
    km.spherical = True
    km.min_points_per_centroid = 1
    km.max_points_per_centroid = 10000
    km.seed = seed
    km.train(all_features, index)
    _, pr_labels = index.search(all_features, 1)
    pr_labels = pr_labels[:, 0]
    pr_labels_list.append(pr_labels)

    centroids = faiss.vector_to_array(km.centroids).reshape((nb_clusters, -1))
    dists = 1 - np.sum(all_features * centroids[pr_labels], axis=1)
    dists_list.append(dists)

  if not args.cpu:
    del res
  del index

  if len(seeds) == 1:
    return pr_labels_list[0], dists_list[0]
  else:
    return pr_labels_list, dists_list


def eval_cam(args, device, save_dir, model, train_dataset, test_dataset, logger):
  """
  Visualize by Grad-CAM
  """
  model.eval()
  target_layers = [model.backbone.layer4[-1]]
  cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

  data_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.nb_workers)

  logger.info("Loading images...")
  class_img_dict, class_var_dict = {}, {}
  for idx, batch in enumerate(data_loader):
    label = batch["labels"].to(device, non_blocking=True)[0].item()
    vis_img = batch["vis_imgs"][0][0].numpy()
    img_gray = cv2.cvtColor(vis_img, cv2.COLOR_RGB2GRAY)
    var = cv2.Laplacian(img_gray, cv2.CV_32F).var()
    if label not in class_var_dict or var > class_var_dict[label]:
      class_img_dict[label] = vis_img
      class_var_dict[label] = var

  data_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.nb_workers)

  logger.info("Visualization...")
  for idx, batch in enumerate(data_loader):
    if idx % 10 != 0:
      continue

    x = batch["imgs"].to(device, non_blocking=True)[0]
    label = batch["labels"].to(device, non_blocking=True)[0]
    vis_img = batch["vis_imgs"][0][0].numpy() / 255.
    vid = test_dataset.data[idx][0][0]
    target_category = label.item()

    logits = model(x)
    sorted_idxs = torch.argsort(logits[0], descending=True).cpu().numpy()
    hard_idxs = sorted_idxs[sorted_idxs != target_category][:5]
    out_imgs = []

    grayscale_cam = cam(input_tensor=x, target_category=target_category)[0]
    vis_cam = show_cam_on_image(vis_img, grayscale_cam, use_rgb=True)
    #out_img = np.concatenate((batch["vis_imgs"][0][0].numpy()[..., ::-1], vis_cam[..., ::-1]), axis=1)
    #cv2.imwrite(osp.join(save_dir, f"{idx:04d}_{vid}.jpg"), out_img)
    out_imgs.append(batch["vis_imgs"][0][0].numpy())
    out_imgs.append(vis_cam)
    for hard_idx in hard_idxs:
      grayscale_cam = cam(input_tensor=x, target_category=hard_idx.item())[0]
      vis_cam = show_cam_on_image(vis_img, grayscale_cam, use_rgb=True)
      hard_img = np.copy(class_img_dict[hard_idx.item()])
      #cv2.putText(hard_img, str(hard_idx.item()), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_4)
      out_imgs.append(hard_img)
      out_imgs.append(vis_cam)
    cv2.imwrite(osp.join(save_dir, f"{idx:04d}_{target_category}_{vid}.jpg"), cv2.resize(np.concatenate(out_imgs, axis=1)[..., ::-1], None, fx=0.5, fy=0.5))

  vis_tar_path = save_dir + ".tar"
  logger.info(f"Creating tar archive: {vis_tar_path}")
  subprocess.run(["tar", "cf", vis_tar_path, "-C", osp.dirname(save_dir), osp.basename(save_dir)])

  logger.info("Done.")


def eval_net(args, device, save_dir, model, datasets, logger):

  """
  Evaluate clustering
  """
  model.eval()
  if args.eval:
    tmp_out_dir = osp.join("tmp", f"{args.model}_{args.dir_name}")
    if osp.exists(tmp_out_dir):
      shutil.rmtree(tmp_out_dir)
    os.makedirs(tmp_out_dir, exist_ok=True)

  if type(datasets) != list:
    datasets = [datasets]

  if args.vis:
    vis_dir = osp.join(save_dir, args.vis_dir_name)
    if osp.exists(vis_dir):
      shutil.rmtree(vis_dir)

  class_offset = 0
  amis, accs, homs, coms, vs, fps, fbs, nb_cluster_list = [], [], [], [], [], [], [], []
  pids = []
  for didx, dataset in enumerate(datasets):
    pid = dataset.pid_list[0] if args.dataset == "epic" else "test"
    pids.append(pid)

    with torch.no_grad():
      test_sampler = BatchSampler(SequentialSampler(range(len(dataset))), batch_size=args.batch_size, drop_last=False)

      data_loader = DataLoader(dataset, batch_size=1 if test_sampler is not None else args.batch_size, num_workers=args.nb_workers, sampler=test_sampler)

      purity_dict, i_purity_dict = {}, {}

      # Main
      all_features, all_labels, all_det_labels = [], [], []
      all_track_imgs, all_det_imgs = [], []
      for cnt, batch in enumerate(data_loader):
        x = batch["imgs"].to(device, non_blocking=True)[0]
        batch_idxs = batch["batch_idxs"].to(device, non_blocking=True)[0]
        labels = batch["labels"].to(device, non_blocking=True)[0]
        vis_imgs = batch["vis_imgs"][0]

        out_dict = model(x, batch_idxs)
        features = out_dict["features"]

        if args.method == "paired":
          features = F.normalize(features, dim=1, p=2)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if args.vis:
          all_track_imgs.append(batch["vis_track_imgs"][0].numpy())
        if args.eval:
          all_det_labels.append(labels[batch_idxs].cpu().numpy())
          all_det_imgs.append(vis_imgs.numpy())

      class_offset += dataset.class_cnt

      del features
      torch.cuda.empty_cache()

      all_features = np.concatenate(all_features, axis=0)
      gt_labels = np.concatenate(all_labels, axis=0)

      nb_clusters = dataset.class_cnt
      logger.info("Number of clusters: {}".format(nb_clusters))

      wrong_links = None
      if args.cluster_alg == "hac":
        pr_labels_list, wrong_links = hac(args, args.eval_hac_dists, nb_clusters, all_features, gt_labels, pid=pid)
      elif args.cluster_alg == "kmeans":
        # Take average on fixed random seeds
        seeds = [37, 123, 1234, 1701, 1901]
        pr_labels_list, _ = kmeans(args, seeds, nb_clusters, all_features, gt_labels)
      else:
        raise NotImplementedError()

      for pr_labels in pr_labels_list:
        result_dict = calc_results(gt_labels, pr_labels)
        amis.append(result_dict["ami"])
        accs.append(result_dict["acc"])
        homs.append(result_dict["hom"])
        coms.append(result_dict["com"])
        vs.append(result_dict["v_measure"])
        fps.append(result_dict["fp"])
        fbs.append(result_dict["fb"])
        nb_cluster_list.append(len(np.unique(pr_labels)))

      for k in sorted(np.unique(pr_labels)):
        gt = gt_labels[pr_labels == k]
        max_cnt = Counter(gt).most_common()[0][1]
        purity = max_cnt / len(gt)
        purity_dict[k] = purity

        i_purity = max([np.sum(gt == k2) / np.sum(gt_labels == k2) for k2 in np.unique(gt)])
        i_purity_dict[k] = i_purity

    # Visualization if needed
    if args.vis:

      all_track_imgs = np.concatenate(all_track_imgs, axis=0)

      vis_dir0 = osp.join(vis_dir, f"wr_{pid}")
      os.makedirs(vis_dir0, exist_ok=True)

      # Visualize wrong linkage
      for ch1, ch2, pa, idxs1, idxs2, gt1, gt2, feature_dist in wrong_links:
        out_path = osp.join(vis_dir0, f"{pa}_{gt1}_{gt2}_{feature_dist:.4f}.jpg")
        cluster_img1 = np.ones((112*((len(idxs1) - 1) // 10 + 1) + 112 // 2, 112*10, 3), dtype=np.uint8) * 255
        for pos, idx in enumerate(idxs1):
          pr_label = pr_labels[idx]
          gt_label = gt_labels[idx]
          img = all_track_imgs[idx]
          y1, x1 = (pos // 10) * 112, (pos % 10) * 112
          cluster_img1[y1:y1+112, x1:x1+112] = img

        cluster_img2 = np.ones((112*((len(idxs2) - 1) // 10 + 1) + 112 // 2, 112*10, 3), dtype=np.uint8) * 255
        for pos, idx in enumerate(idxs2):
          pr_label = pr_labels[idx]
          gt_label = gt_labels[idx]
          img = all_track_imgs[idx]
          y1, x1 = (pos // 10) * 112, (pos % 10) * 112
          cluster_img2[y1:y1+112, x1:x1+112] = img

        cluster_img = np.concatenate((cluster_img1, cluster_img2), axis=0)
        Image.fromarray(cluster_img).save(out_path)

      # Visualization by predicted cluster
      vis_dir1 = osp.join(vis_dir, f"pr_{pid}")
      os.makedirs(vis_dir1, exist_ok=True)

      for k in sorted(np.unique(pr_labels)):
        dir_name = osp.join(vis_dir1, f"{k}_{purity_dict[k]:.3f}_{i_purity_dict[k]:.3f}")
        os.makedirs(dir_name, exist_ok=True)
        idxs = np.arange(len(all_track_imgs))[pr_labels == k]
        cluster_img = np.ones((112*((len(idxs) - 1) // 10 + 1), 112*10, 3), dtype=np.uint8) * 255
        for i, idx in enumerate(idxs):
          pr_label = pr_labels[idx]
          gt_label = gt_labels[idx]
          img = all_track_imgs[idx]
          Image.fromarray(img).save(osp.join(dir_name, f"{idx:03d}_{gt_label}.jpg"))
          y1, x1 = (i // 10) * 112, (i % 10) * 112
          cluster_img[y1:y1+112, x1:x1+112] = img
        Image.fromarray(cluster_img).save(dir_name + ".jpg")

      # Visualization by gt cluster
      vis_dir2 = osp.join(vis_dir, f"gt_{pid}")
      os.makedirs(vis_dir2, exist_ok=True)

      for k in sorted(np.unique(gt_labels)):
        dir_name = osp.join(vis_dir2, f"{k}")
        idxs = np.arange(len(all_track_imgs))[gt_labels == k]
        nb_clusters = len(np.unique(pr_labels[gt_labels == k]))
        cluster_imgs = []
        for k2 in sorted(np.unique(pr_labels[gt_labels == k])):
          idxs = np.arange(len(all_track_imgs))[(gt_labels == k) * (pr_labels == k2)]
          cluster_img = np.ones((112*((len(idxs) - 1) // 10 + 1) + 112 // 2, 112*10, 3), dtype=np.uint8) * 255
          for pos, idx in enumerate(idxs):
            pr_label = pr_labels[idx]
            gt_label = gt_labels[idx]
            img = all_track_imgs[idx]
            y1, x1 = (pos // 10) * 112, (pos % 10) * 112
            cluster_img[y1:y1+112, x1:x1+112] = img
          cluster_imgs.append(cluster_img)
        Image.fromarray(np.concatenate(cluster_imgs, axis=0)).save(dir_name + ".jpg")

  if args.cluster_alg == "hac":
    nb_params = len(args.eval_hac_dists)
    logger.info("\tDist\tAMI\tACC\tHOM\tCOM\tNMI\tFp\tFb\t#CLS")
    for idx, (ami, acc, hom, com, v, fp, fb, nb_clusters) in enumerate(zip(amis, accs, homs, coms, vs, fps, fbs, nb_cluster_list)):
      logger.info(f"{pids[idx//nb_params]} {idx%nb_params+1}\t{args.eval_hac_dists[idx%nb_params]:.3f}\t{ami:.3f}\t{acc:.3f}\t{hom:.3f}\t{com:.3f}\t{v:.3f}\t{fp:.3f}\t{fb:.3f}\t{nb_clusters}")
    for idx, eval_dist in enumerate(args.eval_hac_dists):
      logger.info("Mean\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(
        eval_dist,
        np.mean([amis[tidx] for tidx in range(idx, len(amis), nb_params)]),
        np.mean([accs[tidx] for tidx in range(idx, len(amis), nb_params)]),
        np.mean([homs[tidx] for tidx in range(idx, len(amis), nb_params)]),
        np.mean([coms[tidx] for tidx in range(idx, len(amis), nb_params)]),
        np.mean([vs[tidx] for tidx in range(idx, len(amis), len(args.eval_hac_dists))]),
        np.mean([fps[tidx] for tidx in range(idx, len(amis), len(args.eval_hac_dists))]),
        np.mean([fbs[tidx] for tidx in range(idx, len(amis), len(args.eval_hac_dists))]),
      ))
  else:
    logger.info("\tAMI\tACC\tHOM\tCOM\tNMI\tFp\tFb")
    for idx, (ami, acc, hom, com, v, fp, fb) in enumerate(zip(amis, accs, homs, coms, vs, fps, fbs)):
      logger.info(f"{datasets[idx//len(seeds)].pid_list[0]} {idx%len(seeds)+1}\t{ami:.3f}\t{acc:.3f}\t{hom:.3f}\t{com:.3f}\t{v:.3f}\t{fp:.3f}\t{fb:.3f}")
    logger.info(f"Mean\t{np.mean(amis):.3f}\t{np.mean(accs):.3f}\t{np.mean(homs):.3f}\t{np.mean(coms):.3f}\t{np.mean(vs):.3f}\t{np.mean(fps):.3f}\t{np.mean(fbs):.3f}")

  if args.eval:
    if args.vis:
      vis_tar_path = save_dir + ".tar"
      logger.info(f"Creating tar archive: {vis_tar_path}")
      subprocess.run(["tar", "cf", vis_tar_path, "-C", osp.dirname(save_dir), osp.basename(save_dir)])

    logger.info("Done.")

  model.train()


def main():
  args = get_args()
  ngpus_per_node = torch.cuda.device_count()
  # Simply call main_worker function
  args.gpu = 0
  main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

  args.gpu = gpu
  start_time = time.time()

  if args.resume != "" and args.model == "":
    args.model = osp.basename(osp.dirname(args.resume)).split("_")[1]
    args.method = osp.basename(osp.dirname(args.resume)).split("_")[0]

  # Logger
  logger = MLLogger(init=False)
  dir_name = "{}_{}_{}_{}_{}".format(args.method, args.model, args.dataset, args.dir_name, datetime.datetime.now().strftime('%y%m%d'))
  if args.eval and args.out_dir == "outputs":
    args.out_dir = "predictions"
  logger.initialize(args.out_dir, dir_name)
  logger.info(vars(args))
  save_dir = logger.get_savedir()
  logger.info("Written to {}".format(save_dir))

  device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")

  # XXX Get valid dataset
  if args.dataset == "epic":
    with open(args.train_pids) as f:
      train_pid_list = [x.strip("\n") for x in f.readlines()]
    with open(args.test_pids if args.eval else args.eval_pids) as f:
      valid_pid_list = [x.strip("\n") for x in f.readlines()]
    image_dir = osp.join(args.root_image_dir, "epic_kitchens", "rgb_frames")
    anno_dir = osp.join(args.root_data_dir, "vott")
    #if args.processed:
    #  anno_dir = osp.join(args.root_data_dir, "vott_final")

    if args.grad_cam:
      valid_dataset = [EPICTrackDatasetSingle(args, image_dir, anno_dir, pid_list=valid_pid_list, train=False)]
    elif args.eval_joint:
      valid_dataset = [EPICTrackBatchDataset(args, image_dir, anno_dir, pid_list=valid_pid_list, train=False)]
    else:
      valid_dataset = [EPICTrackBatchDataset(args, image_dir, anno_dir, pid_list=[valid_pid], train=False) for valid_pid in valid_pid_list]
  elif args.dataset == "sop":
    image_dir = osp.join(args.root_image_dir, "Stanford_Online_Products")
    anno_dir = osp.join(args.root_data_dir, "Stanford_Online_Products")
    valid_dataset = [SOPBatchDataset(args, image_dir, anno_dir, split=args.eval_split, train=False)]
  elif args.dataset == "core50":
    image_dir = osp.join(args.root_image_dir, "core50")
    valid_dataset = [CORE50ObjectBatchDataset(image_dir, "all", train=False)]
  else:
    raise NotImplementedError()

  for dataset in valid_dataset:
    print(dataset.class_cnt, len(dataset))

  # XXX Get train dataset
  if not args.eval:
    if args.dataset == "epic":
      train_dataset = EPICTrackBatchDataset(args, image_dir, anno_dir, pid_list=train_pid_list, train=True)
      args.train_class_cnt = train_dataset.class_cnt
    elif args.dataset == "sop":
      train_dataset = SOPBatchDataset(args, image_dir, anno_dir, split=args.split, train=True)
    else:
      raise NotImplementedError()
    print(train_dataset.class_cnt, len(train_dataset))
  else:
    if args.method == "softmax":
      train_dataset = EPICTrackDatasetSingle(args, image_dir, anno_dir, pid_list=train_pid_list, train=False)
      args.train_class_cnt = train_dataset.class_cnt

  # Get model
  base_model = get_model(args)

  if args.cpu:
    logger.info("Note: CPU mode for K-means")
  logger.info("Model: {}".format(base_model.__class__.__name__))
  logger.info("Output dir: {}".format(save_dir))
  if not args.eval and not args.save_model:
    logger.info("NOT saving model!")

  if args.resume != "":
    #base_model.load_state_dict(torch.load(args.resume))
    base_model.load_state_dict(torch.load(args.resume), strict=False)
    logger.info("Resume: {}".format(args.resume))

  base_model.to(device)
  model = base_model  # XXX No data parallel

  if not args.eval:

    criterion = nn.CrossEntropyLoss().to(device)
    arcface = ArcMarginProduct(args.nb_dims, train_dataset.class_cnt, s=args.s, m=args.m).to(device)
    ntxent = NTXentLoss(device, args.batch_size, args.tau, args.track_per_class)

    optimizer = optim.Adam(base_model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))

    scheduler = MultiStepLR(optimizer, args.lr_step_list, 0.1)

    if args.method == "paired" and args.hard_mining:
      pass  # Initialized later
    else:
      train_sampler = BalancedBatchSampler(train_dataset.labels, args.batch_size // args.track_per_class, args.track_per_class)
      train_loader = DataLoader(train_dataset, batch_size=1 if train_sampler is not None else args.batch_size, num_workers=args.nb_workers, shuffle=(train_sampler is None), pin_memory=False, drop_last=True, persistent_workers=True, sampler=train_sampler, prefetch_factor=4)

    epoch_cnt, iter_cnt = 1, 0
    loss_elapsed = []

    model.train()
    torch.backends.cudnn.benchmark = True

    def freeze_backbone(model):
      for name, module in model.backbone.named_modules():
        if not args.freeze_layer4 and name.startswith("layer4"):
          continue
        if hasattr(module, 'weight'):
          module.weight.requires_grad_(False)
        if hasattr(module, 'bias'):
          if module.bias is None:
            continue
          module.bias.requires_grad_(False)
        module.eval()

    # Freeze backbone weights
    if args.freeze_backbone:
      freeze_backbone(base_model)

    st = time.time()
    while iter_cnt != args.nb_iters and (epoch_cnt == 1 or optimizer.param_groups[0]['lr'] > args.min_lr):

      if epoch_cnt % 10 == 0:
        logger.info("Epoch {}".format(epoch_cnt))

      # Hard class mining
      if args.method == "paired" and args.hard_mining:
        # XXX Only supports nb_samples=1 so far
        mine_sampler = ClassUniformSampler(train_dataset.labels, 1, args.batch_size)
        mine_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.nb_workers, shuffle=False, pin_memory=False, persistent_workers=True, sampler=mine_sampler)
        model.eval()

        all_features, all_labels = [], []
        with torch.no_grad():
          for cnt, batch in enumerate(mine_loader):
            x = batch["imgs"].to(device, non_blocking=True)[0]
            batch_idxs = batch["batch_idxs"].to(device, non_blocking=True)[0]
            labels = batch["labels"].to(device, non_blocking=True)[0]

            optimizer.zero_grad()
            out_dict = model.forward(x, batch_idxs)
            all_features.append(out_dict["features"].cpu().numpy())
            all_labels.append(batch["labels"][0])

          N_trk = sum([len(x) for x in all_features])
          sim_matrix = np.zeros((N_trk, N_trk), dtype=np.float32)

          offset1 = 0
          for idx1, feature1 in enumerate(all_features):
            s1 = len(feature1)
            offset2 = 0
            for idx2, feature2 in enumerate(all_features):
              s2 = len(feature2)
              sim_matrix[offset1:offset1+s1, offset2:offset2+s2] = np.sum(feature1[:, None] * feature2[None], axis=2)
              offset2 += s2
            offset1 += s1

          class_labels = np.concatenate(all_labels)
          class_sims = sim_matrix
        torch.cuda.empty_cache()

        model.train()
        if args.freeze_backbone:
          freeze_backbone(base_model)

        # Create sampler
        train_sampler = HardClassMiningBatchSampler(train_dataset.labels, class_labels, class_sims, args.batch_size // args.track_per_class, args.track_per_class, ratio=1.0)
        train_loader = DataLoader(train_dataset, batch_size=1 if train_sampler is not None else args.batch_size, num_workers=args.nb_workers, shuffle=(train_sampler is None), pin_memory=False, drop_last=True, persistent_workers=True, sampler=train_sampler, prefetch_factor=4)

      for cnt, batch in enumerate(train_loader):
        if iter_cnt == args.nb_iters:
          break

        x = batch["imgs"].to(device, non_blocking=True)[0]
        batch_idxs = batch["batch_idxs"].to(device, non_blocking=True)[0]
        labels = batch["labels"].to(device, non_blocking=True)[0]

        optimizer.zero_grad()
        if args.method == "softmax":
          out_dict = model(x, batch_idxs)
          softmax_loss = criterion(out_dict["logits"], labels)
          loss = softmax_loss
        elif args.method == "arcface":
          out_dict = model(x, batch_idxs)
          output = arcface(out_dict["features"], labels)
          arcface_loss = criterion(output, labels)
          loss = arcface_loss
        elif args.method == "paired":
          out_dict = model(x, batch_idxs, labels)

          # Calculate similarity
          features = out_dict["features"]
          B = len(features)

          f1 = torch.tile(features.unsqueeze(0), (B, 1, 1)).view(B*B, -1)
          f2 = torch.tile(features.unsqueeze(1), (1, B, 1)).view(B*B, -1)

          # Dot product
          sims = torch.sum(f1 * f2, dim=1).view(B, B)

          loss = ntxent(sims, labels)
        else:
          raise NotImplementedError()

        # Debug
        if args.debug and iter_cnt % 10 == 0:
          print(loss.item())

        # Backprop
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_elapsed.append(loss.item())

        # Log
        if (iter_cnt + 1) % args.iter_display == 0:
          logger.info("Iter {}: {}, {} iter / s".format(iter_cnt + 1, np.mean(loss_elapsed), args.iter_display / (time.time() - st)))
          loss_elapsed = []
          st = time.time()

        if (iter_cnt + 1) % args.iter_snapshot == 0 and args.save_model:
          model_path = osp.join(save_dir, "model_{:06d}.pth".format(iter_cnt + 1))
          logger.info("Checkpoint: {}".format(model_path))
          torch.save(base_model.state_dict(), model_path)

        if args.iter_evaluation != -1 and (iter_cnt + 1) % args.iter_evaluation == 0:
          logger.info("Validation...")
          eval_net(args, device, save_dir, model, valid_dataset, logger)

          if args.freeze_backbone:
            freeze_backbone(base_model)
          st = time.time()

        iter_cnt += 1

      epoch_cnt += 1

  else:  # Evaluation
    if args.grad_cam:
      eval_cam(args, device, save_dir, model, train_dataset, valid_dataset[0], logger)
    else:
      eval_net(args, device, save_dir, model, valid_dataset, logger)
  logger.info("Done. Elapsed time: {} (s)".format(time.time()-start_time))


if __name__ == "__main__":
  main()
